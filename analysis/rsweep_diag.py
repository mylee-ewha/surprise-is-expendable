"""
rsweep_diag.py — r 스윕으로 circular 감지 가능성 검증
====================================================
NI core(r=32)는 건드리지 않고, detector용 projection을 여러 r로 병렬 계산.
같은 inference 한 번에 r ∈ {32, 64, 128, 256} 전부 기록.

Usage:
  python analysis/rsweep_diag.py            # inference + 저장
  python analysis/rsweep_diag.py --analyze  # 저장된 npz 분석만
"""
import os, sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.core.cache_ops import BlockRegistry, evict_from_cache, RECENT_SIZE
from src.datasets.math500 import load_fixed_subset, build_prompt

# ═══════════════════════════════════════════════════════════
GPU_ID       = "6"
MODEL_NAME   = "Qwen/Qwen3-8B"
R_LIST       = [32, 64, 128, 256]   # detector용 스윕
NI_R         = 32                    # NI core (고정, evict 결정용)
LAMBDA_RIDGE = 1.0
REFRESH      = 256
EVICT_INTERVAL = 128
N_LAYERS     = 36
GEN_SEED     = 42
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20
MAX_NEW_TOKENS = 8192
BUDGET       = 512

# 검증 대상: circular 3 + normal 3 (동일 샘플 재사용)
TARGETS = [
    {"label": "circ_idx328", "idx": 328, "kind": "circ"},
    {"label": "circ_idx338", "idx": 338, "kind": "circ"},
    {"label": "circ_idx469", "idx": 469, "kind": "circ"},
    {"label": "norm_idx4",   "idx": 4,   "kind": "norm"},
    {"label": "norm_idx5",   "idx": 5,   "kind": "norm"},
    {"label": "norm_idx15",  "idx": 15,  "kind": "norm"},
]

LOG_DIR = Path("rsweep_logs")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


# ═══════════════════════════════════════════════════════════
class LeverageScorer:
    def __init__(self, dim, device):
        self.t = 0
        self.A     = LAMBDA_RIDGE * torch.eye(dim, dtype=torch.float64, device=device)
        self.A_inv = (1/LAMBDA_RIDGE) * torch.eye(dim, dtype=torch.float64, device=device)

    def update(self, v):
        v64 = v.double()
        Av  = self.A_inv @ v64
        s   = (v64 @ Av).item()
        den = 1.0 + s
        self.A     += torch.outer(v64, v64)
        self.A_inv -= torch.outer(Av, Av) / den
        self.t += 1
        if self.t % REFRESH == 0:
            self.A_inv = torch.linalg.inv(self.A)
        return s


class MultiRScorer:
    """
    NI core(r=32)와 detector(여러 r)를 동시에 굴린다.
    evict 결정은 NI core 값만 사용 → 기존 동작 완전 동일.
    """
    def __init__(self, R_projs, device):
        self.R_projs = R_projs           # {r: tensor [r, v_dim]}
        self.device  = device
        self.scorers = {r: {} for r in R_projs}   # r → {layer: LeverageScorer}
        self.logs    = {r: [] for r in R_projs}   # r → [S_t, ...]

    def score(self, v_storage) -> float:
        per_r = {}
        for r, R in self.R_projs.items():
            vals = []
            for li in range(1, N_LAYERS + 1):
                if li not in v_storage:
                    continue
                v_vec  = v_storage[li][0, 0].float()
                v_proj = R @ v_vec
                if li not in self.scorers[r]:
                    self.scorers[r][li] = LeverageScorer(r, self.device)
                vals.append(self.scorers[r][li].update(v_proj))
            m = float(np.mean(vals)) if vals else float("nan")
            per_r[r] = m
            self.logs[r].append(m)
        return per_r[NI_R]     # evict 결정은 r=32만


def sample_next_token(logits):
    logits = logits.float() / TEMPERATURE
    if TOP_K > 0:
        kth = torch.topk(logits, min(TOP_K, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if TOP_P < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = torch.cumsum(torch.softmax(sl, -1), -1)
        m = cp > TOP_P; m[..., 1:] = m[..., :-1].clone(); m[..., 0] = False
        sl = sl.masked_fill(m, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(torch.softmax(logits, -1), 1)


def register_v_hook(model):
    storage, handles = {}, []
    def mk(li):
        def h(mod, inp, out): storage[li] = out.detach()
        return h
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(mk(i + 1)))
    return handles, storage


@torch.no_grad()
def run_one(model, tokenizer, prompt, budget, device, R_projs):
    inputs     = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    handles, v_storage = register_v_hook(model)
    try:
        out   = model(inputs.input_ids, use_cache=True, past_key_values=DynamicCache())
        cache = out.past_key_values
        nl    = out.logits[:, -1, :]

        reg = BlockRegistry(); reg.add_tokens_batch(prompt_len)
        scorer = MultiRScorer(R_projs, device)

        think_scores, think_gen = [], 0
        in_think, gen_ids, buf = True, [], ""
        torch.manual_seed(GEN_SEED)
        nt = sample_next_token(nl); gen_ids.append(nt.item())

        for _ in range(MAX_NEW_TOKENS):
            if nt.item() == tokenizer.eos_token_id:
                break
            pid = torch.tensor([[reg.get_next_position_id()]], dtype=torch.long, device=device)
            so  = model(nt, use_cache=True, past_key_values=cache, position_ids=pid)
            cache, nl = so.past_key_values, so.logits[:, -1, :]
            reg.register_new_token()

            if in_think:
                s = scorer.score(v_storage)      # 모든 r 동시 기록
                think_scores.append(s)
                think_gen += 1

            buf += tokenizer.decode([nt.item()], skip_special_tokens=True)
            if in_think and "</think>" in buf:
                in_think = False

            if len(think_scores) > budget and (
                think_gen % EVICT_INTERVAL == 0 or
                len(think_scores) >= budget + EVICT_INTERVAL
            ):
                n_ev  = len(think_scores) - budget
                ncand = len(think_scores) - RECENT_SIZE
                if ncand > 0:
                    valid = [(j, sc) for j, sc in enumerate(think_scores[:ncand])
                             if not np.isnan(sc)]
                    if valid:
                        na = min(n_ev, len(valid))
                        sv = sorted(valid, key=lambda p: p[1], reverse=True)
                        ejs = sorted([j for j, _ in sv[:na]], reverse=True)
                        eset = {prompt_len + j for j in ejs}
                        keep = [i for i in range(cache.get_seq_length()) if i not in eset]
                        evict_from_cache(cache, keep)
                        reg.evict_by_cache_indices([prompt_len + j for j in ejs])
                        for j in ejs:
                            del think_scores[j]

            nt = sample_next_token(nl); gen_ids.append(nt.item())
    finally:
        for h in handles: h.remove()

    return {
        "logs":      {r: np.array(v, dtype=np.float32) for r, v in scorer.logs.items()},
        "T":         think_gen,
        "truncated": in_think,
    }


# ═══════════════════════════════════════════════════════════
def do_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device).eval()

    v_dim = model.model.layers[0].self_attn.v_proj.out_features
    R_projs = {}
    for r in R_LIST:
        g = torch.Generator(device=device).manual_seed(42 if r == NI_R else 1000 + r)
        R_projs[r] = (torch.randint(0, 2, (r, v_dim), generator=g,
                                    device=device).float() * 2 - 1) * (r ** -0.5)
    print(f"v_dim={v_dim}, R: " + ", ".join(f"r={r}{' (NI core)' if r==NI_R else ''}"
                                            for r in R_LIST))

    ds = load_fixed_subset(n=500, seed=42)
    LOG_DIR.mkdir(exist_ok=True)

    for t in tqdm(TARGETS, desc="rsweep"):
        p = LOG_DIR / f"{t['label']}.npz"
        if p.exists():
            print(f"  [skip] {t['label']}"); continue
        prompt = build_prompt(tok, ds[t["idx"]]["problem"])
        print(f"  [run ] {t['label']}")
        res = run_one(model, tok, prompt, BUDGET, device, R_projs)
        np.savez(p, **{f"r{r}": v for r, v in res["logs"].items()},
                 kind=t["kind"], T=res["T"], truncated=res["truncated"])
        print(f"         T={res['T']}, trunc={res['truncated']}")


# ═══════════════════════════════════════════════════════════
def analyze():
    from scipy import stats
    files = sorted(LOG_DIR.glob("*.npz"))
    if not files:
        print("npz 없음"); return
    data = {f.stem: np.load(f) for f in files}

    print("=" * 72)
    print("판정: 같은 위치 구간에서 circular vs normal 이 구분되는가")
    print("=" * 72)
    # normal 샘플 길이에 맞춰 겹치는 구간만
    Tmin = min(int(d["T"]) for d in data.values() if str(d["kind"]) == "norm")
    lo, hi = 200, min(Tmin, 800)
    print(f"비교 구간: position {lo}~{hi}  (자연 감소 confound 제거)\n")

    print(f"  {'r':<6} {'circ mean':<12} {'norm mean':<12} {'차이%':<10} {'p-value':<10} 판정")
    print("  " + "-" * 62)
    verdict = {}
    for r in R_LIST:
        c = np.concatenate([np.log1p(d[f"r{r}"][lo:hi]) for d in data.values()
                            if str(d["kind"]) == "circ"])
        n = np.concatenate([np.log1p(d[f"r{r}"][lo:hi]) for d in data.values()
                            if str(d["kind"]) == "norm"])
        c, n = c[np.isfinite(c)], n[np.isfinite(n)]
        _, p = stats.mannwhitneyu(c, n)
        diff = (c.mean() - n.mean()) / n.mean() * 100
        ok = p < 0.05 and abs(diff) > 10
        verdict[r] = ok
        print(f"  {r:<6} {c.mean():<12.6f} {n.mean():<12.6f} "
              f"{diff:>+7.1f}%  {p:<10.4f} {'✅ 구분됨' if ok else '❌'}")

    print()
    print("=" * 72)
    if any(verdict.values()):
        best = [r for r, v in verdict.items() if v]
        print(f"✅ r={best} 에서 신호 살아남 → detector로 채택 가능")
        print(f"   NI core는 r=32 유지, detector만 r={max(best)} 로 병렬 운용")
    else:
        print("❌ 모든 r에서 구분 불가 → det 방향 기각, n-gram(플랜 B)로 전환")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        do_inference()
    analyze()