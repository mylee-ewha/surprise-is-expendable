"""
vtilde_analysis.py — Ṽ=R·V 분포 분석 및 S_t 궤적 시각화
==========================================================
novelty_inv의 per-token 동작을 직접 확인.

저장하는 것:
  score_log    [T]      : 원래 생성 순서대로의 S_t
  v_tildes     [T, 32]  : 원래 생성 순서대로의 Ṽ_t (layer mean)
  evict_pos    [E]      : evict된 원래 think 토큰 positions (0-indexed)

Usage:
  python vtilde_analysis.py           # inference 실행 → 저장 → 플롯
  python vtilde_analysis.py --plot    # 저장된 .npz로 플롯만 재생성
"""

import os, json, random, re, sys, argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.core.cache_ops import BlockRegistry, evict_from_cache, RECENT_SIZE

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — 이 섹션만 수정
# ═══════════════════════════════════════════════════════════════════════════
GPU_ID          = "4"
MODEL_NAME      = "Qwen/Qwen3-8B"
NOVELTY_K       = 32
LAMBDA_RIDGE    = 1.0
REFRESH_INTERVAL = 256
EVICT_INTERVAL  = 128
N_LAYERS        = 36
GEN_SEED        = 42
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20
MAX_NEW_TOKENS  = 8192

# per-sample 결과 파일 경로 (실험 디렉토리에 맞게 수정)
MATH500_PER_SAMPLE = Path("results/scenario_b_ablation/results_per_sample.jsonl")
GPQA_PER_SAMPLE    = Path("results/gpqa_ablation/results_per_sample.jsonl")

LOG_DIR  = Path("vtilde_logs")
PLOT_DIR = Path("vtilde_plots")

# ── Manual 타겟 지정 ──────────────────────────────────────────
# idx를 이미 알고 있으면 여기 추가. 없으면 auto-detect에서 채워줌.
# category: 'A' | 'B' | 'C' | 'D' (색상 및 제목 구분용)
MANUAL_TARGETS = [
    {"label": "gpqa_C_idx112", "dataset": "gpqa",
     "idx": 112, "budget": 1024, "category": "C"},
    # {"label": "math_A_idx5",  "dataset": "math500",
    #  "idx": 5,   "budget": 2048, "category": "A"},
]

# True: MANUAL_TARGETS만 실행  /  False: auto-detect 결과와 합침
MANUAL_ONLY = False

# auto-detect: 카테고리(A/B/C/D)당 최대 몇 개 선정
N_AUTO_PER_CAT = 1

# ── GPQA 데이터셋 정보 (기존 GPQA inference 코드 참고해서 확인) ──
GPQA_DATASET_NAME   = "Idavidrein/gpqa"   # ⚠️ TODO: 확인
GPQA_DATASET_CONFIG = "gpqa_diamond"       # ⚠️ TODO: 확인
GPQA_SPLIT          = "train"              # ⚠️ TODO: 확인

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ═══════════════════════════════════════════════════════════════════════════
# R_PROJ 전역 (main에서 초기화, 원본과 동일한 seed=42)
# ═══════════════════════════════════════════════════════════════════════════
R_PROJ: torch.Tensor = None


# ═══════════════════════════════════════════════════════════════════════════
# 데이터셋 로딩
# ═══════════════════════════════════════════════════════════════════════════
_ds_cache = {}

def _load_and_cache(key, loader_fn):
    if key not in _ds_cache:
        _ds_cache[key] = loader_fn()
    return _ds_cache[key]

def load_math500(n=100, seed=42):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    return ds.select(idx[:n])

def load_gpqa():
    return load_dataset(GPQA_DATASET_NAME, GPQA_DATASET_CONFIG, split=GPQA_SPLIT)

def get_dataset(dtype):
    return _load_and_cache(
        dtype, load_math500 if dtype == "math500" else load_gpqa
    )

def get_question(ex, dtype):
    if dtype == "math500": return ex["problem"]
    if dtype == "gpqa":    return ex["Question"]
    raise ValueError(f"Unknown dataset: {dtype}")

def build_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# per-sample JSONL에서 타겟 자동 선정
# ═══════════════════════════════════════════════════════════════════════════
def auto_detect_targets(jsonl_path, dtype, n_per_cat=1):
    """
    카테고리별 타겟 선정:
      A: ni@2048 correct+complete, baseline correct          → "잘 작동하는 케이스"
      B: baseline truncated → ni@4096 correct (rescue)      → "truncation 구출 케이스"
      C: ni_trunc / kn_complete @ 1024                      → "ni 실패 케이스"
      D: ni+kn 둘 다 complete+correct @ 2048, baseline ok   → "정상 baseline"
    """
    if not Path(jsonl_path).exists():
        print(f"  [warn] {jsonl_path} 없음 — auto-detect 스킵")
        return []

    data = defaultdict(dict)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                data[r["idx"]][(r["method"], r["budget"])] = r
            except (json.JSONDecodeError, KeyError):
                continue

    pfx = "math" if dtype == "math500" else "gpqa"
    result, counts = [], defaultdict(int)

    for idx, d in sorted(data.items()):
        ni2 = d.get(("novelty_inv", 2048), {})
        ni4 = d.get(("novelty_inv", 4096), {})
        ni1 = d.get(("novelty_inv", 1024), {})
        kn2 = d.get(("k_norm",      2048), {})
        kn1 = d.get(("k_norm",      1024), {})
        bl  = d.get(("baseline",    None),  {})

        if (counts["A"] < n_per_cat and
                ni2.get("correct") and not ni2.get("truncated") and bl.get("correct")):
            result.append({"label": f"{pfx}_A_idx{idx}", "dataset": dtype,
                           "idx": idx, "budget": 2048, "category": "A"})
            counts["A"] += 1

        if (counts["B"] < n_per_cat and
                bl.get("truncated") and ni4.get("correct")):
            result.append({"label": f"{pfx}_B_idx{idx}", "dataset": dtype,
                           "idx": idx, "budget": 4096, "category": "B"})
            counts["B"] += 1

        if (counts["C"] < n_per_cat and
                ni1.get("truncated") and not kn1.get("truncated") and kn1.get("correct")):
            result.append({"label": f"{pfx}_C_idx{idx}", "dataset": dtype,
                           "idx": idx, "budget": 1024, "category": "C"})
            counts["C"] += 1

        if (counts["D"] < n_per_cat and
                not ni2.get("truncated") and ni2.get("correct") and
                not kn2.get("truncated") and kn2.get("correct") and bl.get("correct")):
            result.append({"label": f"{pfx}_D_idx{idx}", "dataset": dtype,
                           "idx": idx, "budget": 2048, "category": "D"})
            counts["D"] += 1

    print(f"  [auto] {dtype}: {len(result)}개 선정 — {dict(counts)}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 모델 유틸 (원본과 동일)
# ═══════════════════════════════════════════════════════════════════════════
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
    def make_hook(li):
        def h(mod, inp, out): storage[li] = out.detach()
        return h
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i + 1)))
    return handles, storage


# ═══════════════════════════════════════════════════════════════════════════
# Leverage scorer (원본과 동일)
# ═══════════════════════════════════════════════════════════════════════════
class CausalLeverageScorer:
    def __init__(self, dim, device):
        self.t     = 0
        self.A     = LAMBDA_RIDGE * torch.eye(dim, dtype=torch.float64, device=device)
        self.A_inv = (1.0 / LAMBDA_RIDGE) * torch.eye(dim, dtype=torch.float64, device=device)

    def update(self, v: torch.Tensor) -> float:
        v64  = v.double()
        Av   = self.A_inv @ v64
        den  = 1.0 + v64 @ Av
        s    = (v64 @ Av).item()          # S_t = ṽ^T A^{-1}_{t-1} ṽ (A 업데이트 전)
        self.A     += torch.outer(v64, v64)
        self.A_inv -= torch.outer(Av, Av) / den
        self.t += 1
        if self.t % REFRESH_INTERVAL == 0:
            self.A_inv = torch.linalg.inv(self.A)
        return s


# ═══════════════════════════════════════════════════════════════════════════
# 로깅 포함 스코어러
# (score_novelty를 대체 — A 업데이트는 한 번만, 동시에 Ṽ_t 기록)
# ═══════════════════════════════════════════════════════════════════════════
class NoveltyInvScorerLog:
    def __init__(self, device):
        self.device  = device
        self.scorers = {}   # layer_idx → CausalLeverageScorer
        # 아래 두 리스트가 핵심 출력물
        self.score_log   = []   # S_t per think token (원래 생성 순서)
        self.v_tilde_log = []   # Ṽ_t per think token [T, NOVELTY_K]

    def score_and_log(self, v_storage) -> float:
        """
        v_storage: {layer_idx: tensor [1, n_heads, seq, v_dim/n_heads]}
        원본 score_novelty와 완전히 동일한 연산 + Ṽ_t 기록.
        반환값은 기존과 동일하게 think_scores에 append할 scalar.
        """
        scores, v_tildes = [], []
        for li in range(1, N_LAYERS + 1):
            if li not in v_storage:
                continue
            v_vec  = v_storage[li][0, 0].float()   # [v_dim]
            v_proj = R_PROJ @ v_vec                  # Ṽ_t ∈ ℝ^{32}
            if li not in self.scorers:
                self.scorers[li] = CausalLeverageScorer(NOVELTY_K, self.device)
            scores.append(self.scorers[li].update(v_proj))
            v_tildes.append(v_proj.detach().cpu().float().numpy())

        mean_score  = float(np.mean(scores))           if scores  else float("nan")
        mean_vtilde = np.mean(v_tildes, axis=0).astype(np.float32) \
                      if v_tildes else np.zeros(NOVELTY_K, dtype=np.float32)

        self.score_log.append(mean_score)
        self.v_tilde_log.append(mean_vtilde)
        return mean_score


# ═══════════════════════════════════════════════════════════════════════════
# Generation with V logging (novelty_inv 전용)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_with_v_logging(model, tokenizer, prompt, budget, device):
    """
    원본 generate_with_scored_eviction (novelty_inv) + 로깅.

    핵심 추가:
      - think_orig_pos: alive think 토큰의 원래 position 추적
        eviction으로 think_scores가 줄어도 원래 몇 번째 토큰인지 알 수 있음
      - score_log / v_tilde_log: 생성 즉시 append (eviction 영향 없음)
      - evict_pos_log: evict 직전에 think_orig_pos에서 읽어서 기록
    """
    inputs     = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids  = inputs.input_ids
    prompt_len = input_ids.shape[1]

    handles, v_storage = register_v_hook(model)
    try:
        prefill_out = model(input_ids, use_cache=True, past_key_values=DynamicCache())
        cache       = prefill_out.past_key_values
        next_logits = prefill_out.logits[:, -1, :]

        reg    = BlockRegistry()
        reg.add_tokens_batch(prompt_len)
        scorer = NoveltyInvScorerLog(device)

        # alive 토큰 추적 (원본의 think_scores + 추가된 think_orig_pos)
        think_scores   = []   # alive 토큰의 S_t (eviction 시 삭제됨)
        think_orig_pos = []   # alive 토큰의 원래 0-indexed position

        evict_pos_log = []    # evict된 원래 position들

        think_gen_count = 0   # 지금까지 생성된 think 토큰 수 (원본과 동일)
        n_evicted  = 0
        in_think   = True
        gen_ids    = []
        text_buf   = ""

        torch.manual_seed(GEN_SEED)
        next_token = sample_next_token(next_logits)
        gen_ids.append(next_token.item())

        for _ in range(MAX_NEW_TOKENS):
            if next_token.item() == tokenizer.eos_token_id:
                break

            pos_id = torch.tensor(
                [[reg.get_next_position_id()]], dtype=torch.long, device=device
            )
            step_out    = model(next_token, use_cache=True,
                                past_key_values=cache, position_ids=pos_id)
            cache       = step_out.past_key_values
            next_logits = step_out.logits[:, -1, :]
            reg.register_new_token()

            if in_think:
                # score_and_log: A 업데이트 + S_t/Ṽ_t 기록 (원본 score_novelty 대체)
                score = scorer.score_and_log(v_storage)

                think_scores.append(score)
                think_orig_pos.append(think_gen_count)   # 원래 position (0-indexed)
                think_gen_count += 1

            piece    = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            text_buf += piece
            if in_think and "</think>" in text_buf:
                in_think = False

            # ── Lazy eviction (원본과 동일, novelty_inv: highest 제거) ──
            should_evict = (
                len(think_scores) > budget and (
                    think_gen_count % EVICT_INTERVAL == 0 or
                    len(think_scores) >= budget + EVICT_INTERVAL
                )
            )
            if should_evict:
                n_to_evict = len(think_scores) - budget
                n_cand     = len(think_scores) - RECENT_SIZE
                if n_cand > 0:
                    valid = [
                        (j, s) for j, s in enumerate(think_scores[:n_cand])
                        if not np.isnan(s)
                    ]
                    if valid:
                        n_actual     = min(n_to_evict, len(valid))
                        sorted_valid = sorted(valid, key=lambda p: p[1], reverse=True)
                        evict_js     = sorted(
                            [j for j, _ in sorted_valid[:n_actual]], reverse=True
                        )

                        # ★ evict 전에 원래 position 기록
                        evict_pos_log.extend(think_orig_pos[j] for j in evict_js)

                        evict_set = {prompt_len + j for j in evict_js}
                        keep_list = [i for i in range(cache.get_seq_length())
                                     if i not in evict_set]
                        evict_from_cache(cache, keep_list)
                        reg.evict_by_cache_indices([prompt_len + j for j in evict_js])
                        for j in evict_js:
                            del think_scores[j]
                            del think_orig_pos[j]    # ★ 병렬 삭제
                        n_evicted += len(evict_js)

            next_token = sample_next_token(next_logits)
            gen_ids.append(next_token.item())

    finally:
        for h in handles: h.remove()

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "text":                   text,
        "truncated":              in_think,
        "think_tokens_generated": think_gen_count,
        "n_evicted":              n_evicted,
        # ── 로그 출력 ──
        "score_log":    np.array(scorer.score_log,    dtype=np.float32),   # [T]
        "v_tilde_log":  np.array(scorer.v_tilde_log,  dtype=np.float32),   # [T, 32]
        "evict_pos_log": np.array(evict_pos_log,       dtype=np.int32),     # [E]
    }

# vtilde_analysis.py에 format_mcq 추가
def format_mcq(ex, idx):
    correct = ex["Correct Answer"]
    wrongs  = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
    choices = [correct] + wrongs
    rng     = random.Random(42 + idx)
    rng.shuffle(choices)
    labels  = "ABCD"
    correct_label = labels[choices.index(correct)]
    choice_text   = "\n".join(f"({labels[i]}) {choices[i]}" for i in range(4))
    body = (
        f"{ex['Question']}\n\n{choice_text}\n\n"
        "Please reason step by step and select the single best answer. "
        "At the end of your response, write exactly: "
        "'The correct answer is (X).' where X is A, B, C, or D."
    )
    return body, correct_label

# get_question 대신 run_inference 안에서 직접 호출하도록 변경
def get_question(ex, dtype, idx=None):
    if dtype == "math500": return ex["problem"]
    if dtype == "gpqa":    return format_mcq(ex, idx)[0]  # body만 반환
    raise ValueError(dtype)

# ═══════════════════════════════════════════════════════════════════════════
# Inference 루프
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(targets, model, tokenizer, device):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for t in tqdm(targets, desc="inference"):
        npz_path = LOG_DIR / f"{t['label']}.npz"
        if npz_path.exists():
            print(f"  [skip] {t['label']} — 이미 완료")
            continue

        ds   = get_dataset(t["dataset"])
        q = get_question(ds[t["idx"]], t["dataset"], idx=t["idx"])
        prom = build_prompt(tokenizer, q)

        print(f"  [run ] {t['label']}  (idx={t['idx']}, budget={t['budget']})")
        res = generate_with_v_logging(model, tokenizer, prom, t["budget"], device)

        np.savez(
            npz_path,
            scores    = res["score_log"],
            v_tildes  = res["v_tilde_log"],
            evict_pos = res["evict_pos_log"],
        )
        meta = {**t,
                "think_tokens_generated": res["think_tokens_generated"],
                "n_evicted":  res["n_evicted"],
                "truncated":  res["truncated"]}
        (LOG_DIR / f"{t['label']}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"         T={res['think_tokens_generated']}, "
              f"evicted={res['n_evicted']}, truncated={res['truncated']}")


# ═══════════════════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════════════════
CAT_COLOR = {"A": "#1565C0", "B": "#2E7D32", "C": "#C62828", "D": "#6A1B9A"}
CAT_LABEL = {
    "A": "A: ni correct @2048",
    "B": "B: rescue (bl_trunc→ni correct) @4096",
    "C": "C: ni_trunc / kn_complete @1024",
    "D": "D: both complete+correct @2048",
}

def _rolling_mean(arr, w):
    if len(arr) < w:
        return arr, np.arange(len(arr))
    return np.convolve(arr, np.ones(w) / w, mode="valid"), np.arange(w - 1, len(arr))


def plot_single(npz_path, label, category):
    """
    3-panel 상세 플롯:
      패널 1: S_t 궤적 + rolling mean + eviction 마커
      패널 2: ||Ṽ_t||_2 궤적 (V 크기 변화)
      패널 3: Ṽ_t 히트맵 [32 dim × T position]
    """
    d         = np.load(npz_path)
    scores    = d["scores"].astype(float)   # [T]
    v_tildes  = d["v_tildes"]               # [T, 32]
    evict_pos = d["evict_pos"]              # [E]
    T         = len(scores)
    color     = CAT_COLOR.get(category, "steelblue")
    w         = max(30, T // 80)            # rolling window

    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.3], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── 패널 1: S_t ──────────────────────────────────────────────────────
    ax1.plot(range(T), scores, lw=0.4, alpha=0.3, color=color)
    sm, sx = _rolling_mean(scores, w)
    ax1.plot(sx, sm, lw=2.0, color=color, label=f"Rolling mean (w={w})")

    if len(evict_pos) > 0:
        vp = evict_pos[evict_pos < T]
        ax1.scatter(vp, scores[vp], s=6, c="orangered", alpha=0.3,
                    zorder=5, rasterized=True,
                    label=f"Evicted ({len(evict_pos)}개 / {len(evict_pos)/T*100:.0f}%)")
        ax1.axvline(int(evict_pos.min()), color="darkorange", ls="--",
                    lw=1.3, label="First eviction")

    ax1.set_ylabel("$S_t$ (novelty score)", fontsize=10)
    ax1.set_title(
        f"{label}    [{CAT_LABEL.get(category, category)}]    (T={T})",
        fontsize=11
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.tick_params(labelbottom=False)

    # ── 패널 2: ||Ṽ_t||_2 ───────────────────────────────────────────────
    l2 = np.linalg.norm(v_tildes, axis=1)
    ax2.plot(range(T), l2, lw=0.4, alpha=0.3, color=color)
    sm2, sx2 = _rolling_mean(l2, w)
    ax2.plot(sx2, sm2, lw=2.0, color=color)
    if len(evict_pos) > 0:
        ax2.axvline(int(evict_pos.min()), color="darkorange", ls="--", lw=1.3)
    ax2.set_ylabel(r"$\|\tilde{V}_t\|_2$", fontsize=10)
    ax2.tick_params(labelbottom=False)

    # ── 패널 3: Ṽ_t 히트맵 ─────────────────────────────────────────────
    vmax = float(np.percentile(np.abs(v_tildes), 95))
    im   = ax3.imshow(
        v_tildes.T, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear",
    )
    # eviction 위치를 반투명 세로선으로 표시 (밀집 구간이 진해져서 density 파악 가능)
    if len(evict_pos) > 0:
        for p in evict_pos[evict_pos < T]:
            ax3.axvline(int(p), color="orangered", lw=0.1, alpha=0.15)
    fig.colorbar(im, ax=ax3, shrink=0.85, label="$\\tilde{V}_t$ value")
    ax3.set_xlabel("Think token position", fontsize=10)
    ax3.set_ylabel("RP dim (0–31)", fontsize=9)
    ax3.set_yticks([0, 15, 31])
    ax3.set_xlim(0, T)

    out = PLOT_DIR / f"{label}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out}")


def plot_comparison(entries):
    """
    S_t 궤적 가로 비교 (1행 N열).
    MATH500 (A/B) vs GPQA (C/D) 패턴 차이가 한눈에 보임.
    """
    n = len(entries)
    if n < 2:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

    for i, (npz_path, label, cat) in enumerate(entries):
        d      = np.load(npz_path)
        scores = d["scores"].astype(float)
        evict  = d["evict_pos"]
        T      = len(scores)
        color  = CAT_COLOR.get(cat, "steelblue")
        w      = max(30, T // 80)
        ax     = axes[0, i]

        ax.plot(range(T), scores, lw=0.3, alpha=0.25, color=color)
        sm, sx = _rolling_mean(scores, w)
        ax.plot(sx, sm, lw=2.0, color=color)

        if len(evict) > 0:
            vp = evict[evict < T]
            ax.scatter(vp, scores[vp], s=4, c="orangered",
                       alpha=0.2, zorder=5, rasterized=True)
            ax.axvline(int(evict.min()), color="darkorange", ls="--", lw=1.0)

        ax.set_title(f"{CAT_LABEL.get(cat, label)}\n{label}", fontsize=8.5)
        ax.set_xlabel("Think token position", fontsize=9)
        if i == 0:
            ax.set_ylabel("$S_t$", fontsize=10)
        ax.set_xlim(0, T)

    plt.suptitle("novelty_inv  S_t 궤적 비교", fontsize=12, y=1.02)
    plt.tight_layout()
    out = PLOT_DIR / "comparison_trajectory.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out}")


def plot_score_distribution(entries):
    """S_t 분포 박스플롯 (카테고리별 variance 한눈에 비교)"""
    fig, ax = plt.subplots(figsize=(3 + 2.5 * len(entries), 5))
    data, xlabels, colors = [], [], []

    for npz_path, label, cat in entries:
        d = np.load(npz_path)
        s = d["scores"].astype(float)
        data.append(s[np.isfinite(s)])
        xlabels.append(f"{label}\n(T={len(s)}, evicted={len(d['evict_pos'])})")
        colors.append(CAT_COLOR.get(cat, "steelblue"))

    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("$S_t$ (novelty score)", fontsize=10)
    ax.set_title("S_t 분포 비교\n(MATH500 vs GPQA의 균질성 검증)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOT_DIR / "score_distribution.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out}")

def plot_microvariance_diagnostic(npz_path, label, category="?", convergence_cutoff=200):
    d = np.load(npz_path)
    scores    = d["scores"].astype(float)  # [T]
    evict_pos = set(d["evict_pos"].tolist())
    T = len(scores)

    # post-convergence 구간만
    post = np.array([
        (t, scores[t], t in evict_pos)
        for t in range(convergence_cutoff, T)
        if not np.isnan(scores[t])
    ], dtype=[('pos', int), ('score', float), ('evicted', bool)])

    evicted_scores = post['score'][post['evicted']]
    kept_scores    = post['score'][~post['evicted']]
    evicted_pos    = post['pos'][post['evicted']]
    kept_pos       = post['pos'][~post['evicted']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Zoomed S_t (log scale)
    ax = axes[0]
    ax.plot(post['pos'], post['score'], lw=0.3, alpha=0.4, color='gray')
    ax.scatter(evicted_pos, evicted_scores,
               s=3, c='orangered', alpha=0.3, label='evicted', rasterized=True)
    ax.set_yscale('log')
    ax.set_xlabel('Think token position')
    ax.set_ylabel('$S_t$ (log scale)')
    ax.set_title(f'Post-convergence S_t (log)\n{label}')
    ax.legend(fontsize=8)

    # Panel 2: S_t 분포 비교 (evicted vs kept)
    ax = axes[1]
    ax.hist(kept_scores,    bins=80, alpha=0.6, color='steelblue',
            label=f'Kept (n={len(kept_scores)})',    density=True)
    ax.hist(evicted_scores, bins=80, alpha=0.6, color='orangered',
            label=f'Evicted (n={len(evicted_scores)})', density=True)
    ax.set_xlabel('$S_t$ value')
    ax.set_ylabel('Density')
    ax.set_title('S_t 분포: evicted vs kept\n(micro-variance signal 유무 판별)')
    ax.legend(fontsize=8)

    # Panel 3: Position 분포 비교 (FIFO 가설 판별)
    ax = axes[2]
    ax.hist(kept_pos,    bins=80, alpha=0.6, color='steelblue',
            label='Kept',    density=True)
    ax.hist(evicted_pos, bins=80, alpha=0.6, color='orangered',
            label='Evicted', density=True)
    ax.set_xlabel('Original token position')
    ax.set_ylabel('Density')
    ax.set_title('Position 분포: evicted vs kept\n(FIFO 패턴 유무 판별)')
    ax.legend(fontsize=8)

    # 수치 요약
    print(f"\n[{label}] post-convergence (t>{convergence_cutoff}) 분석:")
    print(f"  evicted  S_t: mean={evicted_scores.mean():.6f}, "
          f"median={np.median(evicted_scores):.6f}, std={evicted_scores.std():.6f}")
    print(f"  kept     S_t: mean={kept_scores.mean():.6f}, "
          f"median={np.median(kept_scores):.6f}, std={kept_scores.std():.6f}")
    print(f"  evicted  pos: mean={evicted_pos.mean():.1f}")
    print(f"  kept     pos: mean={kept_pos.mean():.1f}")

    plt.suptitle(f'Micro-variance Diagnostic — {label}', fontsize=11)
    plt.tight_layout()
    out = PLOT_DIR / f"{label}_microvar.pdf"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] {out}")

def run_plots(targets):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for t in targets:
        p = LOG_DIR / f"{t['label']}.npz"
        if not p.exists():
            print(f"  [warn] {p} 없음 — 스킵")
            continue
        try:
            plot_single(p, t["label"], t.get("category", "?"))
            plot_microvariance_diagnostic(p, t["label"])
            entries.append((p, t["label"], t.get("category", "?")))
        except Exception as e:
            print(f"  [err ] {t['label']}: {e}")

    if len(entries) >= 2:
        plot_comparison(entries)
        plot_score_distribution(entries)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="저장된 .npz로 플롯만 재생성 (inference 생략)")
    args = parser.parse_args()

    # ── 타겟 수집 ──────────────────────────────────────────────────────────
    targets = list(MANUAL_TARGETS)
    if not MANUAL_ONLY:
        print("[auto-detect] MATH500 타겟 선정...")
        targets += auto_detect_targets(MATH500_PER_SAMPLE, "math500", N_AUTO_PER_CAT)
        print("[auto-detect] GPQA 타겟 선정...")
        targets += auto_detect_targets(GPQA_PER_SAMPLE, "gpqa", N_AUTO_PER_CAT)

    # 중복 label 제거
    seen = set()
    targets = [t for t in targets if t["label"] not in seen and not seen.add(t["label"])]

    print(f"\n총 {len(targets)}개 타겟:")
    for t in targets:
        print(f"  {t['label']:<35} dataset={t['dataset']:<8} "
              f"idx={t['idx']:<5} budget={t['budget']:<5} cat={t.get('category','?')}")

    if not args.plot:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[model] 로딩 중... ({device})")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
        )
        model.eval()

        v_dim = model.model.layers[0].self_attn.v_proj.out_features
        global R_PROJ
        R_PROJ = (
            torch.randint(0, 2, (NOVELTY_K, v_dim),
                          generator=torch.Generator(device=device).manual_seed(42),
                          device=device).float() * 2.0 - 1.0
        ) * (NOVELTY_K ** -0.5)
        print(f"  R_PROJ: {R_PROJ.shape}, v_dim={v_dim}")

        run_inference(targets, model, tokenizer, device)

    print("\n[plot] 시각화 생성...")
    run_plots(targets)
    print("Done.")


if __name__ == "__main__":
    main()