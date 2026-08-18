"""
AIME Ablation — 3 seeds × 60 problems
======================================
- Budget [2048, 4096, 8192, 16384] (MATH500 대비 오른쪽 스케일)
- MAX_NEW_TOKENS = 32768 (32K context 확보)
- baseline도 3 seeds 실행
- 각 (method, budget, seed) 단위로 results.jsonl에 저장 → resume 지원
- 모든 seed 완료 시 results_agg.jsonl에 mean ± std 저장
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.core.scorers as _scorers
from src.core.generation import generate_with_scored_eviction
from src.datasets.aime import load_aime, build_prompt
from src.utils.metrics import extract_aime_answer, is_correct_aime

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
GPU_ID            = "4"
#MODEL_NAME        = "Qwen/Qwen3-8B"
MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
KV_BUDGETS        = [2048, 4096, 8192, 16384]
#METHODS           = ["baseline", "novelty_inv", "lru", "random", "raas", "rkv", "knorm", "novelty"]
METHODS           = ["baseline"]
METHODS_SAVE_TEXT = {"baseline", "novelty_inv", "lru", "raas", "rkv", "knorm"}
SEEDS             = [42, 1234, 5678]          # 3 runs, pass@1 averaged
MAX_NEW_TOKENS_EXP = 32768                    # 32K context 기준 여유있게
#OUT_DIR           = Path("results/aime_ablation")
OUT_DIR           = Path("results/aime_ablation_deepseek")

ENABLE_THINKING   = "Qwen" in MODEL_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


# ═══════════════════════════════════════════════════════════════
# Resume helpers
# ═══════════════════════════════════════════════════════════════
def _load_completed_aime(out_path: Path) -> set:
    """완료된 (method, kv_budget, seed) 조합 반환."""
    completed = set()
    if not out_path.exists():
        return completed
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "seed" in row:    # per-seed row
                    completed.add((row["method"], row["kv_budget"], row["seed"]))
            except json.JSONDecodeError:
                pass
    return completed


def _load_completed_agg(agg_path: Path) -> set:
    """집계 완료된 (method, kv_budget) 조합 반환."""
    completed = set()
    if not agg_path.exists():
        return completed
    with open(agg_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add((row["method"], row["kv_budget"]))
            except json.JSONDecodeError:
                pass
    return completed


# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    attn_impl = "eager" if any(m in METHODS for m in ["raas", "rkv"]) else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()

    v_dim = model.model.layers[0].self_attn.v_proj.out_features
    _scorers.R_PROJ = (
        torch.randint(0, 2, (_scorers.NOVELTY_K, v_dim),
                      generator=torch.Generator(device=device).manual_seed(42),
                      device=device).float() * 2.0 - 1.0
    ) * (_scorers.NOVELTY_K ** -0.5)

    problems = load_aime()
    print(f"[AIME] {len(problems)}문제 로드 완료")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path         = OUT_DIR / "results.jsonl"         # per-seed raw
    agg_path         = OUT_DIR / "results_agg.jsonl"     # mean ± std
    per_sample_path  = OUT_DIR / "results_per_sample.jsonl"
    per_text_path    = OUT_DIR / "results_texts.jsonl"

    completed     = _load_completed_aime(out_path)
    completed_agg = _load_completed_agg(agg_path)

    if completed:
        print(f"[resume] {len(completed)}개 seed-run 스킵")

    with open(out_path, "a") as f_out, \
         open(agg_path, "a") as f_agg:

        for method in METHODS:
            budgets = [None] if method == "baseline" else KV_BUDGETS

            for budget in budgets:
                # ── 각 seed 실행 ───────────────────────────────────────
                seed_accs = []
                for seed in SEEDS:
                    key = (method, budget, seed)
                    if key in completed:
                        # 이미 완료된 seed — acc 복원해서 집계에 포함
                        acc = _read_seed_acc(out_path, method, budget, seed)
                        if acc is not None:
                            seed_accs.append(acc)
                        print(f"[skip] {method} budget={budget} seed={seed}")
                        continue

                    # generation seed 고정
                    if seed is not None:
                        import src.core.generation as _gen_mod
                        _gen_mod.GEN_SEED = seed

                    correct = total_think = total_evicted = 0
                    n_truncated = total_total = n_triggered = 0
                    per_sample_rows, per_text_rows = [], []

                    label = f"{method}" + (f" budget={budget}" if budget else "") \
                                        + (f" seed={seed}" if seed else "")
                    pbar  = tqdm(problems, desc=label)

                    for idx, ex in enumerate(pbar):
                        prompt = build_prompt(
                            tokenizer, ex["problem"], enable_thinking=ENABLE_THINKING
                        )
                        gen = generate_with_scored_eviction(
                            model, tokenizer, prompt, method,
                            budget or 0, device,
                            max_new_tokens=MAX_NEW_TOKENS_EXP,
                        )

                        pred = extract_aime_answer(gen["text"])
                        ok   = is_correct_aime(pred, ex["answer"])

                        correct     += ok
                        total_think += gen["think_tokens_generated"]
                        total_total += gen["total_tokens_generated"]
                        n_truncated += int(gen["truncated"])

                        per_sample_rows.append({
                            "idx":                    idx,
                            "method":                 method,
                            "budget":                 budget,
                            "seed":                   seed,
                            "think_tokens_generated": gen["think_tokens_generated"],
                            "total_tokens_generated": gen["total_tokens_generated"],
                            "truncated":              gen["truncated"],
                            "correct":                ok,
                            "pred":                   pred,
                            "gold":                   ex["answer"],
                        })
                        per_text_rows.append(
                            gen["text"].split("</think>")[0]
                            if "</think>" in gen["text"] else gen["text"]
                        )

                        postfix = {"acc": f"{correct / (idx + 1):.1%}"}
                        if method != "baseline":
                            n_evicted = gen.get("n_evicted", 0)
                            total_evicted += n_evicted
                            if n_evicted > 0:
                                n_triggered += 1
                            postfix["evicted"] = \
                                f"{total_evicted / max(total_think, 1):.1%}"
                        pbar.set_postfix(**postfix)

                    n   = len(problems)
                    acc = correct / n
                    seed_accs.append(acc)

                    row = {
                        "method":                       method,
                        "kv_budget":                    budget,
                        "seed":                         seed,
                        "n":                            n,
                        "acc":                          acc,
                        "avg_think_tokens_generated":   total_think / n,
                        "avg_total_tokens_generated":   total_total / n,
                        "avg_answer_tokens_generated":  (total_total - total_think) / n,
                        "frac_truncated":               n_truncated / n,
                    }
                    if method != "baseline":
                        row["avg_n_evicted"]                   = total_evicted / n
                        row["achieved_eviction_ratio"]         = \
                            total_evicted / max(total_think, 1)
                        row["frac_samples_eviction_triggered"] = n_triggered / n

                    f_out.write(json.dumps(row) + "\n")
                    f_out.flush()

                    # per-sample
                    with open(per_sample_path, "a") as pf:
                        for r in per_sample_rows:
                            pf.write(json.dumps(r) + "\n")

                    # text
                    if method in METHODS_SAVE_TEXT:
                        with open(per_text_path, "a") as tf:
                            for r, text in zip(per_sample_rows, per_text_rows):
                                tf.write(json.dumps({
                                    "idx":       r["idx"],
                                    "method":    r["method"],
                                    "budget":    r["budget"],
                                    "seed":      r["seed"],
                                    "truncated": r["truncated"],
                                    "correct":   r["correct"],
                                    "pred":      r["pred"],
                                    "gold":      r["gold"],
                                    "text":      text,
                                }) + "\n")

                    print(row, flush=True)

                # ── 모든 seed 완료 시 집계 ─────────────────────────────
                if (method, budget) in completed_agg:
                    print(f"[skip agg] {method} budget={budget}")
                    continue

                if len(seed_accs) == len(SEEDS):
                    agg_row = {
                        "method":    method,
                        "kv_budget": budget,
                        "seeds":     SEEDS,
                        "n_runs":    len(seed_accs),
                        "mean_acc":  float(np.mean(seed_accs)),
                        "std_acc":   float(np.std(seed_accs)),
                        "accs":      seed_accs,
                    }
                    f_agg.write(json.dumps(agg_row) + "\n")
                    f_agg.flush()
                    print(f"[AGG] {method} budget={budget} "
                          f"mean={agg_row['mean_acc']:.3f} "
                          f"std={agg_row['std_acc']:.3f}")


def _read_seed_acc(out_path: Path, method, budget, seed) -> float | None:
    """완료된 seed row에서 acc 복원 (집계용)."""
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if (row.get("method") == method
                        and row.get("kv_budget") == budget
                        and row.get("seed") == seed
                        and "seed" in row):
                    return row["acc"]
            except json.JSONDecodeError:
                pass
    return None


if __name__ == "__main__":
    run()