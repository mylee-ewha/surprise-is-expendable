"""
analysis/r_ablation.py
r (projection dimension) ablation for NI eviction
r ∈ {8, 16, 32, 64, 128} — MATH500 Qwen3-8B, B=2048 고정

Usage:
  python analysis/r_ablation.py            # full run (N=50)
  python analysis/r_ablation.py --quick    # quick check (N=20)
"""

import os, sys, json, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.core.scorers as _scorers
from src.core.generation import generate_with_scored_eviction
from src.datasets.math500 import load_fixed_subset

# ── CONFIG ────────────────────────────────────────────────────────────────
GPU_ID     = "6"
MODEL_NAME = "Qwen/Qwen3-8B"
BUDGET     = 2048
N_SAMPLES  = 50       # --quick 시 20으로 override
SEED       = 42
R_VALUES   = [8, 16, 32, 64, 128]
OUT_PATH   = Path("results/r_ablation.json")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoModelForCausalLM, AutoTokenizer


def init_r_proj(r: int, v_dim: int, device: str):
    """R_PROJ와 NOVELTY_K를 scorers 모듈 전역에 주입."""
    _scorers.NOVELTY_K = r
    _scorers.R_PROJ = (
        torch.randint(
            0, 2, (r, v_dim),
            generator=torch.Generator(device=device).manual_seed(42),
            device=device,
        ).float() * 2.0 - 1.0
    ) * (r ** -0.5)
    print(f"  R_PROJ: {_scorers.R_PROJ.shape}  "
          f"(compression {v_dim // r}×)")


def run(n_samples: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map=device, attn_implementation="sdpa")
    model.eval()

    v_dim = model.model.layers[0].self_attn.v_proj.out_features

    ds      = load_fixed_subset(n=500, seed=SEED)
    idxs    = list(range(n_samples))
    mk_pmt  = lambda q: tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompts = [mk_pmt(ds[i]["problem"]) for i in idxs]
    answers = [ds[i]["answer"]           for i in idxs]

    results = {}

    for r in R_VALUES:
        print(f"\n{'='*52}")
        print(f"  r = {r}")
        print(f"{'='*52}")

        init_r_proj(r, v_dim, device)

        correct, truncated, evrs = 0, 0, []

        for i, (prompt, answer) in enumerate(
                tqdm(zip(prompts, answers),
                     total=n_samples, desc=f"r={r}")):

            with torch.no_grad():
                res = generate_with_scored_eviction(
                    model, tokenizer, prompt,
                    method="novelty_inv", budget=BUDGET, device=device)

            pred = res.get("pred", "")
            correct   += int(str(pred).strip() == str(answer).strip())
            truncated += int(res["truncated"])

            T = res["think_tokens_generated"]
            if T > 0:
                evrs.append(res["n_evicted"] / T)

            if (i + 1) % 10 == 0:
                print(f"    [{i+1:3d}/{n_samples}]  "
                      f"acc={correct/(i+1):.3f}  "
                      f"trunc={truncated/(i+1):.3f}")

        results[r] = {
            "r":              r,
            "accuracy":       round(correct / n_samples, 4),
            "frac_truncated": round(truncated / n_samples, 4),
            "eviction_ratio": round(float(np.mean(evrs)), 4),
            "n_samples":      n_samples,
        }
        print(f"  → acc={results[r]['accuracy']:.4f}  "
              f"trunc={results[r]['frac_truncated']:.4f}")

    # ── 저장 ─────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_PATH}")

    # ── 콘솔 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'r':>6} {'Accuracy':>10} {'Frac_Trunc':>12} {'Evict_Ratio':>12}")
    print("-" * 44)
    for r in R_VALUES:
        d = results[r]
        marker = " ←" if r == 32 else ""
        print(f"{r:>6} {d['accuracy']:>10.4f} "
              f"{d['frac_truncated']:>12.4f} "
              f"{d['eviction_ratio']:>12.4f}{marker}")

    # ── LaTeX appendix 테이블 출력 ────────────────────────────────────────
    print("\nLaTeX table rows:")
    print("$r$ & "
          + " & ".join(str(r) for r in R_VALUES)
          + r" \\")
    print("Accuracy & "
          + " & ".join(f"{results[r]['accuracy']:.3f}" for r in R_VALUES)
          + r" \\")
    print("Frac.\ truncated & "
          + " & ".join(f"{results[r]['frac_truncated']:.3f}" for r in R_VALUES)
          + r" \\")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="N=20으로 빠른 확인")
    args = ap.parse_args()
    run(n_samples=20 if args.quick else N_SAMPLES)