"""
analysis/timing_benchmark.py
NI vs LRU decode wall-clock overhead 측정
Usage: python timing_benchmark.py
"""

import os, sys, time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.generation import generate_with_scored_eviction
from src.datasets.math500 import load_fixed_subset
import src.core.scorers as _scorers

# ── CONFIG ────────────────────────────────────────────────────────────────
GPU_ID      = "6"
MODEL_NAME  = "Qwen/Qwen3-8B"
BUDGET      = 2048
N_SAMPLES   = 20
N_WARMUP    = 2
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoModelForCausalLM, AutoTokenizer

def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def run_and_time(model, tokenizer, prompt, method, budget, device):
    t0 = sync_time()
    result = generate_with_scored_eviction(
        model, tokenizer, prompt, method, budget, device)
    t1 = sync_time()
    return result["think_tokens_generated"], t1 - t0

def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # NI/LRU 모두 sdpa 사용 (output_attentions=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="sdpa")
    model.eval()

    v_dim = model.model.layers[0].self_attn.v_proj.out_features
    _scorers.R_PROJ = (
        torch.randint(0, 2, (_scorers.NOVELTY_K, v_dim),
                    generator=torch.Generator(device=device).manual_seed(42),
                    device=device).float() * 2.0 - 1.0
    ) * (_scorers.NOVELTY_K ** -0.5)
    print(f"  R_PROJ initialized: {_scorers.R_PROJ.shape}")

    ds = load_fixed_subset(n=500, seed=42)
    mk_prompt = lambda q: tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)

    # 동일 샘플, 순서 교차로 측정 (GPU cache 편향 방지)
    idxs = list(range(N_WARMUP + N_SAMPLES))
    prompts = [mk_prompt(ds[i]["problem"]) for i in idxs]

    ni_times, lru_times   = [], []
    ni_think_ts           = []

    for i, prompt in enumerate(prompts):
        warmup = (i < N_WARMUP)

        with torch.no_grad():
            # 홀수 샘플은 NI→LRU, 짝수는 LRU→NI (순서 교차)
            if i % 2 == 0:
                T_ni,  t_ni  = run_and_time(model, tokenizer, prompt,
                                             "novelty_inv", BUDGET, device)
                _,     t_lru = run_and_time(model, tokenizer, prompt,
                                             "lru",         BUDGET, device)
            else:
                _,     t_lru = run_and_time(model, tokenizer, prompt,
                                             "lru",         BUDGET, device)
                T_ni,  t_ni  = run_and_time(model, tokenizer, prompt,
                                             "novelty_inv", BUDGET, device)

        if not warmup:
            ni_times.append(t_ni)
            lru_times.append(t_lru)
            ni_think_ts.append(T_ni)
            print(f"  [{i-N_WARMUP+1:3d}/{N_SAMPLES}] "
                  f"NI={t_ni:.2f}s  LRU={t_lru:.2f}s  "
                  f"overhead={((t_ni-t_lru)/t_lru*100):+.1f}%")
        else:
            print(f"  [warmup {i+1}/{N_WARMUP}]")

    ni_arr  = np.array(ni_times)
    lru_arr = np.array(lru_times)
    diff    = ni_arr - lru_arr
    pct     = diff / lru_arr * 100
    per_tok = diff / np.array(ni_think_ts) * 1000  # ms per think token

    print("\n" + "="*50)
    print(f"  Overhead (%)              : "
          f"{pct.mean():.1f} ± {pct.std():.1f}")
    print(f"  Per think-token (ms)      : "
          f"{per_tok.mean():.3f} ± {per_tok.std():.3f}")
    print(f"  NI total (s)              : "
          f"{ni_arr.mean():.2f} ± {ni_arr.std():.2f}")
    print(f"  LRU total (s)             : "
          f"{lru_arr.mean():.2f} ± {lru_arr.std():.2f}")
    print("="*50)
    print(f"\nLaTeX snippet:")
    print(f"NI adds {pct.mean():.1f}\\,\\% wall-clock overhead over LRU "
          f"({per_tok.mean():.2f}\\,ms per think token, "
          f"$n={N_SAMPLES}$, Qwen3-8B, $B={BUDGET}$).")

if __name__ == "__main__":
    main()