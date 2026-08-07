"""
MATH500 Accuracy Ablation
=========================
thin runner — config만 여기, 로직은 전부 src/ 에서 import
"""
import os
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.core.scorers as _scorers
from src.core.generation import generate_with_scored_eviction, MAX_NEW_TOKENS
from src.datasets.math500 import load_fixed_subset, build_prompt
from src.utils.metrics import is_correct
from src.utils.io import _load_completed

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
GPU_ID            = "6"
#MODEL_NAME        = "Qwen/Qwen3-8B"
MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
N_SAMPLES         = 500
KV_BUDGETS        = [512, 1024, 2048, 4096]
#METHODS           = ["baseline", "novelty_inv", "novelty", "k_norm", "lru", "random"]
METHODS           = ["novelty_inv"]
METHODS_SAVE_TEXT = {"lru", "novelty_inv", "k_norm", "baseline"}
MAX_NEW_TOKENS_EXP = MAX_NEW_TOKENS        # 8192 (generation.py 기본값)
OUT_DIR           = Path("results/math500_ablation_deepseek")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    # R_PROJ 초기화 → scorers 모듈 전역에 주입
    v_dim = model.model.layers[0].self_attn.v_proj.out_features
    _scorers.R_PROJ = (
        torch.randint(0, 2, (_scorers.NOVELTY_K, v_dim),
                      generator=torch.Generator(device=device).manual_seed(42),
                      device=device).float() * 2.0 - 1.0
    ) * (_scorers.NOVELTY_K ** -0.5)

    subset = load_fixed_subset(n=N_SAMPLES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path        = OUT_DIR / "results.jsonl"
    per_sample_path = OUT_DIR / "results_per_sample.jsonl"
    per_text_path   = OUT_DIR / "results_texts.jsonl"

    completed = _load_completed(out_path)
    if completed:
        print(f"[resume] {len(completed)}개 스킵: {sorted(completed)}")

    with open(out_path, "a") as f:
        for method in METHODS:
            budgets = [None] if method == "baseline" else KV_BUDGETS
            for budget in budgets:
                if (method, budget) in completed:
                    print(f"[skip] {method} budget={budget}")
                    continue

                correct = total_think = total_evicted = n_truncated = total_total = n_triggered = 0
                per_sample_rows, per_text_rows = [], []

                label = f"{method}" + (f" budget={budget}" if budget else "")
                pbar  = tqdm(subset, desc=label)

                for idx, ex in enumerate(pbar):
                    #prompt = build_prompt(tokenizer, ex["problem"])
                    prompt = build_prompt(tokenizer, ex["problem"], enable_thinking=False)
                    gold   = ex["answer"]

                    gen = generate_with_scored_eviction(
                        model, tokenizer, prompt, method, budget or 0, device,
                        max_new_tokens=MAX_NEW_TOKENS_EXP,
                    )

                    ok = is_correct(gen["pred"], gold)
                    correct     += ok
                    total_think += gen["think_tokens_generated"]
                    total_total += gen["total_tokens_generated"]
                    n_truncated += int(gen["truncated"])

                    per_sample_rows.append({
                        "idx":                    idx,
                        "method":                 method,
                        "budget":                 budget,
                        "think_tokens_generated": gen["think_tokens_generated"],
                        "total_tokens_generated": gen["total_tokens_generated"],
                        "truncated":              gen["truncated"],
                        "correct":                ok,
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
                        postfix["evicted"] = f"{total_evicted / max(total_think, 1):.1%}"
                    pbar.set_postfix(**postfix)

                n   = len(subset)
                row = {
                    "method":                        method,
                    "kv_budget":                     budget,
                    "n":                             n,
                    "acc":                           correct / n,
                    "avg_think_tokens_generated":    total_think / n,
                    "avg_total_tokens_generated":    total_total / n,
                    "avg_answer_tokens_generated":   (total_total - total_think) / n,
                    "frac_truncated":                n_truncated / n,
                }
                if method != "baseline":
                    row["avg_n_evicted"]                    = total_evicted / n
                    row["achieved_eviction_ratio"]          = total_evicted / max(total_think, 1)
                    row["frac_samples_eviction_triggered"]  = n_triggered / n

                f.write(json.dumps(row) + "\n")
                f.flush()

                with open(per_sample_path, "a") as pf:
                    for r in per_sample_rows:
                        pf.write(json.dumps(r) + "\n")

                if method in METHODS_SAVE_TEXT:
                    with open(per_text_path, "a") as tf:
                        for r, text in zip(per_sample_rows, per_text_rows):
                            tf.write(json.dumps({
                                "idx":       r["idx"],
                                "method":    r["method"],
                                "budget":    r["budget"],
                                "truncated": r["truncated"],
                                "correct":   r["correct"],
                                "text":      text,
                            }) + "\n")

                print(row, flush=True)


if __name__ == "__main__":
    run()