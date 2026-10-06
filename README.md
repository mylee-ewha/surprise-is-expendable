# Surprise Is Expendable: Streaming KV Eviction via Information Novelty for Thinking LLMs

> *High-novelty tokens are the most expendable.* We show that tokens with the
> highest V-space leverage scores in a reasoning chain are generic setup phrases
> at the opening of `<think>`—statistically novel only because the V-subspace
> hasn't been established yet. Evicting them improves accuracy.

Calibration-free, FlashAttention-compatible streaming KV eviction for thinking
LLMs (Qwen3-8B, DeepSeek-R1-Distill-8B). No attention materialization.
Single hyperparameter: the KV budget *B*.

---

## Key Results

| Benchmark | Model | Budget | NI (ours) | LRU | Full KV |
|---|---|---|---|---|---|
| AIME 2024–25 | DeepSeek-R1-Distill-8B | B=8192 | **0.400 ± 0.017** | 0.350 ± 0.029 | 0.389 |
| MATH500 | Qwen3-8B | B=4096 | **0.790** | 0.784 | 0.764 |
| GPQA Diamond | Qwen3-8B | B=4096 | 0.571 | 0.581 | 0.571 |

NI matches or exceeds LRU once the eviction ratio falls below ~0.50, and
surpasses the full-context baseline on AIME and MATH500 without calibration,
training, or attention-score access.

---

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+, PyTorch 2.1+, CUDA 12.1.

## Reproducing Paper Results

```bash
# MATH500 / GPQA Diamond
python experiments/math500_ablation.py --model qwen3-8b
python experiments/gpqa_ablation.py   --model qwen3-8b

# AIME 2024–25 (3 seeds)
python experiments/aime_ablation.py --model deepseek-8b --seeds 42 1234 5678

# Mechanism figures (Figure 3)
python analysis/plot_mechanism.py
```

## Code Structure
src/core/ — eviction engine (score tracking, KV cache ops, generation loop)
src/datasets/ — MATH500, GPQA Diamond, AIME loaders
src/methods/ — LRU, Random, RaaS, R-KV, k-norm, Novelty baselines
src/utils/ — metrics, I/O
experiments/ — one runner script per benchmark
analysis/ — visualization and per-sample analysis


## Citation

```bibtex
@article{sie2026,
  title  = {Surprise Is Expendable: Streaming KV Eviction via Information Novelty for Thinking LLMs},
  author = {Anonymous},
  year   = {2026},
}
```
