# Surprise is Expendable: KV Cache Eviction for Thinking Models

Calibration-free, Flash-Attention-compatible KV cache eviction
for LLM reasoning (Qwen3, DeepSeek-R1).

## Structure
src/core/       — eviction engine (cache ops, scorers, generation loop)
src/datasets/   — MATH500, GPQA loaders
src/methods/    — baseline implementations (RaaS, Random, LRU)
src/utils/      — metrics, I/O
experiments/    — experiment runners
analysis/       — visualization and per-sample analysis

## Quick start
pip install -r requirements.txt
python experiments/math500_ablation.py
python experiments/gpqa_ablation.py