"""
R-KV Re-implementation
======================
Original paper: Cai et al. (2025), "R-KV: Redundancy-Aware KV Cache
Compression for Reasoning Models", NeurIPS 2025.
Original code:  https://github.com/Zefan-Cai/R-KV

We re-implement R-KV within our unified token-level eviction framework
to enable fair comparison under identical evaluation conditions
(same lazy-batch schedule I=128, same budget definition, same model
checkpoints and sampling parameters).

Fidelity to original:
  - Importance scoring: attention-based, using last alpha=8 tokens
    as observation window (Algorithm 1, line 3)
  - Redundancy scoring: cosine similarity of Key vectors,
    threshold=0.5, retain_direction='last' (cal_similarity in utils.py)
  - Joint score: Z = lambda * Importance - (1-lambda) * Redundancy,
    lambda=0.07 (Table 1 default)
  - Eviction trigger: every B_buffer=128 tokens (= our EVICT_INTERVAL)

Known deviations (due to unified-framework constraints, applied equally
to all methods including ours):
  1. Eviction granularity: token-level uniform eviction across all heads
     and layers, vs. per-KV-head eviction in the original.
  2. Importance computation: full-cache output_attentions (softmax over
     entire cache), vs. original's candidate-only re-softmax with a
     separate lightweight Q·K pass.
  3. Recent-token protection window: rho=16 tokens (our framework
     default) vs. window_size=8 in original.
  4. Eviction schedule: lazy batch (every 128 steps) vs. original's
     continuous top-k selection.

Deviations (1), (3), (4) apply equally to ALL baselines including our
method, preserving intra-framework fairness. Deviation (2) affects only
R-KV; we discuss the potential impact in Appendix X.
"""