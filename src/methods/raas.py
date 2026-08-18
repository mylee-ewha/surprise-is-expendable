"""
RaaS Re-implementation
======================
Original paper: Hu et al. (2025), "RaaS: Reasoning-Aware Attention
Sparsity for Efficient LLM Reasoning", ACL Findings 2025.
Original code:  Not publicly available; re-implemented from the paper.

We re-implement RaaS within our unified token-level eviction framework
to enable fair comparison under identical evaluation conditions
(same lazy-batch schedule I=128, same budget definition, same model
checkpoints and sampling parameters).

Fidelity to original:
  - Core mechanism: LRU-based eviction with attention-score timestamps.
    At each decode step, tokens receiving attention scores above the
    median (top-r=50% by attention) are assigned the latest timestamp;
    eviction removes tokens with the oldest timestamps (Algorithm 1).
  - Timestamp retention ratio: r=0.5 (paper default, Section 3.2).
  - Prefill tokens: fully retained without eviction, consistent with
    the original design (Algorithm 1, line 22).
  - Milestone token intuition: tokens that continue to receive high
    attention accumulate recent timestamps and are preserved;
    tokens whose attention fades are evicted as "non-milestone".

Known deviations (due to unified-framework constraints, applied equally
to all methods including ours):
  1. Attention computation: token-level naive implementation using
     output_attentions=True (full attention matrix, eager/sdpa mode),
     vs. the original's page-based lightweight Q·K_rep pass compatible
     with FlashAttention (Section 3.3 of the paper). This gives RaaS
     attention-exact scores, which is strictly more favorable to RaaS
     than the approximate page-based variant.
  2. Eviction granularity: token-level uniform eviction across all
     heads and layers, vs. page-level eviction (page_size=16) in the
     original.
  3. Recent-token protection window: rho=16 tokens (our framework
     default) vs. the original's observation window alpha.
  4. Eviction schedule: lazy batch (every 128 steps) vs. the original's
     continuous per-step eviction.

Deviations (2), (3), (4) apply equally to ALL baselines including our
method, preserving intra-framework fairness. Deviation (1) affects only
RaaS and is strictly favorable to RaaS (attention-exact vs. approximate),
meaning our comparison does not disadvantage RaaS relative to its
reported results.
"""

# Timestamp retention ratio r (top-r fraction of tokens receive the
# latest timestamp at each decode step). Default: 0.5 (paper Table 1).
RAAS_R: float = 0.5