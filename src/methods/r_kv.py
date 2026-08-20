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

import torch
import torch.nn.functional as F
from collections import deque
from transformers import DynamicCache

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
RKV_BUFFER_SIZE: int   = 128    # eviction 주기 (= EVICT_INTERVAL과 동일)
RKV_ALPHA:       int   = 8      # observation window (원본 window_size=8)
RKV_LAMBDA:      float = 0.07   # 원본 mix_lambda 기본값
RKV_KERNEL:      int   = 7      # 원본 kernel_size — importance max-pool 스무딩
RKV_SIM_THRESH:  float = 0.5    # 원본 cal_similarity threshold (retain 판정)


# ── Attention history 관리 ────────────────────────────────────────────────────

def make_attn_history() -> deque:
    """alpha 개만 유지하는 deque 반환"""
    return deque(maxlen=RKV_ALPHA)


def update_attn_history(
    history: deque,
    attentions,          # step_out.attentions: tuple of (1, heads, 1, seq_k) per layer
    prompt_len: int,
    n_think: int,
) -> None:
    """
    현재 step의 attention을 전 레이어 평균해서 history에 추가.
    think 토큰 구간(prompt_len : prompt_len+n_think)의 attention만 저장.
    """
    if attentions is None:
        raise RuntimeError(
            "R-KV requires output_attentions=True. "
            "Load model with attn_implementation='eager'."
        )
    if n_think == 0:
        return

    seq_k = attentions[0].shape[-1]
    avg = torch.zeros(seq_k, dtype=torch.float32)
    for la in attentions:
        avg += la[0, :, 0, :seq_k].mean(dim=0).cpu().float()
    avg /= len(attentions)

    history.append(avg)


# ── Z-score 계산 ──────────────────────────────────────────────────────────────

def compute_rkv_scores(
    history:    deque,
    cache:      DynamicCache,
    prompt_len: int,
    n_think:    int,
    lam:        float = RKV_LAMBDA,
    alpha:      int   = RKV_ALPHA,
) -> list:
    """
    n_think개 think 토큰에 대한 Z-score 반환 (높을수록 보존 우선).
    eviction은 Z가 낮은 토큰부터 (= _EVICT_HIGHEST에 없으므로 lowest 먼저).

    Args:
        history:    최근 alpha 개의 avg-attention 벡터 deque (각 shape: seq_k)
        cache  :    DynamicCache (eviction 직전 상태, 아직 compact 안 됨)
        prompt_len: prefill 토큰 수 (보호 영역)
        n_think:    현재 캐시에 남은 think 토큰 수
        lam:        importance 가중치
    """
    if n_think == 0:
        return []

    # ── 1. Importance (attention) ─────────────────────────────────────────────
    recent = list(history)[-alpha:] if len(history) > 0 else []
    if recent:
        L = min(t.shape[0] for t in recent)                         # 가변 길이 정렬
        imp_vec = torch.stack([t[:L] for t in recent]).mean(dim=0)  # 최근 α step 평균(원본 window mean)
        imp_think = torch.tensor([
            float(imp_vec[prompt_len + j])
            if (prompt_len + j) < L else 0.0
            for j in range(n_think)
        ])
    else:
        imp_think = torch.zeros(n_think)

    # 원본 R-KV: max-pool 스무딩(kernel_size=7). min-max 정규화는 원본에 없음 → 제거
    imp_think = torch.nn.functional.max_pool1d(
        imp_think.view(1, 1, -1),
        kernel_size=RKV_KERNEL, padding=RKV_KERNEL // 2, stride=1,
    ).view(-1)[:n_think]

    # ── 2. Redundancy (K-벡터 cosine similarity) ──────────────────────────────
    redundancy = torch.zeros(n_think)
    if n_think > 1:
        k_sum = None
        n_layers = 0

        layer_keys_iter = (
            list(cache.key_cache) if getattr(cache, "key_cache", None)
            else [getattr(_l, "keys", None) for _l in getattr(cache, "layers", [])]
        )
        if layer_keys_iter:
            for layer_k in layer_keys_iter:
                if layer_k is None or layer_k.numel() == 0:
                    continue
                seq_len = layer_k.shape[2]
                end_idx = min(prompt_len + n_think, seq_len)
                if end_idx <= prompt_len:
                    continue
                portion = layer_k[0, :, prompt_len:end_idx, :]  # (heads, n_avail, d_k)
                k_mean  = portion.float().mean(dim=0)            # (n_avail, d_k)
                # 길이가 n_think에 못 미치면 zero-pad (eviction lag 등)
                if k_mean.shape[0] < n_think:
                    pad = torch.zeros(
                        n_think - k_mean.shape[0], k_mean.shape[1],
                        device=k_mean.device
                    )
                    k_mean = torch.cat([k_mean, pad], dim=0)
                k_sum = k_mean if k_sum is None else k_sum + k_mean
                n_layers += 1

        if k_sum is not None and n_layers > 0:
            k_avg  = (k_sum / n_layers).cpu()
            k_norm = F.normalize(k_avg, dim=-1)                  # (n_think, d_k)
            sim    = torch.mm(k_norm, k_norm.T)                  # (n_think, n_think)
            sim.fill_diagonal_(0.0)                               # self-sim 제외
            # 원본 retain_direction="last": 각 토큰과 threshold 이상 유사한 것 중
            # 가장 최근(최대 idx) 토큰과의 유사도를 0으로 (중복 중 최신은 보존)
            sim_mask   = sim > RKV_SIM_THRESH
            col_idx    = torch.arange(n_think).view(1, -1).expand(n_think, -1)
            masked_idx = torch.where(sim_mask, col_idx, torch.zeros_like(col_idx))
            retain     = masked_idx.max(dim=-1).values           # (n_think,)
            sim.scatter_(-1, retain.view(-1, 1), 0.0)
            # 원본: mean(dim=-2).softmax(-1) — 행(query) 평균 후 softmax
            redundancy = sim.mean(dim=0).softmax(dim=0)          # (n_think,)

    # ── 3. Joint score ────────────────────────────────────────────────────────
    z = lam * imp_think - (1.0 - lam) * redundancy   # higher = KEEP
    return z.tolist()