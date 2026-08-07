"""
RaaS: Reasoning-Aware Attention Sparsity
ACL Findings 2025 — Hu et al.

알고리즘 요약 (Section 3.2):
  - Milestone tokens: 처음엔 high attention → 점차 attention 감소 → never recover
    (수학 증명의 lemma처럼 등장 후 소진)
  - Phoenix tokens: 낮은 attention 기간 후 다시 중요해지는 토큰 (주로 prefill에 등장)

스코어링 메커니즘 (timestamp 기반 LRU):
  - 매 decode step에서 attention score가 median 이상인 상위 r=50% 토큰에
    최신 timestamp(현재 step 번호)를 부여
  - Cache full 시 → timestamp가 가장 오래된 토큰(= 오랫동안 attention 못 받은 토큰) evict

LRU와의 차이:
  - Pure LRU: 생성 시점이 가장 오래된 토큰 evict (recency of generation)
  - RaaS: 마지막으로 의미 있는 attention을 받은 시점이 가장 오래된 토큰 evict
           (recency of attention) → milestone token이 오래 살아남음

FA 호환성:
  - Naive RaaS: post-softmax attention 필요 → FA 비호환
  - Page-based RaaS (논문 Section 3.3): Q·K_rep 별도 계산 → FA 호환
  - 우리 구현: accuracy 비교 목적으로 naive 방식 사용 (output_attentions=True)
  - 모델 로드 시 attn_implementation="sdpa" 또는 "eager" 필수

하이퍼파라미터:
  - RAAS_R = 0.5: 매 step 상위 50% 토큰이 timestamp 갱신 (논문 기본값)
  - page_size = 16 (논문), 우리 구현에서는 token-level로 단순화
  - prefill 토큰 전체 보호 (eviction 대상 아님) — 기존 코드와 동일
"""

# 매 decode step에서 attention 상위 r 비율 토큰에 최신 timestamp 부여
RAAS_R: float = 0.5