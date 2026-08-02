"""
Random / LRU eviction baselines — mechanism analysis용 ablation.

Integration:
    generate_with_scored_eviction(... method='lru' ...)
    generate_with_scored_eviction(... method='random' ...)
    (scoring 로직은 generation.py에 직접 삽입, hook 불필요)

실험 목적:
    novelty_inv > LRU   → micro-variance signal의 실질적 기여 확인
    LRU         > Random → FIFO 구조 자체의 기여 확인
    (두 격차를 분리해서 novelty_inv 메커니즘의 두 component를 정량화)

LRU 구현 원리:
    think_generated_count는 단조 증가 카운터.
    오래된 토큰 = 낮은 카운터 = 낮은 score = evict_lowest에서 먼저 제거.
    think 토큰은 순차 생성되므로 re-access 없이 FIFO와 동일하게 동작.
"""