"""
Loop escape via n-gram repetition detection.
NI의 established-token 보존이 circular loop를 고착시키는 문제 완화.
"""
from collections import deque


class LoopDetector:
    """생성 텍스트의 n-gram 반복률을 실시간 추적."""

    def __init__(self, window=200, ngram_n=8,
                 enter_threshold=0.20, exit_threshold=0.08, min_confirm=32):
        self.window = window
        self.n = ngram_n
        self.enter = enter_threshold
        self.exit = exit_threshold
        self.min_confirm = min_confirm

        self._ids = deque(maxlen=window)
        self._in_loop = False
        self._confirm = 0

    def update(self, token_id: int) -> bool:
        self._ids.append(token_id)
        rep = self.rep_rate()
        thr = self.enter if not self._in_loop else self.exit
        hit = (rep > thr) if not self._in_loop else (rep < thr)

        if hit:
            self._confirm += 1
            if self._confirm >= self.min_confirm:
                self._in_loop = not self._in_loop
                self._confirm = 0
        else:
            self._confirm = 0
        return self._in_loop

    def rep_rate(self) -> float:
        ids = list(self._ids)
        if len(ids) < self.n:
            return 0.0
        g = [tuple(ids[i:i + self.n]) for i in range(len(ids) - self.n + 1)]
        return 1.0 - len(set(g)) / len(g)

    @property
    def in_loop(self) -> bool:
        return self._in_loop


def select_evict_indices(scores, in_loop, budget,
                         recent_size, oldest_frac=0.40):
    """
    evict할 think_scores 인덱스 반환 (내림차순 정렬).

    in_loop=False → 기존 NI (S_t 높은 순)
    in_loop=True  → 오래된 것 우선 (LRU fallback), 부족분은 NI
    """
    import numpy as np

    n_to_evict = len(scores) - budget
    n_cand = len(scores) - recent_size
    if n_cand <= 0 or n_to_evict <= 0:
        return []

    valid = [(j, s) for j, s in enumerate(scores[:n_cand]) if not np.isnan(s)]
    if not valid:
        return []
    n_actual = min(n_to_evict, len(valid))

    if not in_loop:
        sv = sorted(valid, key=lambda p: p[1], reverse=True)
        picked = [j for j, _ in sv[:n_actual]]
    else:
        n_force = min(int(len(valid) * oldest_frac), n_actual)
        picked = [j for j, _ in valid[:n_force]]          # 오래된 순
        rest = n_actual - n_force
        if rest > 0:
            remain = [(j, s) for j, s in valid if j not in set(picked)]
            sv = sorted(remain, key=lambda p: p[1], reverse=True)
            picked += [j for j, _ in sv[:rest]]

    return sorted(picked, reverse=True)