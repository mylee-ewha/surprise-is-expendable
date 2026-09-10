"""
V-space expansion tracking — circular loop detection 용.
NI core (scorers.CausalLeverageScorer)와 독립적으로 동작.
"""
import numpy as np
from collections import deque


class VSpaceExpansionTracker:
    """
    Δ log det(A_t) = log(1 + S_t)  ← Matrix Determinant Lemma
    기존 S_t를 그대로 받아서 추가 연산 없이 expansion 추적.
    """

    def __init__(self, window=128, plateau_threshold=0.003,
                 spike_mult=3.0, lam=1.0, dim=32):
        self.window = window
        self.plateau_threshold = plateau_threshold
        self.spike_mult = spike_mult
        self.log_det = dim * np.log(lam)
        self.step = 0

        self._dlogdet_win = deque(maxlen=window)
        self._st_win = deque(maxlen=window)

        # 분석용 로그
        self.history = {"step": [], "st": [], "delta_logdet": [],
                        "log_det": [], "is_plateau": [], "is_spike": []}

    def update(self, st: float) -> dict:
        """generation loop에서 score 계산 직후 호출."""
        self.step += 1
        dlogdet = float(np.log1p(max(st, -0.999)))
        self.log_det += dlogdet
        self._dlogdet_win.append(dlogdet)
        self._st_win.append(st)

        ready = len(self._dlogdet_win) >= self.window // 2
        roll_d = float(np.mean(self._dlogdet_win)) if ready else None
        roll_st = float(np.mean(self._st_win)) if ready else None

        is_plateau = ready and roll_d < self.plateau_threshold
        is_spike = ready and st > roll_st * self.spike_mult and st > 0.01

        rec = {"step": self.step, "st": st, "delta_logdet": dlogdet,
               "log_det": self.log_det, "rolling_dlogdet": roll_d,
               "is_plateau": is_plateau, "is_spike": is_spike}
        for k in self.history:
            self.history[k].append(rec[k])
        return rec

    def to_npz_dict(self) -> dict:
        return {k: np.array(v) for k, v in self.history.items()}