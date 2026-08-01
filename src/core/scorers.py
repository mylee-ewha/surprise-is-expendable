import torch
from collections import deque
import numpy as np

from .cache_ops import extract_last_position_knorm

R_PROJ: torch.Tensor = None
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_LAYERS = 36
WINDOW = 32
MIN_PERIODS = 16
DONUT_BAND = range(24, 37)
LAMBDA_RIDGE = 1.0
REFRESH_INTERVAL = 256
NOVELTY_K      = 32   # RP 압축 차원

# ---------------------------------------------------------------------------
# Causal rolling z-score
# ---------------------------------------------------------------------------

class CausalZScorer:
    def __init__(self, window: int = WINDOW, min_periods: int = MIN_PERIODS):
        self.window = window
        self.min_periods = min_periods
        self.buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        if len(self.buf) < self.min_periods:
            z = float("nan")
        else:
            arr = np.array(self.buf)
            mu, sigma = arr.mean(), arr.std()
            z = (value - mu) / (sigma + 1e-6)
        self.buf.append(value)
        return z


# ---------------------------------------------------------------------------
# Causal leverage / novelty score
# ---------------------------------------------------------------------------

class CausalLeverageScorer:
    def __init__(self, dim: int, lam: float = LAMBDA_RIDGE,
                 refresh_interval: int = REFRESH_INTERVAL, device: str = "cuda"):
        self.lam = lam
        self.refresh_interval = refresh_interval
        self.t = 0
        self.A = lam * torch.eye(dim, dtype=torch.float64, device=device)
        self.A_inv = (1.0 / lam) * torch.eye(dim, dtype=torch.float64, device=device)

    def update(self, v: torch.Tensor) -> float:
        v64 = v.double()
        Av = self.A_inv @ v64
        denom = 1.0 + torch.dot(v64, Av)
        score = torch.dot(v64, Av).item()
        self.A = self.A + torch.outer(v64, v64)
        self.A_inv = self.A_inv - torch.outer(Av, Av) / denom
        self.t += 1
        if self.t % self.refresh_interval == 0:
            self.A_inv = torch.linalg.inv(self.A)
        return score


# ---------------------------------------------------------------------------
# Per-sample running scorer
# ---------------------------------------------------------------------------

class PerSampleScorer:
    def __init__(self, method: str, device: str):
        self.method = method
        self.device = device
        self.donut_zscorers = (
            {i: CausalZScorer() for i in DONUT_BAND} if method in ("donut_a_v2", "donut_a_v2_inv") else None
        )
        self.novelty_scorers = {} if method in ("novelty", "novelty_inv") else None 
        self.prev_v_all = None

    def score_k_norm(self, cache) -> float:
        return extract_last_position_knorm(cache)

    def score_donut_a_v2(self, hs) -> float:
        h0_norm = hs[0].float().norm().item()
        layer_cum = [hs[i].float().norm().item() - h0_norm for i in range(0, 37)]
        donut_sum, any_valid = 0.0, False
        for i in DONUT_BAND:
            inc_i = layer_cum[i] - layer_cum[i - 1]
            z = self.donut_zscorers[i].update(inc_i)
            if not np.isnan(z):
                donut_sum += z
                any_valid = True
        return donut_sum if any_valid else float("nan")

    def score_novelty(self, v_storage) -> float:
        scores = []
        for li in range(1, N_LAYERS + 1):
            if li not in v_storage:
                continue
            v_vec = v_storage[li][0, 0].float()        # [v_dim]
            v_proj = R_PROJ @ v_vec                    # [k]
            if li not in self.novelty_scorers:
                self.novelty_scorers[li] = CausalLeverageScorer(
                    dim=NOVELTY_K, device=self.device
                )
            scores.append(self.novelty_scorers[li].update(v_proj))
        return float(np.mean(scores)) if scores else float("nan")
    
    def score_v_angular(self, v_storage) -> float:
        keys = sorted(v_storage.keys())
        v_all = torch.stack([v_storage[li][0, 0].float() for li in keys])       # [36, v_dim]
        v_cur_all = v_all @ R_PROJ.T 

        if self.prev_v_all is None:
            score = v_cur_all.norm(dim=-1).mean().item()
        else:
            dot      = (v_cur_all * self.prev_v_all).sum(dim=-1, keepdim=True)        # [36, 1]
            pv_norm2 = (self.prev_v_all * self.prev_v_all).sum(dim=-1, keepdim=True) + 1e-8
            parallel = (dot / pv_norm2) * self.prev_v_all                              # [36, 32]
            perp     = v_cur_all - parallel                                             # [36, 32]
            score    = perp.norm(dim=-1).mean().item()

        self.prev_v_all = v_cur_all.clone()
        return score
