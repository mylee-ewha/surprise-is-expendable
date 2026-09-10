"""
circular loop 샘플 vs 정상 샘플의 V-space expansion 패턴 비교.
결론이 나기 전까지 loop escape 구현하지 말 것.

Usage:
  python analysis/expansion_diag.py --collect   # S_t trajectory 수집
  python analysis/expansion_diag.py --plot      # 저장된 npz로 비교 플롯
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.expansion import VSpaceExpansionTracker

LOG_DIR = Path("expansion_logs")
PLOT_DIR = Path("expansion_plots")


def select_targets(per_sample_path: Path, budget: int, n_each=5):
    """
    circular 샘플: truncated=True & correct=False
    정상 샘플:     truncated=False & correct=True
    """
    circ, norm = [], []
    with open(per_sample_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("method") != "novelty_inv" or r.get("kv_budget") != budget:
                continue
            if r.get("truncated") and not r.get("correct"):
                circ.append(r["idx"])
            elif not r.get("truncated") and r.get("correct"):
                norm.append(r["idx"])
    return circ[:n_each], norm[:n_each]


def plot_comparison(circ_npzs, norm_npzs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    smooth = lambda x, w=50: np.convolve(x, np.ones(w)/w, mode="valid")

    for npzs, label, color in [(circ_npzs, "circular", "#C0392B"),
                                (norm_npzs, "normal", "#2471A3")]:
        for i, d in enumerate(npzs):
            lbl = label if i == 0 else None
            axes[0][0].plot(smooth(d["delta_logdet"]), color=color, alpha=0.5, label=lbl)
            axes[0][1].plot(d["log_det"], color=color, alpha=0.5)
            axes[1][0].plot(smooth(d["st"]), color=color, alpha=0.5)
            # plateau 비율을 구간별로
            pl = d["is_plateau"].astype(float)
            axes[1][1].plot(smooth(pl, 200), color=color, alpha=0.5)

    axes[0][0].set_title("Δ log det (rolling 50) — plateau가 구분되는가?")
    axes[0][1].set_title("cumulative log det")
    axes[1][0].set_title("S_t (rolling 50)")
    axes[1][1].set_title("plateau flag 비율 (rolling 200)")
    axes[0][0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(ls=":", alpha=0.4)

    plt.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")


# TODO: --collect 부분은 vtilde_analysis.py의 generation 루프를 재사용
#       (이미 S_t를 per-token으로 뽑는 코드가 있음)