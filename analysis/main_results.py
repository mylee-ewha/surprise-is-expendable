#!/usr/bin/env python3
"""
SiE Paper — Figure Generation Script
  Figure 1 : Main results 2×2  (MATH500 + GPQA) × (Qwen3 + DeepSeek)
  Figure 2 : AIME results 1×2  with error bars
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# RC params  (ACL/LaTeX-compatible: serif, Times-like, embeddable fonts)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family":      "sans-serif",
        "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "font.size":        9,
        "axes.titlesize":   9.5,
        "axes.labelsize":   8.5,
        "legend.fontsize":  8,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "savefig.dpi":      300,
        "pdf.fonttype":     42,   # TrueType → editable in Illustrator/Inkscape
        "ps.fonttype":      42,
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Data  (None = result pending / not run for this model)
# ─────────────────────────────────────────────────────────────────────────────
BUDGETS_MG = [512, 1024, 2048, 4096]
BUDGETS_A  = [2048, 4096, 8192, 16384]

RESULTS_MG = {
    # ── MATH500 ───────────────────────────────────────────────────────────
    ("MATH500", "Qwen3-8B"): {
        "baseline":  0.764,
        "NI (Ours)": [0.628, 0.722, 0.774, 0.790],
        "LRU":       [0.670, 0.730, 0.774, 0.784],
        "Random":    [0.526, 0.630, 0.664, 0.738],
        "k-norm":    [0.540, 0.646, 0.736, 0.766],
        "RaaS":      [0.572, 0.674, 0.726, 0.752],
        "R-KV":      [0.308, 0.476, 0.624, 0.726],
        "Novelty":   [0.042, 0.178, 0.448, 0.634],
    },
    ("MATH500", "DeepSeek-8B"): {
        "baseline":  0.708,
        "NI (Ours)": [0.632, 0.694, 0.714, 0.726],
        "LRU":       [0.668, 0.708, 0.714, 0.730],
        "Random":    [0.538, 0.614, 0.674, 0.710],
        "k-norm":    [0.468, 0.558, 0.620, 0.672],
        "RaaS":      [0.678, 0.700, 0.724, 0.718],
        "R-KV":      [0.572, 0.630, 0.678, 0.716],
        "Novelty":   [0.228, 0.390, 0.570, 0.668],
    },
    # ── GPQA Diamond ──────────────────────────────────────────────────────
    ("GPQA Diamond", "Qwen3-8B"): {
        "baseline":  0.571,
        "NI (Ours)": [0.308, 0.424, 0.475, 0.571],
        "LRU":       [0.338, 0.444, 0.515, 0.581],
        "Random":    [0.247, 0.343, 0.404, 0.525],
        "k-norm":    [0.318, 0.384, 0.434, 0.510],
        "RaaS":      [0.227, 0.434, 0.500, 0.540],
        "R-KV":      [0.147, 0.268, 0.419, 0.515],
        "Novelty":   [0.061, 0.091, 0.167, 0.338],
    },
    ("GPQA Diamond", "DeepSeek-8B"): {
        "baseline":  0.429,
        "NI (Ours)": [0.167, 0.278, 0.364, 0.409],
        "LRU":       [0.258, 0.288, 0.379, 0.424],
        "Random":    [0.177, 0.263, 0.333, 0.389],
        "k-norm":    [0.172, 0.187, 0.237, 0.298],
        "RaaS":      [0.303, 0.374, 0.409, 0.429],
        "R-KV":      [0.202, 0.258, 0.364, 0.419],
        "Novelty":   [0.076, 0.111, 0.172, 0.242],
    },
}

# AIME: (mean, std) tuples
RESULTS_AIME = {
    "Qwen3-8B": {
        "baseline":  (0.694, 0.008),
        "NI (Ours)": [(0.328, 0.042), (0.544, 0.035), (0.711, 0.069), (0.756, 0.010)],
        "LRU":       [(0.422, 0.059), (0.600, 0.017), (0.694, 0.010), (0.744, 0.042)],
        "RaaS":      [None, None, None, None],
        "R-KV":      [None, None, None, None],
    },
    "DeepSeek-8B": {
        "baseline":  (0.389, 0.048),
        "NI (Ours)": [(0.278, 0.042), (0.350, 0.044), (0.400, 0.017), (0.400, 0.058)],
        "LRU":       [(0.339, 0.026), (0.372, 0.019), (0.350, 0.029), (0.400, 0.058)],
        "RaaS":      [None, None, None, None],
        "R-KV":      [None, None, None, None],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Method visual styles
#   Core  (NI, LRU)          — thick solid lines, prominent markers
#   Ablation (Random, k-norm) — thin dashed,   subdued
#   SOTA  (RaaS, R-KV)       — thin dotted,    semi-transparent
# ─────────────────────────────────────────────────────────────────────────────
METHOD_ORDER = ["NI (Ours)", "LRU", "Random", "k-norm", "RaaS", "R-KV", "Novelty"]

STYLE = {
    "NI (Ours)": dict(
        color="#C0392B", lw=2.2, ls="-", marker="*",
        ms=10, zorder=5, markeredgewidth=0.4, markeredgecolor="#7B241C",
    ),
    "LRU": dict(
        color="#2471A3", lw=2.0, ls="-", marker="o",
        ms=6, zorder=4, markeredgewidth=0.4, markeredgecolor="#1A5276",
    ),
    "Random": dict(
        color="#7D8B8C", lw=1.2, ls="--", marker="s",
        ms=4.5, zorder=2, alpha=0.85,
        markeredgewidth=0.3, markeredgecolor="#5D6D7E",
    ),
    "k-norm": dict(
        color="#1A8C4E", lw=1.2, ls="--", marker="^",
        ms=5, zorder=2, alpha=0.85,
        markeredgewidth=0.3, markeredgecolor="#145A32",
    ),
    "RaaS": dict(
        color="#D35400", lw=1.2, ls=":", marker="D",
        ms=4.5, zorder=3, alpha=0.80,
        markeredgewidth=0.3, markeredgecolor="#A04000",
    ),
    "R-KV": dict(
        color="#7D3C98", lw=1.2, ls=":", marker="P",
        ms=5, zorder=3, alpha=0.80,
        markeredgewidth=0.3, markeredgecolor="#512E5F",
    ),
    "Novelty": dict(
        color="#B7770D", lw=1.0, ls=":", marker="x",
        ms=5, zorder=1, alpha=0.70,
        markeredgewidth=0.8, markeredgecolor="#7D5A0A",
    )
}

BASELINE_STYLE = dict(color="#2C3E50", lw=1.1, ls="-.", alpha=0.55, zorder=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw one panel
# ─────────────────────────────────────────────────────────────────────────────
def _plot_panel(ax, data, budgets, panel_label, title, show_ylabel=True):
    baseline = data["baseline"]

    ax.axhline(baseline, label="Full KV (baseline)", **BASELINE_STYLE)

    for method in METHOD_ORDER:
        if method not in data:
            continue
        vals = data[method]
        xs = [b for b, v in zip(budgets, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if not xs:
            continue
        ax.plot(xs, ys, label=method, **STYLE[method])

    # Title / labels
    ax.set_title(f"{panel_label} {title}", loc="left",
                 fontsize=9.5, fontweight="bold", pad=4)
    ax.set_xlabel("KV Budget (think tokens)", labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Accuracy", labelpad=2)

    # x-axis: log₂ scale
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.tick_params(which="both", direction="in", top=True, right=True, length=3)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.4)

    # y-axis: tight range with a small margin
    all_vals = [baseline] + [
        v for m in METHOD_ORDER if m in data
        for v in data[m] if v is not None
    ]
    lo, hi = min(all_vals), max(all_vals)
    margin = max((hi - lo) * 0.10, 0.02)
    ax.set_ylim(lo - margin, hi + margin)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — MATH500 + GPQA Diamond  (2 × 2)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure1(out_path):
    LAYOUT = [
        ("(a)", "MATH500",      "Qwen3-8B",    "MATH500 — Qwen3-8B"),
        ("(b)", "MATH500",      "DeepSeek-8B", "MATH500 — DeepSeek-8B"),
        ("(c)", "GPQA Diamond", "Qwen3-8B",    "GPQA Diamond — Qwen3-8B (16K)"),
        ("(d)", "GPQA Diamond", "DeepSeek-8B", "GPQA Diamond — DeepSeek-8B (16K)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.26,
                        left=0.08, right=0.98, top=0.96, bottom=0.18)
    axes_flat = axes.flatten()

    for i, (plabel, task, model, title) in enumerate(LAYOUT):
        key = (task, model)
        _plot_panel(
            axes_flat[i],
            RESULTS_MG[key],
            BUDGETS_MG,
            plabel, title,
            show_ylabel=(i % 2 == 0),   # only left column gets y-label
        )

    # ── Shared legend (bottom centre) ────────────────────────────────────
    seen, handles, labels = set(), [], []
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)

    # order: baseline first, then METHOD_ORDER
    order = ["Full KV (baseline)"] + METHOD_ORDER
    pairs = {l: h for l, h in zip(labels, handles)}
    final_h = [pairs[l] for l in order if l in pairs]
    final_l = [l        for l in order if l in pairs]

    fig.legend(
        final_h, final_l,
        loc="lower center",
        ncol=4,
        frameon=True, edgecolor="#AAAAAA",
        columnspacing=0.7, handlelength=1.8, handletextpad=0.4,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(out_path, bbox_inches="tight")
    print(f"[Figure 1] saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — AIME 2024-2025  (1 × 2 with ±1σ error bars)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure_aime(out_path):
    MODELS   = ["Qwen3-8B", "DeepSeek-8B"]
    PLABELS  = ["(a)", "(b)"]
    XTICK_L  = ["2K", "4K", "8K", "16K"]

    CORE_STYLE = {
        "NI (Ours)": dict(color="#C0392B", lw=2.0, ls="-",  marker="*",
                          ms=9, markeredgewidth=0.4, markeredgecolor="#7B241C"),
        "LRU":       dict(color="#2471A3", lw=1.8, ls="-",  marker="o",
                          ms=6, markeredgewidth=0.4, markeredgecolor="#1A5276"),
        "RaaS":      dict(color="#D35400", lw=1.2, ls=":",  marker="D",
                          ms=4.5, alpha=0.80, markeredgewidth=0.3, markeredgecolor="#A04000"),
        "R-KV":      dict(color="#7D3C98", lw=1.2, ls=":",  marker="P",
                          ms=5, alpha=0.80, markeredgewidth=0.3, markeredgecolor="#512E5F"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    fig.subplots_adjust(wspace=0.30,
                        left=0.09, right=0.98, top=0.88, bottom=0.24)

    for i, (model, plabel) in enumerate(zip(MODELS, PLABELS)):
        ax = axes[i]
        d = RESULTS_AIME[model]
        bl_mean, bl_std = d["baseline"]

        # Baseline line + ±1σ band
        ax.axhline(bl_mean, label="Full KV (baseline)", **BASELINE_STYLE)
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std,
                   color="#2C3E50", alpha=0.07, zorder=0)

        for method in ["NI (Ours)", "LRU", "RaaS", "R-KV"]:
            vals = d[method]
            budgets_valid = [b for b, v in zip(BUDGETS_A, vals) if v is not None]
            means_valid   = [v[0] for v in vals if v is not None]
            stds_valid    = [v[1] for v in vals if v is not None]
            if not budgets_valid:
                continue
            s = CORE_STYLE[method]
            ax.errorbar(
                budgets_valid, means_valid, yerr=stds_valid,
                label=method,
                capsize=3.5, capthick=1.2, elinewidth=1.1,
                **s,
            )

        ax.set_title(f"{plabel} {model}", loc="left",
                     fontsize=9.5, fontweight="bold", pad=4)
        ax.set_xlabel("KV Budget (think tokens)", labelpad=2)
        if i == 0:
            ax.set_ylabel("pass@1 Accuracy", labelpad=2)

        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS_A)
        ax.set_xticklabels(XTICK_L)
        ax.tick_params(which="both", direction="in", top=True, right=True, length=3)
        ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # y-limits: same range for both panels → easier visual comparison
    y_all = []
    for model in MODELS:
        d = RESULTS_AIME[model]
        bl_m, bl_s = d["baseline"]
        y_all.append(bl_m)
        for method in ["NI (Ours)", "LRU", "RaaS", "R-KV"]:
            if method in d:
                y_all += [v[0] for v in d[method] if v is not None]
    lo, hi = min(y_all), max(y_all)
    mg = (hi - lo) * 0.12
    for ax in axes:
        ax.set_ylim(lo - mg, hi + mg)

    # Shared legend
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3,
               frameon=True, edgecolor="#AAAAAA",
               columnspacing=1.0, handlelength=1.8,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "AIME 2024–2025  (pass@1, mean ± std over 3 seeds)",
        fontsize=9.5, y=0.98,
    )

    fig.savefig(out_path, bbox_inches="tight")
    print(f"[Figure AIME] saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_figure1("analysis/plots/figure1_main_results.pdf")
    #make_figure_aime("/mnt/user-data/outputs/figure2_aime.pdf")
    print("All figures done.")