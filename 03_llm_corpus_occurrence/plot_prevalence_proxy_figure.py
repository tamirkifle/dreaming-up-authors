"""Plot Figure 2 of the paper: prevalence proxies vs. citation bucket (log2 scale).

Left panel: mean Google search hits per bucket after removing hit counts above
the 80th percentile within each bucket (module 02). Right panel: mean
Infini-gram n-gram count per bucket in the OLMo2-32B corpus with capitalized
title queries (module 03). Sample sizes are annotated at every point.

Input (produced by the two analysis notebooks):
    config.GOOGLE_HITS_DIR / citation_bucket_stats_80.csv
    config.INFINI_GRAM_DIR / citation_bucket_stats_olmo2_capitalize.csv
Output:
    config.FIGURES_DIR / proxy_means_by_citation_bin_log.pdf

Usage:
    python 03_llm_corpus_occurrence/plot_prevalence_proxy_figure.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GOOGLE_STATS_CSV = config.GOOGLE_HITS_DIR / "citation_bucket_stats_80.csv"
NGRAM_STATS_CSV = config.INFINI_GRAM_DIR / "citation_bucket_stats_olmo2_capitalize.csv"
OUT_PDF = config.FIGURES_DIR / "proxy_means_by_citation_bin_log.pdf"
BIN_TICK_LABELS = [label.replace("-", "–") for label in config.CITATION_BIN_LABELS]


def log2_formatter(value: float, _pos: int) -> str:
    """Format a log2-scale tick as the original (un-logged) value."""
    original = 2 ** value
    if original >= 1:
        return f"{int(original)}" if original == int(original) else f"{original:.1f}"
    return f"{original:.2f}"


def draw_panel(ax: Axes, means: np.ndarray, counts: np.ndarray, color: str, ylabel: str) -> None:
    """Draw one log2-scale panel with per-bucket sample sizes.

    Args:
        ax: Target axes.
        means: Mean value per citation bucket.
        counts: Number of papers per citation bucket.
        color: Line color.
        ylabel: Y-axis label.
    """
    x = np.arange(len(means))
    y = np.log2(means)
    ax.plot(x, y, marker="o", linewidth=2.5, color=color)
    ax.set_xlabel("Citation Count", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_TICK_LABELS, rotation=35, ha="right", fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(log2_formatter))
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    dy = (y_max - y_min) * 0.03
    for xi, yi, n in zip(x, y, counts):
        ax.text(xi, yi + dy, f"n={int(n)}", ha="center", va="bottom", fontsize=8, alpha=0.85)


def main() -> None:
    """Read both bucket-statistics tables and save the two-panel figure."""
    google = pd.read_csv(GOOGLE_STATS_CSV)
    ngram = pd.read_csv(NGRAM_STATS_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    draw_panel(axes[0], google["mean_hits"].to_numpy(), google["count"].to_numpy(),
               "#ff5733", "Mean Google Hits (Log2 Scale)")
    draw_panel(axes[1], ngram["mean_count"].to_numpy(), ngram["count"].to_numpy(),
               "#377eb8", "Mean n-gram Count (Log2 Scale)")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
