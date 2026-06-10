# -*- coding: utf-8 -*-
"""Generate the cross-township bar chart used in the Paper 3 CEUS manuscript.

The values are hard-coded from the current manuscript main-results table so the
figure can be regenerated without controlled-access parcel geometry.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402


TOWNSHIPS = ["A\n(78 blk)", "B\n(132 blk)", "C\n(338 blk)"]
METHODS = [
    "Greedy-Global",
    "Greedy-Seq",
    "Random",
    "Round-Robin",
    "Reward-Greedy",
    "DRL",
]
COLORS = ["#2E7D32", "#616161", "#D4A017", "#9C27B0", "#C0392B", "#2980B9"]

SLOPE = np.array([
    [-6.36, -3.79, -1.48],
    [-2.12, -1.85, -1.17],
    [-0.79, -1.31, -0.24],
    [-0.33, -0.28, +1.10],
    [-3.03, -4.24, -0.52],
    [-2.61, -1.64, -0.33],
])

BAIMU_HA = np.array([
    [-31.5,  -6.9, +34.6],
    [-12.0, -60.4, -43.0],
    [ +1.4, -38.1,  +9.6],
    [+43.8,  -8.9,+114.4],
    [+13.5,-101.9, +28.0],
    [ +8.8, +12.9, +27.7],
])

CONTIGUITY = np.array([
    [+0.439, +0.226, +0.121],
    [+0.050, +0.003, +0.020],
    [+0.051, +0.032, +0.026],
    [+0.087, +0.062, +0.047],
    [+0.198, +0.036, +0.045],
    [+0.130, +0.115, +0.023],
])

BAIMU_COUNT = np.array([
    [-2.0, -1.0, -1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -0.6, +0.8],
    [-1.0, -1.0, +2.0],
    [ 0.0,  0.0, +6.0],
    [ 0.0, -1.0, +5.0],
])


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(TOWNSHIPS))
    width = 0.12

    panels = [
        (SLOPE, "Slope change (%)", "(a) Slope reduction"),
        (BAIMU_HA, "Baimu fang area change (ha)", "(b) Baimu fang area"),
        (CONTIGUITY, "Contiguity change", "(c) Contiguity"),
        (BAIMU_COUNT, "Baimu fang count change", "(d) Baimu fang count"),
    ]

    for ax, (values, ylabel, title) in zip(axes.flat, panels):
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            offset = (i - (len(METHODS) - 1) / 2) * width
            bars = ax.bar(x + offset, values[i], width, label=method,
                          color=color, edgecolor="white", linewidth=0.5)
            if method == "DRL":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.3)

        ax.set_xticks(x)
        ax.set_xticklabels(TOWNSHIPS)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(True, alpha=0.2, axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle("Cross-township comparison of block-selection methods",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out_path = FIGURES_DIR / "paper3_cross_township_bars.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
