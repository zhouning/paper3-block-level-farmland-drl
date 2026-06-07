# -*- coding: utf-8 -*-
"""Regenerate the two Pareto figures used in Section 4 (Multi-objective Pareto
analysis) so that the Reward-Greedy one-step lookahead baseline appears as a
red triangle, matching the caption text added in the latest revision.

Two figures:
  paper3_pareto_all_slope_baimu.png    -- slope vs baimu_area (3 panels A/B/C)
  paper3_pareto_all_slope_baimucnt.png -- slope vs baimu_count (3 panels A/B/C)

Per panel:
  - Greedy-Global (square, dark green)
  - Greedy-Sequential (square, dark gray)
  - Random (square, gold; mean over 5 seeds)
  - Round-Robin (square, magenta)
  - Reward-Greedy (red triangle)
  - DRL: 5 seeds as small circles + mean as star (blue)

Reads numbers directly from the manuscript Table 4 hard-coded here, so this
script is self-contained and reproduces the figure files identically each run.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402

# Hard-coded from manuscript Table 4 (paper3_block_level.tex)
RESULTS = {
    "A": {
        "Greedy-Global":   {"slope": -6.36, "cont": +0.439, "cnt": -2,    "ha": -31.5},
        "Greedy-Seq":      {"slope": -2.12, "cont": +0.050, "cnt": -1,    "ha": -12.0},
        "Random":          {"slope": -0.79, "cont": +0.051, "cnt":  0.0,  "ha":  +1.4},
        "Round-Robin":     {"slope": -0.33, "cont": +0.087, "cnt": -1,    "ha": +43.8},
        "Reward-Greedy":   {"slope": -3.03, "cont": +0.198, "cnt":  0,    "ha": +13.5},
        "DRL_seeds": [
            (-2.67,  0, +14.4), (-1.78,  0, -10.4),
            (-2.92,  0, +14.4), (-2.92,  0, +14.4), (-2.74,  0, +11.0),
        ],
        "DRL_mean": {"slope": -2.61, "cnt":  0,   "ha":  +8.8},
    },
    "B": {
        "Greedy-Global":   {"slope": -3.79, "cont": +0.226, "cnt": -1,    "ha":  -6.9},
        "Greedy-Seq":      {"slope": -1.85, "cont": +0.003, "cnt":  0,    "ha": -60.4},
        "Random":          {"slope": -1.31, "cont": +0.032, "cnt": -0.6,  "ha": -38.1},
        "Round-Robin":     {"slope": -0.28, "cont": +0.062, "cnt": -1,    "ha":  -8.9},
        "Reward-Greedy":   {"slope": -4.24, "cont": +0.036, "cnt":  0,    "ha": -101.9},
        "DRL_seeds": [
            (-1.64, -1, +10.5), (-1.56, -1, +11.2),
            (-1.63, -1, +18.2), (-1.66, -1, +15.0), (-1.69, -1,  +9.6),
        ],
        "DRL_mean": {"slope": -1.64, "cnt": -1,   "ha": +12.9},
    },
    "C": {
        "Greedy-Global":   {"slope": -1.48, "cont": +0.121, "cnt": -1,    "ha": +34.6},
        "Greedy-Seq":      {"slope": -1.17, "cont": +0.020, "cnt":  0,    "ha": -43.0},
        "Random":          {"slope": -0.24, "cont": +0.026, "cnt": +0.8,  "ha":  +9.6},
        "Round-Robin":     {"slope": +1.10, "cont": +0.047, "cnt": +2,    "ha": +114.4},
        "Reward-Greedy":   {"slope": -0.52, "cont": +0.045, "cnt": +6,    "ha": +28.0},
        "DRL_seeds": [
            (-0.32, +5, +28.4), (-0.35, +5, +27.5),
            (-0.33, +5, +27.7), (-0.34, +5, +27.0), (-0.33, +5, +27.9),
        ],
        "DRL_mean": {"slope": -0.33, "cnt": +5,   "ha": +27.7},
    },
}

BASELINE_STYLE = {
    "Greedy-Global":   {"marker": "s", "color": "#1e7e34", "label": "Greedy-Global",  "size": 70},
    "Greedy-Seq":      {"marker": "s", "color": "#444444", "label": "Greedy-Seq",     "size": 70},
    "Random":          {"marker": "s", "color": "#d4a017", "label": "Random",         "size": 70},
    "Round-Robin":     {"marker": "s", "color": "#a040a0", "label": "Round-Robin",    "size": 70},
    "Reward-Greedy":   {"marker": "^", "color": "#c0392b", "label": "Reward-Greedy",  "size": 95},
}

DRL_COLOR = "#2980b9"


def render_panel(ax, t_label: str, y_key: str, y_label: str) -> None:
    rows = RESULTS[t_label]

    # Baselines
    for name, vals in [(n, rows[n]) for n in BASELINE_STYLE]:
        st = BASELINE_STYLE[name]
        ax.scatter(vals["slope"], vals[y_key],
                   marker=st["marker"], color=st["color"],
                   s=st["size"], zorder=3, label=st["label"],
                   edgecolors="black", linewidths=0.5)

    # DRL seeds (small) + mean (star)
    for s, c, h in rows["DRL_seeds"]:
        y = c if y_key == "cnt" else h
        ax.scatter(s, y, marker="o", color=DRL_COLOR, s=30, zorder=4,
                   edgecolors="white", linewidths=0.5, alpha=0.85)
    ax.scatter(rows["DRL_mean"]["slope"], rows["DRL_mean"][y_key],
               marker="*", color=DRL_COLOR, s=260, zorder=5,
               edgecolors="black", linewidths=0.7, label="DRL (mean of 5 seeds)")

    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Slope change (%)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Township {t_label}")

    # ideal-direction arrow (lower slope, higher y)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.annotate("", xy=(xlim[0] + 0.05 * (xlim[1] - xlim[0]),
                        ylim[1] - 0.05 * (ylim[1] - ylim[0])),
                xytext=(xlim[0] + 0.25 * (xlim[1] - xlim[0]),
                        ylim[1] - 0.25 * (ylim[1] - ylim[0])),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
    ax.text(xlim[0] + 0.27 * (xlim[1] - xlim[0]),
            ylim[1] - 0.27 * (ylim[1] - ylim[0]),
            "ideal", color="#27ae60", fontsize=8, va="top")


def render_figure(y_key: str, y_label: str, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, lab in zip(axes, ["A", "B", "C"]):
        render_panel(ax, lab, y_key, y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05),
               fontsize=9, frameon=False)

    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main() -> None:
    render_figure(
        y_key="ha",
        y_label="Baimu fang area change (ha)",
        out_path=FIGURES_DIR / "paper3_pareto_all_slope_baimu.png",
        title="Pareto: slope reduction vs. baimu fang area change",
    )
    render_figure(
        y_key="cnt",
        y_label="Baimu fang count change",
        out_path=FIGURES_DIR / "paper3_pareto_all_slope_baimucnt.png",
        title="Pareto: slope reduction vs. baimu fang count change",
    )


if __name__ == "__main__":
    main()
