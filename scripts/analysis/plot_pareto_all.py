# -*- coding: utf-8 -*-
"""Pareto front visualization for Paper 3 — all 3 townships."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================================================================
# Data for all 3 townships: (method, slope_pct, baimu_ha, cont, baimu_cnt)
# ======================================================================

data_109 = {
    'baselines': [
        ('Greedy-Global',   -6.36, -31.5,  +0.439, -2),
        ('Greedy-Seq',      -2.12, -12.0,  +0.050, -1),
        ('Random (avg)',    -0.79,  +1.4,  +0.051,  0),
        ('Round-Robin',     -0.33, +43.8,  +0.087, -1),
    ],
    'drl_seeds': [
        ('s0', -2.67,  +7.8, +0.149, 0),
        ('s1', -1.78, +24.8, +0.154, 0),
        ('s2', -2.92,  +9.6, +0.092, 0),
        ('s3', -2.92,  -4.9, +0.140, 0),
        ('s4', -2.74,  +6.7, +0.113, 0),
    ],
    'drl_mean': ('DRL mean', -2.61, +8.8, +0.130, 0),
    'title': 'Township 109 (A-Small, 78 blocks)',
}

data_108 = {
    'baselines': [
        ('Greedy-Global',   -3.79,  -6.9,  +0.226, -1),
        ('Greedy-Seq',      -1.85, -60.4,  +0.003,  0),
        ('Random (avg)',    -1.31, -38.1,  +0.032, -0.6),
        ('Round-Robin',     -0.28,  -8.9,  +0.062, -1),
    ],
    'drl_seeds': [
        ('s0', -3.53,  -92.6, +0.006, 1),
        ('s1', -3.50,  -97.7, -0.001, 1),
        ('s2', -3.43,  -90.9, +0.019, 1),
        ('s3', -3.49,  -94.4, +0.010, 1),
        ('s4', -3.90, -111.9, +0.002, 1),
    ],
    'drl_mean': ('DRL mean', -3.59, -97.5, +0.007, 1),
    'title': 'Township 108 (B-Medium, 132 blocks)',
}

data_105 = {
    'baselines': [
        ('Greedy-Global',   -1.48, +34.6,  +0.121, -1),
        ('Greedy-Seq',      -1.17, -43.0,  +0.020,  0),
        ('Random (avg)',    -0.24,  +9.6,  +0.026, +0.8),
        ('Round-Robin',     +1.10,+114.4,  +0.047, +2),
    ],
    'drl_seeds': [
        ('s0', -0.32, +29.2, +0.018, 5),
        ('s1', -0.35, +27.8, +0.027, 5),
        ('s2', -0.33, +28.8, +0.024, 5),
        ('s3', -0.34, +25.3, +0.021, 5),
        ('s4', -0.33, +27.6, +0.025, 5),
    ],
    'drl_mean': ('DRL mean', -0.33, +27.7, +0.023, 5),
    'title': 'Township 105 (C-Large, 338 blocks)',
}

all_data = [data_109, data_108, data_105]

# ======================================================================
# Figure 1: 2D Pareto fronts — Slope vs Baimu Area (3 subplots)
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for idx, (d, ax) in enumerate(zip(all_data, axes)):
    # Baselines
    for name, slope, baimu, cont, bcnt in d['baselines']:
        ax.scatter(-slope, baimu, s=120, zorder=5, marker='s',
                   edgecolors='black', linewidth=1)
        # Offset annotations to avoid overlap
        ox, oy = 0, 3
        if name == 'Round-Robin':
            ox, oy = 0, -8
        elif name == 'Greedy-Global':
            ox, oy = 0, 4
        elif name == 'Random (avg)':
            ox, oy = 0, -8
        ax.annotate(name, (-slope, baimu), textcoords="offset points",
                    xytext=(ox, oy), fontsize=8.5, ha='center')

    # DRL seeds
    for name, slope, baimu, cont, bcnt in d['drl_seeds']:
        ax.scatter(-slope, baimu, s=60, color='#2196F3', alpha=0.6,
                   zorder=4, edgecolors='white', linewidth=0.5)

    # DRL mean
    dm = d['drl_mean']
    ax.scatter(-dm[1], dm[2], s=200, color='#2196F3', zorder=6,
               marker='*', edgecolors='black', linewidth=1)
    ax.annotate(f'DRL (mean)\nbaimu +{dm[4]}', (-dm[1], dm[2]),
                textcoords="offset points", xytext=(40, -15),
                fontsize=10, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    ax.set_xlabel('Slope reduction (%, larger=better)', fontsize=11)
    ax.set_ylabel('Baimu area change (ha)', fontsize=11)
    ax.set_title(d['title'], fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Ideal direction arrow
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.annotate('', xy=(xl[1]*0.9, yl[1]*0.85),
                xytext=(xl[1]*0.7, yl[1]*0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(xl[1]*0.8, yl[1]*0.55, 'Ideal', fontsize=9, color='green', ha='center')

plt.suptitle('Paper 3: Multi-Objective Pareto Analysis — Slope vs Baimu Area\n'
             'Block-Level MDP across 3 Townships',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_all_slope_baimu.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_all_slope_baimu.png")

# ======================================================================
# Figure 2: 2D Pareto fronts — Slope vs Contiguity (3 subplots)
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for idx, (d, ax) in enumerate(zip(all_data, axes)):
    for name, slope, baimu, cont, bcnt in d['baselines']:
        ax.scatter(-slope, cont, s=120, zorder=5, marker='s',
                   edgecolors='black', linewidth=1)
        ox, oy = 0, 5
        if name == 'Greedy-Seq':
            ox, oy = 0, -10
        elif name == 'Random (avg)':
            ox, oy = 0, -10
        ax.annotate(name, (-slope, cont), textcoords="offset points",
                    xytext=(ox, oy), fontsize=8.5, ha='center')

    for name, slope, baimu, cont, bcnt in d['drl_seeds']:
        ax.scatter(-slope, cont, s=60, color='#2196F3', alpha=0.6,
                   zorder=4, edgecolors='white', linewidth=0.5)

    dm = d['drl_mean']
    ax.scatter(-dm[1], dm[3], s=200, color='#2196F3', zorder=6,
               marker='*', edgecolors='black', linewidth=1)
    ax.annotate('DRL (mean)', (-dm[1], dm[3]),
                textcoords="offset points", xytext=(40, -10),
                fontsize=10, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    ax.set_xlabel('Slope reduction (%, larger=better)', fontsize=11)
    ax.set_ylabel('Contiguity change', fontsize=11)
    ax.set_title(d['title'], fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Paper 3: Multi-Objective Pareto Analysis — Slope vs Contiguity\n'
             'Block-Level MDP across 3 Townships',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_all_slope_cont.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_all_slope_cont.png")

# ======================================================================
# Figure 3: 2D Pareto fronts — Slope vs Baimu Count (3 subplots)
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for idx, (d, ax) in enumerate(zip(all_data, axes)):
    for name, slope, baimu, cont, bcnt in d['baselines']:
        ax.scatter(-slope, bcnt, s=120, zorder=5, marker='s',
                   edgecolors='black', linewidth=1)
        ox, oy = 0, 0.2
        if name == 'Greedy-Seq':
            ox, oy = 20, -0.2
        elif name == 'Random (avg)':
            ox, oy = 20, 0.1
        ax.annotate(name, (-slope, bcnt), textcoords="offset points",
                    xytext=(ox, oy*50), fontsize=8.5, ha='center')

    for name, slope, baimu, cont, bcnt in d['drl_seeds']:
        ax.scatter(-slope, bcnt, s=60, color='#2196F3', alpha=0.6,
                   zorder=4, edgecolors='white', linewidth=0.5)

    dm = d['drl_mean']
    ax.scatter(-dm[1], dm[4], s=200, color='#2196F3', zorder=6,
               marker='*', edgecolors='black', linewidth=1)
    ax.annotate(f'DRL (+{dm[4]})', (-dm[1], dm[4]),
                textcoords="offset points", xytext=(40, -10),
                fontsize=10, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    ax.set_xlabel('Slope reduction (%, larger=better)', fontsize=11)
    ax.set_ylabel('New baimu count', fontsize=11)
    ax.set_title(d['title'], fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.suptitle('Paper 3: Multi-Objective Pareto Analysis — Slope vs Baimu Count\n'
             'Block-Level MDP across 3 Townships',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_all_slope_baimucnt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_all_slope_baimucnt.png")

# ======================================================================
# Figure 4: Cross-township summary bar chart
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

townships = ['109\n(Small)', '108\n(Medium)', '105\n(Large)']
methods = ['Greedy-Global', 'Greedy-Seq', 'Random', 'Round-Robin', 'DRL']
colors = ['#E53935', '#FF9800', '#9E9E9E', '#4CAF50', '#2196F3']

# Slope data
slope_vals = [
    [-6.36, -3.79, -1.48],  # Greedy-Global
    [-2.12, -1.85, -1.17],  # Greedy-Seq
    [-0.79, -1.31, -0.24],  # Random
    [-0.33, -0.28, +1.10],  # Round-Robin
    [-2.61, -3.59, -0.33],  # DRL
]

# Baimu area
baimu_vals = [
    [-31.5,  -6.9, +34.6],
    [-12.0, -60.4, -43.0],
    [ +1.4, -38.1,  +9.6],
    [+43.8,  -8.9,+114.4],
    [ +8.8, -97.5, +27.7],
]

# Cont
cont_vals = [
    [+0.439, +0.226, +0.121],
    [+0.050, +0.003, +0.020],
    [+0.051, +0.032, +0.026],
    [+0.087, +0.062, +0.047],
    [+0.130, +0.007, +0.023],
]

# Baimu count
bcnt_vals = [
    [-2, -1, -1],
    [-1,  0,  0],
    [ 0, -0.6, +0.8],
    [-1, -1, +2],
    [ 0, +1, +5],
]

x = np.arange(len(townships))
width = 0.15

for panel_idx, (vals, ylabel, title) in enumerate([
    (slope_vals, 'Slope change (%)', '(a) Slope Reduction'),
    (baimu_vals, 'Baimu area change (ha)', '(b) Baimu Area Change'),
    (cont_vals, 'Contiguity change', '(c) Contiguity Change'),
    (bcnt_vals, 'New baimu count', '(d) New Baimu Count'),
]):
    ax = axes.flatten()[panel_idx]
    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals[i], width, label=method, color=color,
                      edgecolor='white', linewidth=0.5)
        # Bold DRL bars
        if method == 'DRL':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(townships, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')
    if panel_idx == 0:
        ax.legend(fontsize=8, loc='lower left')

plt.suptitle('Paper 3: Cross-Township Comparison — Block-Level MDP\n'
             'All methods, 4 metrics',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_cross_township_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_cross_township_bars.png")
