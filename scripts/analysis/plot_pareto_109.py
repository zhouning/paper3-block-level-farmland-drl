# -*- coding: utf-8 -*-
"""Pareto front visualization for Paper 3 Block-Level results (Township 109)."""

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

# Data: [method, slope_pct, baimu_ha, cont]
baselines = [
    ('Greedy-Global',   -6.36, -31.5,  +0.439),
    ('Greedy-Seq',      -2.12, -12.0,  +0.050),
    ('Random (avg)',    -0.79,  +1.4,  +0.051),
    ('Round-Robin',     -0.33, +43.8,  +0.087),
]

drl_seeds = [
    ('DRL seed0', -2.67,  +7.8, +0.149),
    ('DRL seed1', -1.78, +24.8, +0.154),
    ('DRL seed2', -2.92,  +9.6, +0.092),
    ('DRL seed3', -2.92,  -4.9, +0.140),
    ('DRL seed4', -2.74,  +6.7, +0.113),
]
drl_mean = ('DRL mean', -2.61, +8.8, +0.130)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ============================================================
# Plot 1: Slope vs Baimu Area
# ============================================================
ax = axes[0]

# Baselines
for name, slope, baimu, cont in baselines:
    ax.scatter(-slope, baimu, s=120, zorder=5, marker='s', edgecolors='black', linewidth=1)
    offset_x, offset_y = 0.1, 2
    if name == 'Round-Robin':
        offset_x, offset_y = -0.3, -5
    elif name == 'Greedy-Global':
        offset_x, offset_y = -0.3, 3
    ax.annotate(name, (-slope, baimu), textcoords="offset points",
                xytext=(offset_x*30, offset_y), fontsize=10, ha='center')

# DRL individual seeds
for name, slope, baimu, cont in drl_seeds:
    ax.scatter(-slope, baimu, s=60, color='#2196F3', alpha=0.6, zorder=4, edgecolors='white', linewidth=0.5)

# DRL mean
ax.scatter(-drl_mean[1], drl_mean[2], s=200, color='#2196F3', zorder=6,
           marker='*', edgecolors='black', linewidth=1)
ax.annotate('DRL (mean)', (-drl_mean[1], drl_mean[2]), textcoords="offset points",
            xytext=(40, -15), fontsize=11, fontweight='bold', color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

# Pareto front line (connecting non-dominated points)
# Pareto-optimal: Greedy-Global (best slope), DRL mean (good both), Round-Robin (best baimu)
pareto_x = [6.36, 2.61, 0.33]
pareto_y = [-31.5, 8.8, 43.8]
ax.plot(pareto_x, pareto_y, '--', color='red', alpha=0.7, linewidth=2, label='Pareto front')

# Shade dominated region
ax.fill_between([0, 7], [-40, -40], [-31.5, -31.5], alpha=0.05, color='red')

# Labels
ax.set_xlabel('Slope reduction (%, larger = better)', fontsize=12)
ax.set_ylabel('Baimu area change (ha, larger = better)', fontsize=12)
ax.set_title('(a) Slope vs Baimu Area', fontsize=14)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

# Mark ideal direction
ax.annotate('', xy=(6.5, 45), xytext=(5, 30),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(6.3, 28, 'Ideal\ndirection', fontsize=9, color='green', ha='center')

# ============================================================
# Plot 2: Slope vs Contiguity
# ============================================================
ax = axes[1]

for name, slope, baimu, cont in baselines:
    ax.scatter(-slope, cont, s=120, zorder=5, marker='s', edgecolors='black', linewidth=1)
    offset_x, offset_y = 0, 0.015
    if name == 'Greedy-Seq':
        offset_x, offset_y = 0, -0.025
    elif name == 'Round-Robin':
        offset_x, offset_y = 0.3, 0.01
    elif name == 'Random (avg)':
        offset_x, offset_y = 0, -0.025
    ax.annotate(name, (-slope, cont), textcoords="offset points",
                xytext=(offset_x*30, offset_y*500), fontsize=10, ha='center')

for name, slope, baimu, cont in drl_seeds:
    ax.scatter(-slope, cont, s=60, color='#2196F3', alpha=0.6, zorder=4, edgecolors='white', linewidth=0.5)

ax.scatter(-drl_mean[1], drl_mean[3], s=200, color='#2196F3', zorder=6,
           marker='*', edgecolors='black', linewidth=1)
ax.annotate('DRL (mean)', (-drl_mean[1], drl_mean[3]), textcoords="offset points",
            xytext=(45, -10), fontsize=11, fontweight='bold', color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

# Pareto front: Greedy-Global (best both), DRL is dominated by GG on this pair
# But DRL dominates Greedy-Seq and Random
pareto_x2 = [6.36, 2.61, 0.33]
pareto_y2 = [0.439, 0.130, 0.087]
ax.plot(pareto_x2, pareto_y2, '--', color='red', alpha=0.7, linewidth=2, label='Pareto front')

ax.set_xlabel('Slope reduction (%, larger = better)', fontsize=12)
ax.set_ylabel('Contiguity change (larger = better)', fontsize=12)
ax.set_title('(b) Slope vs Contiguity', fontsize=14)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

ax.annotate('', xy=(6.5, 0.45), xytext=(5, 0.35),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(6.3, 0.34, 'Ideal\ndirection', fontsize=9, color='green', ha='center')

plt.suptitle('Township 109 (A-Small): Multi-Objective Pareto Analysis\n'
             'Block-Level MDP — DRL vs Baselines', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_109.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_109.png")
