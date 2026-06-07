# -*- coding: utf-8 -*-
"""3D Pareto front visualization for Paper 3 Block-Level results (Township 109)."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys
from mpl_toolkits.mplot3d import Axes3D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# Data: [method, slope_reduction(positive=better), baimu_ha, cont]
baselines = {
    'Greedy-Global':  (6.36, -31.5,  0.439),
    'Greedy-Seq':     (2.12, -12.0,  0.050),
    'Random (avg)':   (0.79,  +1.4,  0.051),
    'Round-Robin':    (0.33, +43.8,  0.087),
}

drl_seeds = [
    (2.67,  +7.8, 0.149),
    (1.78, +24.8, 0.154),
    (2.92,  +9.6, 0.092),
    (2.92,  -4.9, 0.140),
    (2.74,  +6.7, 0.113),
]
drl_mean = (2.61, +8.8, 0.130)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Color scheme
colors_bl = {
    'Greedy-Global': '#E53935',
    'Greedy-Seq':    '#FF9800',
    'Random (avg)':  '#9E9E9E',
    'Round-Robin':   '#4CAF50',
}

# Plot baselines
for name, (s, b, c) in baselines.items():
    ax.scatter(s, b, c, s=150, c=colors_bl[name], marker='s',
               edgecolors='black', linewidth=1, zorder=5, label=name)
    # Add vertical drop lines to help 3D perception
    ax.plot([s, s], [b, b], [0, c], color=colors_bl[name], alpha=0.3, linewidth=1, linestyle=':')

# Plot DRL seeds
for i, (s, b, c) in enumerate(drl_seeds):
    ax.scatter(s, b, c, s=80, c='#2196F3', alpha=0.5,
               edgecolors='white', linewidth=0.5, zorder=4)
    ax.plot([s, s], [b, b], [0, c], color='#2196F3', alpha=0.15, linewidth=1, linestyle=':')

# Plot DRL mean
ax.scatter(*drl_mean, s=300, c='#2196F3', marker='*',
           edgecolors='black', linewidth=1.2, zorder=6, label='DRL (mean)')
ax.plot([drl_mean[0], drl_mean[0]], [drl_mean[1], drl_mean[1]],
        [0, drl_mean[2]], color='#1565C0', alpha=0.4, linewidth=1.5, linestyle=':')

# Draw Pareto front surface (triangulated between non-dominated points)
# Pareto-optimal points: Greedy-Global, DRL mean, Round-Robin
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
pareto_pts = np.array([
    [6.36, -31.5, 0.439],  # Greedy-Global
    [2.61,   8.8, 0.130],  # DRL mean
    [0.33,  43.8, 0.087],  # Round-Robin
])
verts = [list(zip(pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2]))]
poly = Poly3DCollection(verts, alpha=0.12, facecolor='red', edgecolor='red', linewidth=2, linestyle='--')
ax.add_collection3d(poly)

# Draw Pareto front edges
for i in range(len(pareto_pts)):
    j = (i + 1) % len(pareto_pts)
    ax.plot([pareto_pts[i, 0], pareto_pts[j, 0]],
            [pareto_pts[i, 1], pareto_pts[j, 1]],
            [pareto_pts[i, 2], pareto_pts[j, 2]],
            'r--', linewidth=2, alpha=0.7)

# Project shadows onto walls for better 3D reading
# XY plane (z=0): slope vs baimu
for name, (s, b, c) in baselines.items():
    ax.scatter(s, b, 0, s=40, c=colors_bl[name], alpha=0.2, marker='s')
ax.scatter(drl_mean[0], drl_mean[1], 0, s=80, c='#2196F3', alpha=0.2, marker='*')

# XZ plane (y=-40): slope vs cont
for name, (s, b, c) in baselines.items():
    ax.scatter(s, -40, c, s=40, c=colors_bl[name], alpha=0.2, marker='s')
ax.scatter(drl_mean[0], -40, drl_mean[2], s=80, c='#2196F3', alpha=0.2, marker='*')

# Annotations with 3D offset
ax.text(6.36 + 0.2, -31.5, 0.439 + 0.02, 'Greedy-Global\n(slope best)', fontsize=9, color='#C62828')
ax.text(0.33 - 0.5, 43.8, 0.087 + 0.02, 'Round-Robin\n(baimu best)', fontsize=9, color='#2E7D32')
ax.text(2.12 + 0.2, -12.0, 0.050 - 0.03, 'Greedy-Seq\n(dominated)', fontsize=9, color='#E65100')
ax.text(drl_mean[0] + 0.3, drl_mean[1] + 3, drl_mean[2] + 0.02,
        'DRL (mean)\nBalanced\noptimal', fontsize=10, fontweight='bold', color='#1565C0')

# Ideal direction arrow
ax.quiver(5.5, 30, 0.35, 0.8, 8, 0.06, color='green', alpha=0.8,
          arrow_length_ratio=0.3, linewidth=2)
ax.text(6.5, 40, 0.42, 'Ideal', fontsize=10, color='green', fontweight='bold')

# Axis labels
ax.set_xlabel('\nSlope reduction (%)\n(larger = better)', fontsize=11, labelpad=10)
ax.set_ylabel('\nBaimu area change (ha)\n(larger = better)', fontsize=11, labelpad=10)
ax.set_zlabel('\nContiguity change\n(larger = better)', fontsize=11, labelpad=10)

ax.set_title('Township 109: 3D Pareto Front\nBlock-Level MDP — DRL vs Baselines',
             fontsize=14, fontweight='bold', pad=20)

# Set viewing angle
ax.view_init(elev=25, azim=135)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_3d_109.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_3d_109.png")

# Also generate a second angle for better understanding
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# Replot everything
for name, (s, b, c) in baselines.items():
    ax2.scatter(s, b, c, s=150, c=colors_bl[name], marker='s',
                edgecolors='black', linewidth=1, zorder=5, label=name)
    ax2.plot([s, s], [b, b], [0, c], color=colors_bl[name], alpha=0.3, linewidth=1, linestyle=':')

for i, (s, b, c) in enumerate(drl_seeds):
    ax2.scatter(s, b, c, s=80, c='#2196F3', alpha=0.5,
                edgecolors='white', linewidth=0.5, zorder=4)

ax2.scatter(*drl_mean, s=300, c='#2196F3', marker='*',
            edgecolors='black', linewidth=1.2, zorder=6, label='DRL (mean)')

ax2.add_collection3d(Poly3DCollection(verts, alpha=0.12, facecolor='red',
                                       edgecolor='red', linewidth=2, linestyle='--'))
for i in range(len(pareto_pts)):
    j = (i + 1) % len(pareto_pts)
    ax2.plot([pareto_pts[i, 0], pareto_pts[j, 0]],
             [pareto_pts[i, 1], pareto_pts[j, 1]],
             [pareto_pts[i, 2], pareto_pts[j, 2]],
             'r--', linewidth=2, alpha=0.7)

ax2.text(6.36 + 0.1, -31.5 - 3, 0.439, 'Greedy-Global', fontsize=9, color='#C62828')
ax2.text(0.33, 43.8 + 3, 0.087, 'Round-Robin', fontsize=9, color='#2E7D32')
ax2.text(drl_mean[0] + 0.3, drl_mean[1], drl_mean[2] + 0.02,
         'DRL', fontsize=11, fontweight='bold', color='#1565C0')

ax2.set_xlabel('\nSlope reduction (%)', fontsize=11, labelpad=10)
ax2.set_ylabel('\nBaimu area (ha)', fontsize=11, labelpad=10)
ax2.set_zlabel('\nContiguity', fontsize=11, labelpad=10)
ax2.set_title('Township 109: 3D Pareto Front (alternate view)',
              fontsize=14, fontweight='bold', pad=20)
ax2.view_init(elev=20, azim=45)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_pareto_3d_109_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_pareto_3d_109_v2.png")
