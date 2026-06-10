# -*- coding: utf-8 -*-
"""Generate framework architecture diagram for Paper 3."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Colors ──
C_TITLE = '#1A237E'
C_DRL = '#E3F2FD'
C_DRL_BORDER = '#1565C0'
C_GREEDY = '#FFF3E0'
C_GREEDY_BORDER = '#E65100'
C_REWARD = '#E8F5E9'
C_REWARD_BORDER = '#2E7D32'
C_ENV = '#F3E5F5'
C_ENV_BORDER = '#6A1B9A'
C_ARROW = '#424242'

def add_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9,
            text_color='black', bold=False, alpha=0.9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=weight,
            wrap=True, linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ── Title ──
ax.text(8, 8.6, 'Framework: "Macroscopic Planning, Microscopic Execution"',
        ha='center', va='center', fontsize=16, fontweight='bold', color=C_TITLE)

# ── Left Panel: Macroscopic Planning ──
ax.add_patch(FancyBboxPatch((0.3, 1.0), 6.4, 7.0, boxstyle="round,pad=0.2",
             facecolor='#F5F5F5', edgecolor=C_DRL_BORDER, linewidth=2.5, alpha=0.3))
ax.text(3.5, 7.7, 'Macroscopic Planning', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C_DRL_BORDER)
ax.text(3.5, 7.3, '(DRL Agent)', ha='center', va='center',
        fontsize=11, color=C_DRL_BORDER, style='italic')

# State observation
add_box(ax, 0.6, 6.0, 5.8, 1.0,
        'State Observation\n'
        'Per-block features $\\mathbf{X}_t$ ($N_b \\times 17$)  +  Global state $\\mathbf{g}_t$ ($9$)',
        C_DRL, C_DRL_BORDER, fontsize=9, bold=True)

# Policy network
add_box(ax, 0.6, 4.5, 5.8, 1.1,
        'BlockScoringPolicy (14,530 params)\n'
        'Scorer: [26 → 128 → 64 → 1] per block\n'
        'Value:  [9 → 64 → 32 → 1] global',
        C_DRL, C_DRL_BORDER, fontsize=9, bold=True)

# Action selection
add_box(ax, 0.6, 3.0, 5.8, 1.0,
        'Masked Softmax → Select Block $a_t$\n'
        'Action mask: blocks with farmland & forest parcels',
        C_DRL, C_DRL_BORDER, fontsize=9, bold=True)

# Budget
add_box(ax, 1.2, 1.5, 4.6, 0.9,
        'Budget: 20 steps × 5 swaps = 100 total\n'
        'Each selection: 5% of total budget',
        '#ECEFF1', '#546E7A', fontsize=8.5, bold=False)

arrow(ax, 3.5, 6.0, 3.5, 5.6)
arrow(ax, 3.5, 4.5, 3.5, 4.0)

# ── Right Panel: Microscopic Execution ──
ax.add_patch(FancyBboxPatch((7.3, 1.0), 8.4, 7.0, boxstyle="round,pad=0.2",
             facecolor='#F5F5F5', edgecolor=C_GREEDY_BORDER, linewidth=2.5, alpha=0.3))
ax.text(11.5, 7.7, 'Microscopic Execution', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C_GREEDY_BORDER)
ax.text(11.5, 7.3, '(Connectivity-Aware Greedy Engine)', ha='center', va='center',
        fontsize=11, color=C_GREEDY_BORDER, style='italic')

# Selected block
add_box(ax, 7.6, 6.0, 3.5, 1.0,
        'Selected Block $b$\n'
        'Contains farmland + forest parcels',
        C_GREEDY, C_GREEDY_BORDER, fontsize=9, bold=True)

# Removal scoring
add_box(ax, 7.6, 4.6, 3.5, 1.0,
        'Farmland Removal\n'
        '$score = s_i - 0.5 \\cdot |F_{nbr}|$\n'
        'Steep + isolated → remove',
        '#FFECB3', '#FF8F00', fontsize=8.5, bold=True)

# Conversion scoring
add_box(ax, 11.7, 4.6, 3.7, 1.0,
        'Forest Conversion\n'
        '$score = s_j - 1.0 \\cdot |F_{nbr}|$\n'
        'Gentle + F-adjacent → convert',
        '#C8E6C9', '#388E3C', fontsize=8.5, bold=True)

# Execute swaps
add_box(ax, 7.6, 3.0, 7.8, 1.0,
        'Execute $\\kappa = 5$ paired swaps per step\n'
        'Constraint: $slope_{remove} > slope_{convert}$  (each swap must improve slope)',
        C_GREEDY, C_GREEDY_BORDER, fontsize=9, bold=True)

# Reward
add_box(ax, 7.6, 1.5, 7.8, 0.9,
        'Reward: $r_t = 2000 \\cdot \\Delta\\bar{S} + 500 \\cdot \\Delta\\bar{C} '
        '+ 500 \\cdot \\Delta A_B + 20 \\cdot \\Delta n_B$',
        C_REWARD, C_REWARD_BORDER, fontsize=9, bold=True)

arrow(ax, 9.35, 6.0, 9.35, 5.6)
arrow(ax, 13.5, 6.0, 13.5, 5.6)
arrow(ax, 9.35, 4.6, 11.5, 4.0)
arrow(ax, 13.5, 4.6, 11.5, 4.0)
arrow(ax, 11.5, 3.0, 11.5, 2.4)

# ── Connecting arrows ──
# DRL → Greedy (block selection)
ax.annotate('', xy=(7.6, 6.5), xytext=(6.4, 6.5),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=3))
ax.text(7.0, 6.85, 'Block\nselection', ha='center', va='bottom',
        fontsize=8, color='#D32F2F', fontweight='bold')

# Reward → DRL (feedback)
ax.annotate('', xy=(3.5, 1.5), xytext=(7.6, 1.9),
            arrowprops=dict(arrowstyle='->', color=C_REWARD_BORDER, lw=2.5,
                           connectionstyle='arc3,rad=0.3'))
ax.text(5.5, 0.9, 'Reward feedback', ha='center', va='center',
        fontsize=9, color=C_REWARD_BORDER, fontweight='bold')

# ── Spatial spillover annotation ──
ax.add_patch(FancyBboxPatch((11.8, 6.2), 3.5, 0.7, boxstyle="round,pad=0.1",
             facecolor='#FCE4EC', edgecolor='#C62828', linewidth=1.5, alpha=0.8))
ax.text(13.55, 6.55, 'Spatial Spillover\n'
        'Cross-block connectivity changes',
        ha='center', va='center', fontsize=8, color='#C62828',
        fontweight='bold', style='italic')

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out_path = FIGURES_DIR / 'paper3_framework_diagram.png'
plt.savefig(out_path, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {out_path}")
