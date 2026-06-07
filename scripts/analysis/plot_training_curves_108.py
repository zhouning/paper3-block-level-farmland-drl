# -*- coding: utf-8 -*-
"""Training curves for Paper 3 Block-Level DRL (Township 108, 5 seeds)."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import BLOCK_RESULTS_DIR, FIGURES_DIR  # noqa: E402

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

LOG_DIR = BLOCK_RESULTS_DIR / 'township_500227108_v2'

all_data = {}
for seed in range(5):
    with open(LOG_DIR / f'block_training_log_seed{seed}.json') as f:
        all_data[seed] = json.load(f)
    print(f"Seed {seed}: {len(all_data[seed])} episodes, "
          f"timesteps {all_data[seed][0]['timestep']}-{all_data[seed][-1]['timestep']}")

def smooth(values, window=200):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

metrics = [
    ('reward', 'Episode Reward'),
    ('slope_change_pct', 'Slope Change (%)'),
    ('cont_change', 'Contiguity Change'),
    ('baimu_area_change_ha', 'Baimu Area Change (ha)'),
    ('budget_used', 'Budget Used (swaps)'),
]

baselines_ref = {
    'slope_change_pct': [
        (-1.85, 'Greedy-Seq', '#FF9800'),
        (-3.79, 'Greedy-Global', '#E53935'),
    ],
    'cont_change': [
        (0.003, 'Greedy-Seq', '#FF9800'),
        (0.226, 'Greedy-Global', '#E53935'),
    ],
    'baimu_area_change_ha': [
        (-60.4, 'Greedy-Seq', '#FF9800'),
        (-6.9, 'Greedy-Global', '#E53935'),
        (-8.9, 'Round-Robin', '#4CAF50'),
    ],
}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
window = 200

for idx, (key, title) in enumerate(metrics):
    ax = axes[idx]
    for seed in range(5):
        data = all_data[seed]
        timesteps = [d['timestep'] for d in data]
        values = [d[key] for d in data]
        ax.plot(timesteps, values, alpha=0.05, color=colors[seed], linewidth=0.5)
        ts_smooth = smooth(timesteps, window)
        val_smooth = smooth(values, window)
        ax.plot(ts_smooth, val_smooth, color=colors[seed], linewidth=1.5,
                label=f'Seed {seed}', alpha=0.8)

    if key in baselines_ref:
        for val, name, color in baselines_ref[key]:
            ax.axhline(y=val, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(195000, val, f'  {name}', fontsize=8, color=color,
                    va='bottom' if val > 0 else 'top')

    ax.set_xlabel('Timesteps', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.axvspan(50000, 200000, alpha=0.03, color='green')

# Convergence analysis
ax = axes[5]
for seed in range(5):
    data = all_data[seed]
    timesteps = [d['timestep'] for d in data]
    rewards = [d['reward'] for d in data]
    chunk_size = 500
    chunk_ts = []
    chunk_means = []
    for i in range(0, len(rewards) - chunk_size, chunk_size // 2):
        chunk = rewards[i:i + chunk_size]
        chunk_ts.append(timesteps[i + chunk_size // 2])
        chunk_means.append(np.mean(chunk))
    ax.plot(chunk_ts, chunk_means, color=colors[seed], linewidth=2,
            label=f'Seed {seed}', alpha=0.8)

ax.set_xlabel('Timesteps', fontsize=10)
ax.set_ylabel('Mean Reward (500-ep window)', fontsize=10)
ax.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.axvline(x=50000, color='red', linestyle=':', alpha=0.5)
ax.text(52000, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
        '~50K: convergence', fontsize=9, color='red')

plt.suptitle('Township 108 (B-Medium): Training Curves — 200K Steps x 5 Seeds\n'
             'Episode = 20 steps, ~10,000 episodes per seed',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'paper3_training_curves_108.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: paper3_training_curves_108.png")

print("\n=== Convergence Analysis ===")
for seed in range(5):
    data = all_data[seed]
    rewards = [d['reward'] for d in data]
    first_half = rewards[:len(rewards)//2]
    second_half = rewards[len(rewards)//2:]
    print(f"Seed {seed}: first 100K mean={np.mean(first_half):.1f}, "
          f"last 100K mean={np.mean(second_half):.1f}, "
          f"improvement={np.mean(second_half)-np.mean(first_half):+.1f}")
