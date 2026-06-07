# -*- coding: utf-8 -*-
"""
Colab Retraining Script: Township 108 with balanced reward weights.

Problem: Original config (slope_weight=2000, baimu_weight=500) caused
DRL to learn "pure slope" strategy on 108, resulting in baimu_area=-97.5ha.

Fix: Rebalance weights to slope:baimu = 1:1 (both 1000).

Usage on Colab:
    !python colab_retrain_108.py

Results saved to: results_real/blocks/township_500227108_v2/
Estimated time: ~50 min on A100
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributions
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from block_level_env import BlockLevelEnv, K_BLOCK, K_GLOBAL
from parcel_scoring_policy import ParcelScoringPolicy
from train_block import BlockMetricsCallback, evaluate_block
from paper3_paths import BLOCK_RESULTS_DIR

TOWNSHIP = '500227108'
OUTPUT_DIR = Path(os.getenv("PAPER3_BLOCK_RESULTS_DIR", BLOCK_RESULTS_DIR)) / 'township_500227108_v2'

# Rebalanced config: slope:baimu = 1:1 (was 4:1)
CONFIG = dict(
    total_timesteps=200_000,
    total_budget=100,
    swaps_per_step=5,
    n_steps=512,
    batch_size=256,
    slope_weight=1000.0,    # was 2000
    cont_weight=500.0,      # unchanged
    baimu_weight=1000.0,    # was 500
    baimu_bonus=50.0,       # was 20
    learning_rate=1e-3,
    ent_coef=0.01,
)


def train_single(seed):
    """Train one seed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = OUTPUT_DIR / f'block_model_seed{seed}.zip'
    if model_path.exists():
        print(f"  [SKIP] {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Training 108 v2: seed={seed}")
    print(f"  Weights: slope={CONFIG['slope_weight']}, cont={CONFIG['cont_weight']}, "
          f"baimu={CONFIG['baimu_weight']}, bonus={CONFIG['baimu_bonus']}")
    print(f"{'='*60}")

    env = BlockLevelEnv(
        TOWNSHIP,
        total_budget=CONFIG['total_budget'],
        swaps_per_step=CONFIG['swaps_per_step'],
        slope_weight=CONFIG['slope_weight'],
        cont_weight=CONFIG['cont_weight'],
        baimu_weight=CONFIG['baimu_weight'],
        baimu_bonus=CONFIG['baimu_bonus'],
    )
    env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy,
        env,
        learning_rate=CONFIG['learning_rate'],
        n_steps=CONFIG['n_steps'],
        batch_size=CONFIG['batch_size'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=CONFIG['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=str(OUTPUT_DIR / 'tb_block_logs'),
        policy_kwargs=dict(
            k_parcel=K_BLOCK,
            k_global=K_GLOBAL,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
        device='auto',
    )

    log_path = OUTPUT_DIR / f'block_training_log_seed{seed}.json'
    callback = BlockMetricsCallback(log_path=str(log_path))

    t0 = time.time()
    model.learn(total_timesteps=CONFIG['total_timesteps'], callback=callback,
                progress_bar=True)
    train_time = time.time() - t0

    model.save(str(OUTPUT_DIR / f'block_model_seed{seed}'))
    print(f"\n  Trained in {train_time:.0f}s ({train_time/60:.1f} min)")

    if callback.episode_data:
        last_50 = callback.episode_data[-50:]
        print(f"  Last 50 eps: reward={np.mean([d['reward'] for d in last_50]):.2f}, "
              f"slope={np.mean([d['slope_change_pct'] for d in last_50]):+.2f}%, "
              f"baimu_ha={np.mean([d['baimu_area_change_ha'] for d in last_50]):+.1f}")

    return train_time


def evaluate_single(seed):
    """Evaluate one seed with deterministic policy."""
    model_path = OUTPUT_DIR / f'block_model_seed{seed}.zip'
    if not model_path.exists():
        print(f"  [SKIP] No model: {model_path}")
        return None

    eval_path = OUTPUT_DIR / f'block_eval_seed{seed}.json'
    if eval_path.exists():
        print(f"  [SKIP] Already evaluated: {eval_path}")
        with open(eval_path) as f:
            return json.load(f)

    print(f"\n  Evaluating seed {seed}...")

    # Create a fresh env (no reward weights needed for eval)
    env = BlockLevelEnv(
        TOWNSHIP,
        total_budget=CONFIG['total_budget'],
        swaps_per_step=CONFIG['swaps_per_step'],
    )

    model = MaskablePPO.load(str(model_path), env=env)

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    block_history = []

    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        block_history.append({
            'step': info.get('step', len(block_history) + 1),
            'block': int(action),
            'swaps': info.get('swaps_done', 0),
            'slope_pct': info.get('slope_change_pct', 0),
            'cont': info.get('cont_change', 0),
        })

    result = {
        'method': 'DRL-Block-v2',
        'seed': seed,
        'slope_change_pct': info.get('slope_change_pct', 0),
        'cont_change': info.get('cont_change', 0),
        'baimu_count_change': info.get('baimu_count_change', 0),
        'baimu_area_change_ha': info.get('baimu_area_change_ha', 0),
        'budget_used': info.get('budget_used', 0),
        'total_reward': total_reward,
        'block_history': block_history,
    }

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {eval_path}")
    print(f"  slope={result['slope_change_pct']:+.2f}%, "
          f"cont={result['cont_change']:+.4f}, "
          f"baimu_cnt={result['baimu_count_change']:+d}, "
          f"baimu_ha={result['baimu_area_change_ha']:+.1f}")

    return result


def main():
    seeds = list(range(5))

    print(f"  Township: 108 (B-Medium)")
    print(f"  Version: v2 (rebalanced weights)")
    print(f"  Seeds: {seeds}")
    print(f"  Config: slope={CONFIG['slope_weight']}, baimu={CONFIG['baimu_weight']}, "
          f"bonus={CONFIG['baimu_bonus']}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    total_start = time.time()

    # Train all seeds
    for seed in seeds:
        train_single(seed)

    # Evaluate all seeds
    results = []
    for seed in seeds:
        r = evaluate_single(seed)
        if r:
            results.append(r)

    # Summary
    if results:
        slopes = [r['slope_change_pct'] for r in results]
        conts = [r['cont_change'] for r in results]
        baimu_cnt = [r['baimu_count_change'] for r in results]
        baimu_ha = [r['baimu_area_change_ha'] for r in results]

        print(f"\n{'='*60}")
        print(f"  108 v2 RESULTS (rebalanced weights)")
        print(f"{'='*60}")
        for r in results:
            print(f"  Seed {r['seed']}: slope={r['slope_change_pct']:+.2f}%, "
                  f"cont={r['cont_change']:+.4f}, "
                  f"baimu_cnt={r['baimu_count_change']:+d}, "
                  f"baimu_ha={r['baimu_area_change_ha']:+.1f}")
        print(f"  Mean: slope={np.mean(slopes):+.2f}%+-{np.std(slopes):.2f}%, "
              f"cont={np.mean(conts):+.4f}, "
              f"baimu_cnt={np.mean(baimu_cnt):+.1f}, "
              f"baimu_ha={np.mean(baimu_ha):+.1f}+-{np.std(baimu_ha):.1f}")
        print(f"\n  vs Original v1:")
        print(f"    v1: slope=-3.59%, baimu_ha=-97.5")
        print(f"    v2: slope={np.mean(slopes):+.2f}%, baimu_ha={np.mean(baimu_ha):+.1f}")
        print(f"{'='*60}")

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
