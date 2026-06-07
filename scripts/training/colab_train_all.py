# -*- coding: utf-8 -*-
"""
Colab Training Script for Paper 3: Block-Level DRL (All 3 Townships × 5 Seeds).

Upload the paper3_colab.zip to Google Drive, then run this in Colab:

    !pip install geopandas libpysal sb3-contrib shimmy tqdm
    %cd /content
    !cp /content/drive/MyDrive/paper3_colab.zip .
    !unzip -qo paper3_colab.zip
    !python colab_train_all.py

Or run individual townships:
    !python colab_train_all.py --township 500227109

Estimated A100 times:
    500227109 (78 blocks):  ~5 min/seed × 5 = ~25 min
    500227108 (132 blocks): ~10 min/seed × 5 = ~50 min
    500227105 (338 blocks): ~25 min/seed × 5 = ~2 h
    Total: ~3 hours
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Ensure repository src/ is on path.
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

TOWNSHIPS = ['500227109', '500227108', '500227105']
TOWNSHIP_NAMES = {'500227109': 'A-Small', '500227108': 'B-Medium', '500227105': 'C-Large'}

RESULTS_DIR = Path(os.getenv("PAPER3_BLOCK_RESULTS_DIR", BLOCK_RESULTS_DIR))

# Tuned hyperparameters (from local experiments)
DEFAULT_CONFIG = dict(
    total_timesteps=200_000,
    total_budget=100,
    swaps_per_step=5,
    n_steps=512,
    batch_size=256,
    slope_weight=2000.0,
    cont_weight=500.0,
    baimu_weight=500.0,
    baimu_bonus=20.0,
    learning_rate=1e-3,
    ent_coef=0.01,
)


def train_single(township_code, seed, config):
    """Train a single model (one township × one seed)."""
    output_dir = RESULTS_DIR / f'township_{township_code}'
    os.makedirs(output_dir, exist_ok=True)

    model_path = output_dir / f'block_model_seed{seed}.zip'
    if model_path.exists():
        print(f"  [SKIP] Model already exists: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Training: {TOWNSHIP_NAMES[township_code]} (seed={seed})")
    print(f"  Config: ts={config['total_timesteps']:,}, lr={config['learning_rate']}, "
          f"ent={config['ent_coef']}")
    print(f"  Weights: slope={config['slope_weight']}, cont={config['cont_weight']}, "
          f"baimu={config['baimu_weight']}, bonus={config['baimu_bonus']}")
    print(f"{'='*60}")

    env = BlockLevelEnv(
        township_code,
        total_budget=config['total_budget'],
        swaps_per_step=config['swaps_per_step'],
        slope_weight=config['slope_weight'],
        cont_weight=config['cont_weight'],
        baimu_weight=config['baimu_weight'],
        baimu_bonus=config['baimu_bonus'],
    )
    env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy,
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=config['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / 'tb_block_logs'),
        policy_kwargs=dict(
            k_parcel=K_BLOCK,
            k_global=K_GLOBAL,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
        device='auto',
    )

    log_path = output_dir / f'block_training_log_seed{seed}.json'
    callback = BlockMetricsCallback(log_path=str(log_path))

    t0 = time.time()
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callback,
        progress_bar=True,
    )
    train_time = time.time() - t0

    model.save(str(output_dir / f'block_model_seed{seed}'))
    print(f"\n  Trained in {train_time:.0f}s ({train_time/60:.1f} min)")

    if callback.episode_data:
        last_50 = callback.episode_data[-50:]
        print(f"  Last 50 eps: reward={np.mean([d['reward'] for d in last_50]):.2f}, "
              f"slope={np.mean([d['slope_change_pct'] for d in last_50]):+.2f}%, "
              f"baimu_cnt={np.mean([d['baimu_count_change'] for d in last_50]):+.1f}")

    return train_time


def run_baselines(township_code, config):
    """Run all baselines for a township."""
    from baselines_block import (run_greedy_global, run_greedy_sequential,
                                  run_random_block, run_round_robin)

    print(f"\n{'='*60}")
    print(f"  Baselines: {TOWNSHIP_NAMES[township_code]}")
    print(f"{'='*60}")

    env = BlockLevelEnv(
        township_code,
        total_budget=config['total_budget'],
        swaps_per_step=config['swaps_per_step'],
    )

    results = {}

    print("  Greedy-Global...", end=" ", flush=True)
    r = run_greedy_global(env)
    results['greedy_global'] = r
    print(f"slope={r['slope_change_pct']:+.2f}%, baimu={r['baimu_count_change']:+d}")

    print("  Greedy-Sequential...", end=" ", flush=True)
    r = run_greedy_sequential(env)
    results['greedy_sequential'] = r
    print(f"slope={r['slope_change_pct']:+.2f}%, baimu={r['baimu_count_change']:+d}")

    print("  Random-Block (5 seeds)...", end=" ", flush=True)
    random_results = [run_random_block(env, seed=s) for s in range(5)]
    results['random_block'] = {
        'method': 'Random-Block',
        'slope_pct_mean': float(np.mean([r['slope_change_pct'] for r in random_results])),
        'cont_mean': float(np.mean([r['cont_change'] for r in random_results])),
        'baimu_count_mean': float(np.mean([r['baimu_count_change'] for r in random_results])),
        'baimu_area_mean': float(np.mean([r['baimu_area_change_ha'] for r in random_results])),
        'per_seed': random_results,
    }
    print(f"slope={results['random_block']['slope_pct_mean']:+.2f}%, "
          f"baimu={results['random_block']['baimu_count_mean']:+.1f}")

    print("  Round-Robin...", end=" ", flush=True)
    r = run_round_robin(env)
    results['round_robin'] = r
    print(f"slope={r['slope_change_pct']:+.2f}%, baimu={r['baimu_count_change']:+d}")

    out_dir = RESULTS_DIR / f'township_{township_code}'
    with open(out_dir / 'baselines_block.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_dir / 'baselines_block.json'}")

    return results


def summarize_results(townships, seeds, config):
    """Print summary table across all townships and seeds."""
    print(f"\n\n{'='*90}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*90}")

    for tc in townships:
        name = TOWNSHIP_NAMES[tc]
        out_dir = RESULTS_DIR / f'township_{tc}'

        print(f"\n  --- {name} ({tc}) ---")

        # Baselines
        bl_path = out_dir / 'baselines_block.json'
        if bl_path.exists():
            with open(bl_path) as f:
                baselines = json.load(f)
            for key in ['greedy_global', 'greedy_sequential', 'round_robin']:
                if key in baselines:
                    b = baselines[key]
                    print(f"    {b['method']:<22}: slope={b['slope_change_pct']:+.2f}%, "
                          f"cont={b['cont_change']:+.4f}, "
                          f"baimu_cnt={b.get('baimu_count_change', 0):+d}")
            if 'random_block' in baselines:
                rb = baselines['random_block']
                print(f"    {'Random-Block (avg)':<22}: slope={rb['slope_pct_mean']:+.2f}%, "
                      f"cont={rb['cont_mean']:+.4f}, "
                      f"baimu_cnt={rb['baimu_count_mean']:+.1f}")

        # DRL results
        drl_results = []
        for seed in seeds:
            eval_path = out_dir / f'block_eval_seed{seed}.json'
            if eval_path.exists():
                with open(eval_path) as f:
                    drl_results.append(json.load(f))
                r = drl_results[-1]
                print(f"    DRL seed={seed:<18}: slope={r['slope_change_pct']:+.2f}%, "
                      f"cont={r['cont_change']:+.4f}, "
                      f"baimu_cnt={r['baimu_count_change']:+d}")

        if len(drl_results) > 1:
            slopes = [r['slope_change_pct'] for r in drl_results]
            conts = [r['cont_change'] for r in drl_results]
            baimu = [r['baimu_count_change'] for r in drl_results]
            print(f"    {'DRL mean±std':<22}: slope={np.mean(slopes):+.2f}%±{np.std(slopes):.2f}%, "
                  f"cont={np.mean(conts):+.4f}±{np.std(conts):.4f}, "
                  f"baimu_cnt={np.mean(baimu):+.1f}")

    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(description='Train block-level DRL (all townships)')
    parser.add_argument('--township', type=str, default=None,
                        help='Single township code (default: all 3)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    townships = [args.township] if args.township else TOWNSHIPS
    seeds = list(range(args.seeds))

    config = DEFAULT_CONFIG.copy()
    config['total_timesteps'] = args.timesteps

    print(f"  Townships: {townships}")
    print(f"  Seeds: {seeds}")
    print(f"  Timesteps: {config['total_timesteps']:,}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    if args.summary_only:
        summarize_results(townships, seeds, config)
        return

    total_start = time.time()

    for tc in townships:
        # Baselines
        if not args.eval_only:
            if not args.skip_baselines:
                run_baselines(tc, config)

            if not args.baselines_only:
                # Training
                for seed in seeds:
                    train_single(tc, seed, config)

        # Evaluation (all seeds)
        for seed in seeds:
            model_path = RESULTS_DIR / f'township_{tc}' / f'block_model_seed{seed}.zip'
            if model_path.exists():
                evaluate_block(tc, seed=seed,
                              total_budget=config['total_budget'],
                              swaps_per_step=config['swaps_per_step'])

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")

    # Summary
    summarize_results(townships, seeds, config)

    # Save timing
    timing = {
        'total_seconds': total_time,
        'townships': townships,
        'seeds': seeds,
        'timesteps': config['total_timesteps'],
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'none',
    }
    with open(RESULTS_DIR / 'colab_timing.json', 'w') as f:
        json.dump(timing, f, indent=2)


if __name__ == '__main__':
    main()
