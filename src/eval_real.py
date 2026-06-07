"""Evaluate trained DRL model on real cadastral data with paired inference.

Loads a trained model and runs alternating farmland/forest swaps (paired
inference) to guarantee farmland count conservation (FC=0). Reports slope
reduction, contiguity improvement, and comparison with initial state.

Usage:
    python eval_real.py --township 500227109
    python eval_real.py --township 500227109 --seed 0 --n-pairs 100
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO

from land_use_env_real import RealDataLandUseEnv, K_PARCEL, K_GLOBAL
from paper3_paths import (
    ADJACENCY_DIR as DEFAULT_ADJACENCY_DIR,
    PARCEL_FEATURES_DIR as DEFAULT_FEATURES_DIR,
    RESULTS_DIR as DEFAULT_RESULTS_DIR,
)

# Repository-relative defaults; override with PAPER3_* environment variables.
FEATURES_DIR = Path(os.getenv("PAPER3_PARCEL_FEATURES_DIR", DEFAULT_FEATURES_DIR))
ADJACENCY_DIR = Path(os.getenv("PAPER3_ADJACENCY_DIR", DEFAULT_ADJACENCY_DIR))
RESULTS_DIR = Path(os.getenv("PAPER3_RESULTS_DIR", DEFAULT_RESULTS_DIR))


def evaluate_paired(township_code, seed=0, n_pairs=100, model_path=None):
    """Run paired inference evaluation on real data.

    Paired inference: alternate farmland->forest (even steps) and
    forest->farmland (odd steps) to guarantee FC=0 after each pair.
    """
    features_csv = FEATURES_DIR / f'township_{township_code}_features.csv'
    adjacency_npz = ADJACENCY_DIR / f'township_{township_code}_adj.npz'
    if model_path is None:
        model_path = RESULTS_DIR / f'township_{township_code}' / f'model_seed{seed}.zip'
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return None

    print("=" * 60)
    print(f"  Paired Inference Evaluation: Township {township_code}")
    print("=" * 60)

    # Create environment with enforce_pairs=True
    env = RealDataLandUseEnv(
        features_csv=str(features_csv),
        adjacency_npz=str(adjacency_npz),
        max_conversions=n_pairs * 2,  # 2 steps per pair
        enforce_pairs=True,
    )

    # Load model
    print(f"\nLoading model: {model_path}")
    model = MaskablePPO.load(str(model_path), env=env)

    # Run single episode with paired inference
    print(f"\nRunning paired inference ({n_pairs} pairs = {n_pairs*2} steps)...")
    t0 = time.time()

    obs, info = env.reset()
    initial_slope = info['avg_slope']
    initial_cont = info['contiguity']

    n_farm_initial = env.n_farmland
    n_forest_initial = env.n_forest

    step_log = []
    done = False
    step = 0

    while not done and step < n_pairs * 2:
        action_masks = env.action_masks()
        if not action_masks.any():
            print(f"  No valid actions at step {step}. Stopping.")
            break

        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        step_log.append({
            'step': step,
            'avg_slope': float(info['avg_slope']),
            'contiguity': float(info['contiguity']),
            'farmland_change': int(info['farmland_change']),
            'reward': float(reward),
        })

    dt = time.time() - t0
    final_slope = info['avg_slope']
    final_cont = info['contiguity']
    farmland_change = info['farmland_change']
    completed_pairs = info['completed_pairs']

    # Compute metrics
    slope_change = final_slope - initial_slope
    slope_change_pct = 100 * slope_change / initial_slope if initial_slope != 0 else 0
    cont_change = final_cont - initial_cont

    results = {
        'township_code': township_code,
        'seed': seed,
        'n_pairs_requested': n_pairs,
        'completed_pairs': completed_pairs,
        'total_steps': step,
        'initial_avg_slope': float(initial_slope),
        'final_avg_slope': float(final_slope),
        'slope_change': float(slope_change),
        'slope_change_pct': float(slope_change_pct),
        'initial_contiguity': float(initial_cont),
        'final_contiguity': float(final_cont),
        'cont_change': float(cont_change),
        'farmland_change': int(farmland_change),
        'n_farmland': int(env.n_farmland),
        'n_forest': int(env.n_forest),
        'inference_time_s': float(dt),
        'step_log': step_log,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Completed pairs: {completed_pairs}")
    print(f"  Slope: {initial_slope:.4f} -> {final_slope:.4f} "
          f"(change: {slope_change:.4f}, {slope_change_pct:+.2f}%)")
    print(f"  Contiguity: {initial_cont:.4f} -> {final_cont:.4f} "
          f"(change: {cont_change:+.4f})")
    print(f"  Farmland count change: {farmland_change} (FC={farmland_change})")
    print(f"  Inference time: {dt:.2f}s ({dt/max(completed_pairs,1)*1000:.1f} ms/pair)")
    print(f"{'='*60}")

    # Save results
    output_dir = RESULTS_DIR / f'township_{township_code}'
    os.makedirs(output_dir, exist_ok=True)
    results_path = output_dir / f'eval_paired_seed{seed}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate DRL on real data')
    parser.add_argument('--township', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-pairs', type=int, default=100)
    args = parser.parse_args()

    evaluate_paired(
        township_code=args.township,
        seed=args.seed,
        n_pairs=args.n_pairs,
    )


if __name__ == '__main__':
    main()
