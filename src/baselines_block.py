# -*- coding: utf-8 -*-
"""
Block-Level Baselines for Paper 3 (Direction 2: 百亩方 Formation).

Compares different block selection strategies using the same
greedy intra-block execution engine:
  1. Greedy-Global: ignore blocks, sort ALL parcels globally (Paper 2 baseline)
  2. Greedy-Sequential: process blocks by slope gap (descending)
  3. Random-Block: randomly select blocks
  4. Round-Robin: cycle through all blocks

All baselines now report 百亩方 metrics alongside slope and contiguity.

Usage:
    python baselines_block.py --township 500227109
    python baselines_block.py --township 500227109 --budget 100 --swaps-per-step 5
"""

import os
import sys
import json
import time
import argparse
import numpy as np

from block_level_env import BlockLevelEnv
from paper3_paths import BLOCK_RESULTS_DIR

RESULTS_DIR = str(BLOCK_RESULTS_DIR)


def run_greedy_global(env):
    """Greedy-Global: sort ALL parcels globally, ignore block structure.

    This is Paper 2's greedy baseline applied to the same parcels.
    Directly swaps highest-slope farmland with lowest-slope forest.
    """
    env.reset()
    initial_slope = env.avg_farmland_slope
    initial_cont = env.contiguity
    initial_baimu_count, initial_baimu_area = env._count_baimu_fang()

    t0 = time.time()
    completed = 0

    for _ in range(env.total_budget):
        # Find all available farmland and forest (not yet swapped, in any block)
        farm_avail = np.where((env.land_use == 1) & ~env.swapped &
                              (env.parcel_to_block >= 0))[0]
        forest_avail = np.where((env.land_use == 2) & ~env.swapped &
                                (env.parcel_to_block >= 0))[0]

        if len(farm_avail) == 0 or len(forest_avail) == 0:
            break

        # Connectivity-aware scoring (same as intra-block engine)
        farm_scores = (env.slopes[farm_avail]
                       - env.delta_conn * env.farmland_nbr_count[farm_avail])
        best_farm = farm_avail[np.argmax(farm_scores)]

        forest_scores = (env.slopes[forest_avail]
                         - env.gamma_conn * env.farmland_nbr_count[forest_avail])
        best_forest = forest_avail[np.argmin(forest_scores)]

        if env.slopes[best_farm] <= env.slopes[best_forest]:
            break

        env._swap_to_forest(best_farm)
        env._swap_to_farmland(best_forest)
        env.swapped[best_farm] = True
        env.swapped[best_forest] = True
        completed += 1

    elapsed = time.time() - t0
    final_slope = env.avg_farmland_slope
    final_cont = env.contiguity
    final_baimu_count, final_baimu_area = env._count_baimu_fang()

    return {
        'method': 'Greedy-Global',
        'slope_change_pct': 100 * (final_slope - initial_slope) / (abs(initial_slope) + 1e-8),
        'cont_change': final_cont - initial_cont,
        'baimu_count_change': final_baimu_count - initial_baimu_count,
        'baimu_area_change_ha': (final_baimu_area - initial_baimu_area) / 10000.0,
        'completed_swaps': completed,
        'elapsed': elapsed,
        'initial_slope': initial_slope,
        'final_slope': final_slope,
        'initial_cont': initial_cont,
        'final_cont': final_cont,
        'initial_baimu_count': initial_baimu_count,
        'final_baimu_count': final_baimu_count,
    }


def run_greedy_sequential(env):
    """Greedy-Sequential: process blocks ordered by slope gap (descending).

    Allocates swaps_per_step to each block in order of decreasing
    farmland-forest slope gap until budget is exhausted.
    """
    env.reset()
    initial_slope = env.avg_farmland_slope
    initial_cont = env.contiguity
    initial_baimu_count, initial_baimu_area = env._count_baimu_fang()

    t0 = time.time()
    total_completed = 0
    steps = 0

    for _ in range(env.max_steps):
        # Compute slope gap for each valid block
        mask = env.action_masks()
        if not mask.any():
            break

        gaps = np.full(env.n_blocks, -np.inf)
        for b in np.where(mask)[0]:
            parcels = env.block_parcels[b]
            types = env.land_use[parcels]
            avail = ~env.swapped[parcels]

            fm = (types == 1) & avail
            ff = (types == 2) & avail
            if fm.any() and ff.any():
                farm_slope = np.average(env.slopes[parcels[fm]],
                                        weights=env.areas[parcels[fm]])
                forest_slope = np.average(env.slopes[parcels[ff]],
                                          weights=env.areas[parcels[ff]])
                gaps[b] = farm_slope - forest_slope

        best_block = int(np.argmax(gaps))
        completed = env._execute_greedy_in_block(best_block, env.swaps_per_step)
        env.budget_used += completed
        env.swaps_in_block[best_block] += completed
        total_completed += completed
        steps += 1

    elapsed = time.time() - t0
    final_slope = env.avg_farmland_slope
    final_cont = env.contiguity
    final_baimu_count, final_baimu_area = env._count_baimu_fang()

    return {
        'method': 'Greedy-Sequential',
        'slope_change_pct': 100 * (final_slope - initial_slope) / (abs(initial_slope) + 1e-8),
        'cont_change': final_cont - initial_cont,
        'baimu_count_change': final_baimu_count - initial_baimu_count,
        'baimu_area_change_ha': (final_baimu_area - initial_baimu_area) / 10000.0,
        'completed_swaps': total_completed,
        'steps': steps,
        'elapsed': elapsed,
        'initial_slope': initial_slope,
        'final_slope': final_slope,
        'initial_cont': initial_cont,
        'final_cont': final_cont,
    }


def run_random_block(env, seed=42):
    """Random-Block: randomly select blocks, greedy execution within."""
    rng = np.random.default_rng(seed)
    env.reset()
    initial_slope = env.avg_farmland_slope
    initial_cont = env.contiguity
    initial_baimu_count, initial_baimu_area = env._count_baimu_fang()

    t0 = time.time()
    total_completed = 0

    for step in range(env.max_steps):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break

        block_id = int(rng.choice(valid))
        completed = env._execute_greedy_in_block(block_id, env.swaps_per_step)
        env.budget_used += completed
        env.swaps_in_block[block_id] += completed
        total_completed += completed

    elapsed = time.time() - t0
    final_slope = env.avg_farmland_slope
    final_cont = env.contiguity
    final_baimu_count, final_baimu_area = env._count_baimu_fang()

    return {
        'method': 'Random-Block',
        'seed': seed,
        'slope_change_pct': 100 * (final_slope - initial_slope) / (abs(initial_slope) + 1e-8),
        'cont_change': final_cont - initial_cont,
        'baimu_count_change': final_baimu_count - initial_baimu_count,
        'baimu_area_change_ha': (final_baimu_area - initial_baimu_area) / 10000.0,
        'completed_swaps': total_completed,
        'elapsed': elapsed,
        'initial_slope': initial_slope,
        'final_slope': final_slope,
        'initial_cont': initial_cont,
        'final_cont': final_cont,
    }


def run_round_robin(env):
    """Round-Robin: cycle through blocks in fixed order."""
    env.reset()
    initial_slope = env.avg_farmland_slope
    initial_cont = env.contiguity
    initial_baimu_count, initial_baimu_area = env._count_baimu_fang()

    t0 = time.time()
    total_completed = 0
    block_order = list(range(env.n_blocks))
    step = 0

    for _ in range(env.max_steps):
        mask = env.action_masks()
        if not mask.any():
            break

        # Find next valid block in round-robin order
        found = False
        for _ in range(env.n_blocks):
            b = block_order[step % len(block_order)]
            step += 1
            if mask[b]:
                found = True
                break
        if not found:
            break

        completed = env._execute_greedy_in_block(b, env.swaps_per_step)
        env.budget_used += completed
        env.swaps_in_block[b] += completed
        total_completed += completed

    elapsed = time.time() - t0
    final_slope = env.avg_farmland_slope
    final_cont = env.contiguity
    final_baimu_count, final_baimu_area = env._count_baimu_fang()

    return {
        'method': 'Round-Robin',
        'slope_change_pct': 100 * (final_slope - initial_slope) / (abs(initial_slope) + 1e-8),
        'cont_change': final_cont - initial_cont,
        'baimu_count_change': final_baimu_count - initial_baimu_count,
        'baimu_area_change_ha': (final_baimu_area - initial_baimu_area) / 10000.0,
        'completed_swaps': total_completed,
        'elapsed': elapsed,
        'initial_slope': initial_slope,
        'final_slope': final_slope,
        'initial_cont': initial_cont,
        'final_cont': final_cont,
    }


def main():
    parser = argparse.ArgumentParser(description='Block-level baselines')
    parser.add_argument('--township', type=str, default='500227109')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--swaps-per-step', type=int, default=5)
    parser.add_argument('--random-seeds', type=int, default=5)
    args = parser.parse_args()

    env = BlockLevelEnv(args.township, total_budget=args.budget,
                        swaps_per_step=args.swaps_per_step)

    print(f"\n{'='*70}")
    print(f"  Block-Level Baselines: Township {args.township}")
    print(f"  Budget={args.budget}, SwapsPerStep={args.swaps_per_step}")
    print(f"{'='*70}")

    all_results = {}

    # 1. Greedy-Global
    print("\n  [1/4] Greedy-Global...")
    r = run_greedy_global(env)
    all_results['greedy_global'] = r
    print(f"    Slope: {r['slope_change_pct']:+.2f}%, Cont: {r['cont_change']:+.4f}, "
          f"Swaps: {r['completed_swaps']}, 百亩方: {r['baimu_count_change']:+d}, Time: {r['elapsed']:.3f}s")

    # 2. Greedy-Sequential
    print("\n  [2/4] Greedy-Sequential...")
    r = run_greedy_sequential(env)
    all_results['greedy_sequential'] = r
    print(f"    Slope: {r['slope_change_pct']:+.2f}%, Cont: {r['cont_change']:+.4f}, "
          f"Swaps: {r['completed_swaps']}, 百亩方: {r['baimu_count_change']:+d}, Time: {r['elapsed']:.3f}s")

    # 3. Random-Block (multi-seed)
    print(f"\n  [3/4] Random-Block ({args.random_seeds} seeds)...")
    random_results = []
    for seed in range(args.random_seeds):
        r = run_random_block(env, seed=seed)
        random_results.append(r)
        print(f"    Seed {seed}: Slope {r['slope_change_pct']:+.2f}%, "
              f"Cont {r['cont_change']:+.4f}, Swaps {r['completed_swaps']}, "
              f"百亩方 {r['baimu_count_change']:+d}")
    slope_pcts = [r['slope_change_pct'] for r in random_results]
    cont_changes = [r['cont_change'] for r in random_results]
    baimu_counts = [r['baimu_count_change'] for r in random_results]
    baimu_areas = [r['baimu_area_change_ha'] for r in random_results]
    all_results['random_block'] = {
        'method': 'Random-Block',
        'slope_pct_mean': float(np.mean(slope_pcts)),
        'slope_pct_std': float(np.std(slope_pcts)),
        'cont_mean': float(np.mean(cont_changes)),
        'cont_std': float(np.std(cont_changes)),
        'baimu_count_mean': float(np.mean(baimu_counts)),
        'baimu_area_mean': float(np.mean(baimu_areas)),
        'per_seed': random_results,
    }
    print(f"    Mean: Slope {np.mean(slope_pcts):+.2f}% +/- {np.std(slope_pcts):.2f}%, "
          f"Cont {np.mean(cont_changes):+.4f}, "
          f"百亩方 cnt {np.mean(baimu_counts):+.1f}, area {np.mean(baimu_areas):+.2f} ha")

    # 4. Round-Robin
    print("\n  [4/4] Round-Robin...")
    r = run_round_robin(env)
    all_results['round_robin'] = r
    print(f"    Slope: {r['slope_change_pct']:+.2f}%, Cont: {r['cont_change']:+.4f}, "
          f"Swaps: {r['completed_swaps']}, 百亩方: {r['baimu_count_change']:+d}, Time: {r['elapsed']:.3f}s")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE — Township {args.township}")
    print(f"{'='*80}")
    print(f"  {'Method':<22} {'Slope %':>10} {'Cont':>10} {'百亩方 cnt':>10} {'百亩方 ha':>10} {'Swaps':>8} {'Time':>8}")
    print(f"  {'-'*72}")

    for key, data in all_results.items():
        if key == 'random_block':
            sp = data['slope_pct_mean']
            cc = data['cont_mean']
            bc = data['baimu_count_mean']
            ba = data['baimu_area_mean']
            sw = int(np.mean([r['completed_swaps'] for r in data['per_seed']]))
            t = np.mean([r['elapsed'] for r in data['per_seed']])
            label = f"Random-Block (avg)"
        else:
            sp = data['slope_change_pct']
            cc = data['cont_change']
            bc = data.get('baimu_count_change', 0)
            ba = data.get('baimu_area_change_ha', 0.0)
            sw = data['completed_swaps']
            t = data['elapsed']
            label = data['method']
        print(f"  {label:<22} {sp:>+9.2f}% {cc:>+10.4f} {bc:>+10.0f} {ba:>+9.2f} {sw:>8d} {t:>7.3f}s")

    print(f"{'='*80}")

    # Save results
    out_dir = os.path.join(RESULTS_DIR, f'township_{args.township}')
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'baselines_block.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")


if __name__ == '__main__':
    main()
