# -*- coding: utf-8 -*-
"""
Training script for Block-Level MDP (Paper 3).

Uses MaskablePPO with ParcelScoringPolicy (dimension-invariant architecture)
to train a block selection agent. The policy scores each block independently
using shared MLP weights, enabling cross-township generalization.

Usage:
    python train_block.py --township 500227109 --seed 0
    python train_block.py --township 500227109 --seed 0 --timesteps 500000
    python train_block.py --township 500227105 --budget 100 --swaps-per-step 5
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.distributions
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from block_level_env import BlockLevelEnv, K_BLOCK, K_GLOBAL
from parcel_scoring_policy import ParcelScoringPolicy
from paper3_paths import BLOCK_RESULTS_DIR

RESULTS_DIR = Path(os.getenv("PAPER3_BLOCK_RESULTS_DIR", BLOCK_RESULTS_DIR))


class BlockMetricsCallback(BaseCallback):
    """Track block-level metrics during training."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_data = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_data.append({
                    'reward': float(info['episode']['r']),
                    'length': int(info['episode']['l']),
                    'avg_slope': float(info.get('avg_slope', 0)),
                    'contiguity': float(info.get('contiguity', 0)),
                    'budget_used': int(info.get('budget_used', 0)),
                    'slope_change_pct': float(info.get('slope_change_pct', 0)),
                    'cont_change': float(info.get('cont_change', 0)),
                    'baimu_count_change': int(info.get('baimu_count_change', 0)),
                    'baimu_area_change_ha': float(info.get('baimu_area_change_ha', 0)),
                    'timestep': self.num_timesteps,
                })
        if self.num_timesteps % 5000 == 0 and len(self.episode_data) > 0:
            recent = self.episode_data[-50:]
            self.logger.record('custom/slope_pct',
                               np.mean([d['slope_change_pct'] for d in recent]))
            self.logger.record('custom/cont_change',
                               np.mean([d['cont_change'] for d in recent]))
            self.logger.record('custom/budget_used',
                               np.mean([d['budget_used'] for d in recent]))
            self.logger.record('custom/ep_reward',
                               np.mean([d['reward'] for d in recent]))
            self.logger.record('custom/baimu_count_change',
                               np.mean([d['baimu_count_change'] for d in recent]))
            self.logger.record('custom/baimu_area_ha',
                               np.mean([d['baimu_area_change_ha'] for d in recent]))
        return True

    def _on_training_end(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.episode_data, f, indent=2)
        print(f"\nTraining log saved: {self.log_path} ({len(self.episode_data)} episodes)")


def train_block(township_code, seed=0, total_timesteps=100_000,
                total_budget=100, swaps_per_step=5,
                n_steps=512, batch_size=256,
                slope_weight=2000.0, cont_weight=500.0,
                baimu_weight=500.0, baimu_bonus=20.0):
    """Train MaskablePPO block-level agent."""

    output_dir = RESULTS_DIR / f'township_{township_code}'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Block-Level DRL Training: Township {township_code}")
    print(f"  Seed: {seed}, Timesteps: {total_timesteps:,}")
    print(f"  Budget: {total_budget}, SwapsPerStep: {swaps_per_step}")
    print("=" * 60)

    # Create environment
    env = BlockLevelEnv(
        township_code,
        total_budget=total_budget,
        swaps_per_step=swaps_per_step,
        slope_weight=slope_weight,
        cont_weight=cont_weight,
        baimu_weight=baimu_weight,
        baimu_bonus=baimu_bonus,
    )
    n_blocks = env.n_blocks
    max_steps = env.max_steps

    # Memory estimate
    obs_dim = n_blocks * K_BLOCK + K_GLOBAL
    rollout_mb = n_steps * obs_dim * 4 / (1024 ** 2)
    mask_mb = n_steps * n_blocks / (1024 ** 2)
    print(f"\n  Memory estimate: obs={rollout_mb:.1f} MB, masks={mask_mb:.1f} MB")
    print(f"  Episode length: {max_steps} steps")
    print(f"  Scorer input dim: {K_BLOCK + K_GLOBAL}")

    env = Monitor(env)

    # Create model with ParcelScoringPolicy (works for blocks too)
    model = MaskablePPO(
        ParcelScoringPolicy,
        env,
        learning_rate=1e-3,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # low entropy: 78 actions, need sharper policy
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

    total_params = sum(p.numel() for p in model.policy.parameters())
    scorer_params = sum(p.numel() for p in model.policy.scorer_net.parameters())
    value_params = sum(p.numel() for p in model.policy.value_net.parameters())
    print(f"\n  Policy: ParcelScoringPolicy (block-level)")
    print(f"    Scorer: ({K_BLOCK+K_GLOBAL}) -> [128,64] -> (1)  [{scorer_params:,} params]")
    print(f"    Value:  ({K_GLOBAL}) -> [64,32] -> (1)  [{value_params:,} params]")
    print(f"    Total: {total_params:,} params")

    # Train
    log_path = output_dir / f'block_training_log_seed{seed}.json'
    callback = BlockMetricsCallback(log_path=str(log_path))

    print(f"\n  Starting training...\n")
    t_start = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    train_time = time.time() - t_start

    # Save model
    model_path = output_dir / f'block_model_seed{seed}'
    model.save(str(model_path))
    print(f"\n  Model saved: {model_path}.zip")

    # Print summary
    if callback.episode_data:
        last_50 = callback.episode_data[-50:]
        print(f"\n{'='*60}")
        print(f"  Training Summary (last 50 episodes)")
        print(f"{'='*60}")
        print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")
        print(f"  Avg reward:    {np.mean([d['reward'] for d in last_50]):.2f}")
        print(f"  Avg slope %:   {np.mean([d['slope_change_pct'] for d in last_50]):+.2f}%")
        print(f"  Avg cont:      {np.mean([d['cont_change'] for d in last_50]):+.4f}")
        print(f"  Avg baimu cnt: {np.mean([d['baimu_count_change'] for d in last_50]):+.1f}")
        print(f"  Avg baimu ha:  {np.mean([d['baimu_area_change_ha'] for d in last_50]):+.2f}")
        print(f"  Avg budget:    {np.mean([d['budget_used'] for d in last_50]):.0f}")
        print(f"  Total episodes:{len(callback.episode_data)}")
        print(f"{'='*60}")

    return model, callback


def evaluate_block(township_code, seed=0, total_budget=100, swaps_per_step=5):
    """Evaluate trained block-level agent."""
    output_dir = RESULTS_DIR / f'township_{township_code}'
    model_path = output_dir / f'block_model_seed{seed}.zip'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    env = BlockLevelEnv(township_code, total_budget=total_budget,
                        swaps_per_step=swaps_per_step)

    model = MaskablePPO.load(str(model_path), env=env)

    obs, info = env.reset()
    total_reward = 0
    block_history = []

    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        block_history.append({
            'step': info['step'],
            'block': info['block_selected'],
            'swaps': info['completed_swaps'],
            'slope_pct': info['slope_change_pct'],
            'cont': info['cont_change'],
        })
        if done:
            break

    results = {
        'method': 'DRL-Block',
        'seed': seed,
        'slope_change_pct': info['slope_change_pct'],
        'cont_change': info['cont_change'],
        'baimu_count_change': info['baimu_count_change'],
        'baimu_area_change_ha': info['baimu_area_change_ha'],
        'budget_used': info['budget_used'],
        'total_reward': float(total_reward),
        'block_history': block_history,
    }

    print(f"\n  DRL-Block Evaluation (seed {seed}):")
    print(f"    Slope: {results['slope_change_pct']:+.2f}%")
    print(f"    Cont:  {results['cont_change']:+.4f}")
    print(f"    百亩方 count: {results['baimu_count_change']:+d}")
    print(f"    百亩方 area:  {results['baimu_area_change_ha']:+.2f} ha")
    print(f"    Budget used: {results['budget_used']}")
    print(f"    Reward: {results['total_reward']:.2f}")

    eval_path = output_dir / f'block_eval_seed{seed}.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved: {eval_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train block-level DRL agent')
    parser.add_argument('--township', type=str, default='500227109')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--swaps-per-step', type=int, default=5)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--slope-weight', type=float, default=2000.0)
    parser.add_argument('--baimu-weight', type=float, default=500.0)
    args = parser.parse_args()

    if args.eval_only:
        evaluate_block(args.township, seed=args.seed,
                       total_budget=args.budget,
                       swaps_per_step=args.swaps_per_step)
    else:
        model, callback = train_block(
            township_code=args.township,
            seed=args.seed,
            total_timesteps=args.timesteps,
            total_budget=args.budget,
            swaps_per_step=args.swaps_per_step,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            slope_weight=args.slope_weight,
            baimu_weight=args.baimu_weight,
        )
        # Auto-evaluate after training
        evaluate_block(args.township, seed=args.seed,
                       total_budget=args.budget,
                       swaps_per_step=args.swaps_per_step)


if __name__ == '__main__':
    main()
