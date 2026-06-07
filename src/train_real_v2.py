"""Training script for MaskablePPO with area-weighted constraint.

Same as train_real.py but uses RealDataLandUseEnvV2 which penalizes
farmland AREA deviation instead of COUNT deviation. This ensures the
model learns to preserve total farmland area, not just count.

Usage:
    python train_real_v2.py --township 500227109
    python train_real_v2.py --township 500227109 --area-penalty 1000
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

from land_use_env_real_v2 import RealDataLandUseEnvV2, K_PARCEL, K_GLOBAL
from parcel_scoring_policy import ParcelScoringPolicy
from paper3_paths import (
    ADJACENCY_DIR as DEFAULT_ADJACENCY_DIR,
    PARCEL_FEATURES_DIR as DEFAULT_FEATURES_DIR,
    RESULTS_DIR as DEFAULT_RESULTS_DIR,
)

FEATURES_DIR = Path(os.getenv("PAPER3_PARCEL_FEATURES_DIR", DEFAULT_FEATURES_DIR))
ADJACENCY_DIR = Path(os.getenv("PAPER3_ADJACENCY_DIR", DEFAULT_ADJACENCY_DIR))
RESULTS_DIR = Path(os.getenv("PAPER3_RESULTS_DIR", DEFAULT_RESULTS_DIR))


class MetricsCallbackV2(BaseCallback):
    """Track metrics including area deviation during training."""

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
                    'completed_pairs': int(info.get('completed_pairs', 0)),
                    'farmland_change': int(info.get('farmland_change', 0)),
                    'area_deviation': float(info.get('area_deviation', 0)),
                    'area_change_m2': float(info.get('area_change_m2', 0)),
                    'timestep': self.num_timesteps,
                })
        if self.num_timesteps % 10000 == 0 and len(self.episode_data) > 0:
            recent = self.episode_data[-20:]
            self.logger.record('custom/avg_slope',
                               np.mean([d['avg_slope'] for d in recent]))
            self.logger.record('custom/contiguity',
                               np.mean([d['contiguity'] for d in recent]))
            self.logger.record('custom/area_deviation',
                               np.mean([d['area_deviation'] for d in recent]))
            self.logger.record('custom/area_change_m2',
                               np.mean([d['area_change_m2'] for d in recent]))
            self.logger.record('custom/avg_pairs',
                               np.mean([d['completed_pairs'] for d in recent]))
        return True

    def _on_training_end(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.episode_data, f, indent=2)
        print(f"\nTraining log saved: {self.log_path} ({len(self.episode_data)} episodes)")


def train_v2(township_code, seed=0, total_timesteps=1_000_000,
             n_steps=2048, batch_size=256, max_conversions=200,
             area_penalty_weight=500.0):
    """Train MaskablePPO with area-weighted constraint."""

    features_csv = FEATURES_DIR / f'township_{township_code}_features.csv'
    adjacency_npz = ADJACENCY_DIR / f'township_{township_code}_adj.npz'
    output_dir = RESULTS_DIR / f'township_{township_code}'
    os.makedirs(output_dir, exist_ok=True)

    if not features_csv.exists():
        print(f"ERROR: Features not found: {features_csv}")
        return
    if not adjacency_npz.exists():
        print(f"ERROR: Adjacency not found: {adjacency_npz}")
        return

    print("=" * 60)
    print(f"  Training DRL v2 (Area-Weighted): Township {township_code}")
    print(f"  Seed: {seed}, Timesteps: {total_timesteps:,}")
    print(f"  Area penalty weight: {area_penalty_weight}")
    print("=" * 60)

    # Create v2 environment
    env = RealDataLandUseEnvV2(
        features_csv=str(features_csv),
        adjacency_npz=str(adjacency_npz),
        max_conversions=max_conversions,
        area_penalty_weight=area_penalty_weight,
    )

    # Memory check
    obs_dim = env.n_swappable * K_PARCEL + K_GLOBAL
    rollout_mb = n_steps * obs_dim * 4 / (1024 ** 2)
    print(f"\n  Memory: obs_dim={obs_dim:,}, rollout={rollout_mb:.0f} MB")

    env = Monitor(env)

    # Create model (K_GLOBAL=9 for area deviation feature)
    model = MaskablePPO(
        ParcelScoringPolicy,
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=3,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / 'tb_logs_v2'),
        policy_kwargs=dict(
            k_parcel=K_PARCEL,
            k_global=K_GLOBAL,    # 9 (includes area deviation)
            scorer_hiddens=[128, 64],
            value_hiddens=[128, 64],
        ),
        device='auto',
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    scorer_input = K_PARCEL + K_GLOBAL
    print(f"\nPolicy: ParcelScoringPolicy (area-weighted v2)")
    print(f"  Scorer input: {scorer_input} (10 parcel + 9 global)")
    print(f"  Total parameters: {total_params:,}")

    # Train
    log_path = output_dir / f'training_log_v2_seed{seed}.json'
    callback = MetricsCallbackV2(log_path=str(log_path))

    print(f"\nStarting training for {total_timesteps:,} timesteps...\n")
    t_start = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    train_time = time.time() - t_start

    # Save model
    model_path = output_dir / f'model_v2_seed{seed}'
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}.zip")

    # Save scorer weights
    weights_path = output_dir / f'scorer_weights_v2_seed{seed}.pt'
    torch.save({
        'scorer_net': model.policy.scorer_net.state_dict(),
        'value_net': model.policy.value_net.state_dict(),
        'k_parcel': K_PARCEL,
        'k_global': K_GLOBAL,
        'scorer_hiddens': [128, 64],
        'value_hiddens': [128, 64],
    }, str(weights_path))
    print(f"Scorer weights saved: {weights_path}")

    # Summary
    if callback.episode_data:
        last_100 = callback.episode_data[-100:]
        print(f"\n{'='*60}")
        print(f"  Training Summary v2 (last 100 episodes)")
        print(f"{'='*60}")
        print(f"  Training time: {train_time/3600:.1f} hours")
        print(f"  Avg reward:          {np.mean([d['reward'] for d in last_100]):.4f}")
        print(f"  Avg slope:           {np.mean([d['avg_slope'] for d in last_100]):.4f}")
        print(f"  Avg contiguity:      {np.mean([d['contiguity'] for d in last_100]):.4f}")
        print(f"  Avg area deviation:  {np.mean([d['area_deviation'] for d in last_100]):.6f}")
        print(f"  Avg area change (m2):{np.mean([d['area_change_m2'] for d in last_100]):+.0f}")
        print(f"  Avg pairs/ep:        {np.mean([d['completed_pairs'] for d in last_100]):.1f}")
        print(f"  Avg farmland change: {np.mean([d['farmland_change'] for d in last_100]):+.1f}")
        print(f"  Total episodes:      {len(callback.episode_data)}")
        print(f"{'='*60}")

    return model, callback


def main():
    parser = argparse.ArgumentParser(description='Train DRL v2 with area constraint')
    parser.add_argument('--township', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-conversions', type=int, default=200)
    parser.add_argument('--area-penalty', type=float, default=500.0)
    args = parser.parse_args()

    train_v2(
        township_code=args.township,
        seed=args.seed,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_conversions=args.max_conversions,
        area_penalty_weight=args.area_penalty,
    )


if __name__ == '__main__':
    main()
