# -*- coding: utf-8 -*-
"""DRL vs Reward-Greedy trajectory comparison on Township B.

This is the figure that supports the paper's core claim that on Township B
DRL discovers a strategy Reward-Greedy cannot match: DRL preserves and grows
baimu fang area while RG dismantles it in pursuit of slope reduction.

Reads:
  results/blocks/township_500227108_v2/block_eval_seed0.json
    (DRL block_history)
  results/derived_analyses/paper3_reward_greedy_results.json
    (Reward-Greedy block_history for B)

Replays both sequences, records per-step slope, baimu_count, baimu_area,
and renders a 3-panel figure: slope, baimu_count, baimu_area vs step.

Output: figures/paper3_trajectory_B_compare.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_paths import BLOCK_RESULTS_DIR, DERIVED_RESULTS_DIR, FIGURES_DIR  # noqa: E402

DRL_EVAL = BLOCK_RESULTS_DIR / "township_500227108_v2" / "block_eval_seed0.json"
RG_RESULTS = DERIVED_RESULTS_DIR / "paper3_reward_greedy_results.json"
OUT_PNG = FIGURES_DIR / "paper3_trajectory_B_compare.png"
TOWNSHIP_CODE = "500227108"


def replay_and_record(history: list[dict]) -> dict:
    env = BlockLevelEnv(TOWNSHIP_CODE, total_budget=100, swaps_per_step=5)
    env.reset(seed=0)
    init_baimu_count = env.initial_baimu_count
    init_baimu_area = env.initial_baimu_area

    steps = [0]
    slope_pct = [0.0]
    baimu_cnt_chg = [0]
    baimu_area_chg_ha = [0.0]

    for entry in history:
        action = int(entry["block"])
        env.step(action)
        steps.append(len(steps))
        slope_pct.append(
            100.0 * (env.avg_farmland_slope - env.initial_slope) / (abs(env.initial_slope) + 1e-8)
        )
        baimu_cnt_chg.append(env.baimu_count - init_baimu_count)
        baimu_area_chg_ha.append((env.baimu_total_area - init_baimu_area) / 10000.0)

    return {
        "steps": steps,
        "slope_pct": slope_pct,
        "baimu_cnt_chg": baimu_cnt_chg,
        "baimu_area_chg_ha": baimu_area_chg_ha,
    }


def main() -> None:
    drl_eval = json.load(DRL_EVAL.open())
    rg_all = json.load(RG_RESULTS.open())
    rg_b = rg_all["B"]

    print("Replaying DRL trajectory on B...")
    drl_traj = replay_and_record(drl_eval["block_history"])
    print("Replaying Reward-Greedy trajectory on B...")
    rg_traj = replay_and_record(rg_b["block_history"])

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)

    # Panel 1: slope
    axes[0].plot(drl_traj["steps"], drl_traj["slope_pct"], "o-",
                 color="#2980b9", lw=1.8, ms=4.5, label="DRL (seed 0)")
    axes[0].plot(rg_traj["steps"], rg_traj["slope_pct"], "s--",
                 color="#c0392b", lw=1.8, ms=4.5, label="Reward-Greedy")
    axes[0].axhline(0, color="gray", lw=0.7, ls=":", alpha=0.6)
    axes[0].set_ylabel("Slope change (%)")
    axes[0].set_title("Township B (132 blocks): DRL vs.\\ Reward-Greedy per-step trajectory")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: baimu count
    axes[1].plot(drl_traj["steps"], drl_traj["baimu_cnt_chg"], "o-",
                 color="#2980b9", lw=1.8, ms=4.5, label="DRL")
    axes[1].plot(rg_traj["steps"], rg_traj["baimu_cnt_chg"], "s--",
                 color="#c0392b", lw=1.8, ms=4.5, label="Reward-Greedy")
    axes[1].axhline(0, color="gray", lw=0.7, ls=":", alpha=0.6)
    axes[1].set_ylabel(r"$\Delta$ baimu fang count")
    axes[1].grid(alpha=0.3)

    # Panel 3: baimu area
    axes[2].plot(drl_traj["steps"], drl_traj["baimu_area_chg_ha"], "o-",
                 color="#2980b9", lw=1.8, ms=4.5, label="DRL")
    axes[2].plot(rg_traj["steps"], rg_traj["baimu_area_chg_ha"], "s--",
                 color="#c0392b", lw=1.8, ms=4.5, label="Reward-Greedy")
    axes[2].axhline(0, color="gray", lw=0.7, ls=":", alpha=0.6)
    axes[2].set_ylabel(r"$\Delta$ baimu fang area (ha)")
    axes[2].set_xlabel("MDP step")
    axes[2].grid(alpha=0.3)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"DRL final: slope={drl_traj['slope_pct'][-1]:+.2f}%, "
          f"baimu_cnt={drl_traj['baimu_cnt_chg'][-1]:+d}, "
          f"baimu_ha={drl_traj['baimu_area_chg_ha'][-1]:+.1f}")
    print(f"RG  final: slope={rg_traj['slope_pct'][-1]:+.2f}%, "
          f"baimu_cnt={rg_traj['baimu_cnt_chg'][-1]:+d}, "
          f"baimu_ha={rg_traj['baimu_area_chg_ha'][-1]:+.1f}")


if __name__ == "__main__":
    main()
