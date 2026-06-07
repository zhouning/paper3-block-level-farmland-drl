# -*- coding: utf-8 -*-
"""Reward-Greedy strong baseline for Paper 3.

At each step, evaluates *every* unmasked block by simulating the within-block
greedy engine on a deepcopy of the env, computing the SAME multi-component
reward used by DRL (Eq. 4 in the manuscript), and selecting the block with the
highest one-step reward. This is a one-step lookahead baseline that has access
to the same multi-objective signal DRL optimizes — the natural strong baseline
that reviewers asked for.

Run:
    python paper3_reward_greedy_baseline.py
Outputs:
    results/derived_analyses/paper3_reward_greedy_results.json
    results/tables/paper3_reward_greedy_table_fragment.tex
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_paths import DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402

TOWNSHIPS = {
    "A": "500227109",
    "B": "500227108",
    "C": "500227105",
}

# Use the same reward weights as the DRL training config (DEFAULT_CONFIG in
# colab_train_all.py): slope=2000, cont=500, baimu=500, bonus=20.
WEIGHTS = dict(slope_weight=2000.0, cont_weight=500.0,
               baimu_weight=500.0, baimu_bonus=20.0)

OUT_JSON = DERIVED_RESULTS_DIR / "paper3_reward_greedy_results.json"
OUT_TEX = TABLES_DIR / "paper3_reward_greedy_table_fragment.tex"


def _snapshot(env: BlockLevelEnv) -> dict:
    """Capture mutable state needed to roll back env after a trial step."""
    return {
        "land_use": env.land_use.copy(),
        "swapped": env.swapped.copy(),
        "n_farmland": env.n_farmland,
        "n_forest": env.n_forest,
        "total_weighted_slope": env.total_weighted_slope,
        "total_farm_area": env.total_farm_area,
        "farmland_nbr_count": env.farmland_nbr_count.copy(),
        "total_farmland_adj": env.total_farmland_adj,
        "block_farm_avail": env._block_farm_avail.copy(),
        "block_forest_avail": env._block_forest_avail.copy(),
        "swaps_in_block": env.swaps_in_block.copy(),
        "budget_used": env.budget_used,
        "step_count": env.step_count,
        "baimu_count": env.baimu_count,
        "baimu_total_area": env.baimu_total_area,
        "prev_slope": env.prev_slope,
        "prev_cont": env.prev_cont,
        "prev_baimu_count": env.prev_baimu_count,
        "prev_baimu_area": env.prev_baimu_area,
    }


def _restore(env: BlockLevelEnv, snap: dict) -> None:
    env.land_use = snap["land_use"].copy()
    env.swapped = snap["swapped"].copy()
    env.n_farmland = snap["n_farmland"]
    env.n_forest = snap["n_forest"]
    env.total_weighted_slope = snap["total_weighted_slope"]
    env.total_farm_area = snap["total_farm_area"]
    env.farmland_nbr_count = snap["farmland_nbr_count"].copy()
    env.total_farmland_adj = snap["total_farmland_adj"]
    env._block_farm_avail = snap["block_farm_avail"].copy()
    env._block_forest_avail = snap["block_forest_avail"].copy()
    env.swaps_in_block = snap["swaps_in_block"].copy()
    env.budget_used = snap["budget_used"]
    env.step_count = snap["step_count"]
    env.baimu_count = snap["baimu_count"]
    env.baimu_total_area = snap["baimu_total_area"]
    env.prev_slope = snap["prev_slope"]
    env.prev_cont = snap["prev_cont"]
    env.prev_baimu_count = snap["prev_baimu_count"]
    env.prev_baimu_area = snap["prev_baimu_area"]


def run_reward_greedy(township_code: str) -> dict:
    env = BlockLevelEnv(township_code, total_budget=100, swaps_per_step=5,
                        **WEIGHTS)
    env.reset(seed=0)

    block_history = []
    total_reward = 0.0

    for step in range(env.max_steps):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break

        snap = _snapshot(env)
        best_block = -1
        best_r = -np.inf

        for b in valid:
            _restore(env, snap)
            _, r, _, _, _ = env.step(int(b))
            if r > best_r:
                best_r = r
                best_block = int(b)

        # Now actually commit the best block
        _restore(env, snap)
        _, r, terminated, truncated, info = env.step(best_block)
        total_reward += r
        block_history.append({
            "step": step + 1,
            "block": best_block,
            "reward": r,
            "swaps": info["completed_swaps"],
            "slope_pct": info["slope_change_pct"],
            "cont": info["cont_change"],
            "baimu_cnt": info["baimu_count_change"],
            "baimu_ha": info["baimu_area_change_ha"],
        })
        if terminated:
            break

    return {
        "method": "reward-greedy-1step",
        "township_code": township_code,
        "total_reward": total_reward,
        "slope_change_pct": info["slope_change_pct"],
        "cont_change": info["cont_change"],
        "baimu_count_change": info["baimu_count_change"],
        "baimu_area_change_ha": info["baimu_area_change_ha"],
        "budget_used": info["budget_used"],
        "n_steps": len(block_history),
        "block_history": block_history,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, code in TOWNSHIPS.items():
        print(f"=== Township {label} ({code}) — Reward-Greedy 1-step lookahead ===", flush=True)
        results[label] = run_reward_greedy(code)
        r = results[label]
        print(
            f"  slope={r['slope_change_pct']:+.2f}%  cont={r['cont_change']:+.4f}  "
            f"baimu_cnt={r['baimu_count_change']:+d}  baimu_ha={r['baimu_area_change_ha']:+.1f}  "
            f"R={r['total_reward']:.1f}  steps={r['n_steps']}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")

    # LaTeX table fragment: rows for the three townships
    lines = []
    for lab in ["A", "B", "C"]:
        r = results[lab]
        lines.append(
            f" & Reward-Greedy & ${r['slope_change_pct']:+.2f}$ & ${r['cont_change']:+.3f}$ & "
            f"${r['baimu_count_change']:+d}$ & ${r['baimu_area_change_ha']:+.1f}$ \\\\"
        )
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
