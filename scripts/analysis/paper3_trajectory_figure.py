# -*- coding: utf-8 -*-
"""Generate Township C "invest-then-recover" trajectory figure.

Replays the saved DRL block selection sequence (seed 0) on Township C through
BlockLevelEnv, measuring at every step:
  - cumulative slope_change_pct
  - cumulative contiguity change
  - baimu fang count change vs initial
  - baimu fang area change (ha) vs initial

Saves a 2-panel figure: figures/paper3_trajectory_C.png
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

EVAL_PATH = BLOCK_RESULTS_DIR / "township_500227105" / "block_eval_seed0.json"
OUT_PNG = FIGURES_DIR / "paper3_trajectory_C.png"
DERIVED_JSON = DERIVED_RESULTS_DIR / "paper3_trajectory_C.json"
TOWNSHIP_CODE = "500227105"


def load_trajectory() -> dict:
    if DERIVED_JSON.exists():
        print(f"Reading derived trajectory {DERIVED_JSON}")
        return json.loads(DERIVED_JSON.read_text(encoding="utf-8"))

    with EVAL_PATH.open() as f:
        ev = json.load(f)
    history = ev["block_history"]

    env = BlockLevelEnv(TOWNSHIP_CODE, total_budget=100, swaps_per_step=5)
    env.reset(seed=0)

    init_baimu_count = env.initial_baimu_count
    init_baimu_area = env.initial_baimu_area  # m^2

    steps = [0]
    slope_pct = [0.0]
    cont_chg = [0.0]
    baimu_cnt_chg = [0]
    baimu_area_chg_ha = [0.0]
    blocks_visited = [None]

    for entry in history:
        action = int(entry["block"])
        env.step(action)
        steps.append(entry["step"])
        slope_pct.append(
            100.0 * (env.avg_farmland_slope - env.initial_slope) / (abs(env.initial_slope) + 1e-8)
        )
        cont_chg.append(env.contiguity - env.initial_cont)
        baimu_cnt_chg.append(env.baimu_count - init_baimu_count)
        baimu_area_chg_ha.append((env.baimu_total_area - init_baimu_area) / 10000.0)
        blocks_visited.append(action)

    trajectory = {
        "step": steps,
        "block": blocks_visited,
        "slope_change_pct": slope_pct,
        "contiguity_change": cont_chg,
        "baimu_count_change": baimu_cnt_chg,
        "baimu_area_change_ha": baimu_area_chg_ha,
    }
    DERIVED_JSON.parent.mkdir(parents=True, exist_ok=True)
    DERIVED_JSON.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"Wrote {DERIVED_JSON}")
    return trajectory


def main() -> None:
    trajectory = load_trajectory()
    steps = trajectory["step"]
    slope_pct = trajectory["slope_change_pct"]
    cont_chg = trajectory["contiguity_change"]
    baimu_cnt_chg = trajectory["baimu_count_change"]
    baimu_area_chg_ha = trajectory["baimu_area_change_ha"]
    n_steps = max(steps)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)

    ax_top = axes[0]
    color_s = "#c0392b"
    color_c = "#2980b9"
    ax_top.plot(steps, slope_pct, "o-", color=color_s, lw=1.6, ms=4, label="Slope change (%)")
    ax_top.set_ylabel("Slope change (%)", color=color_s)
    ax_top.tick_params(axis="y", labelcolor=color_s)
    ax_top.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax_top.grid(alpha=0.3)

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(steps, cont_chg, "s-", color=color_c, lw=1.6, ms=4, label="Contiguity change")
    ax_top_r.set_ylabel("Contiguity change", color=color_c)
    ax_top_r.tick_params(axis="y", labelcolor=color_c)

    # Phase shading: "investment" (steps 1-10), "recovery" (steps 11-20)
    ax_top.axvspan(0.5, 10.5, alpha=0.10, color="#f39c12", label="_invest")
    ax_top.axvspan(10.5, 20.5, alpha=0.10, color="#27ae60", label="_recover")
    ax_top.text(5.5, ax_top.get_ylim()[1] * 0.92, "Phase 1: invest",
                ha="center", fontsize=9, color="#7f5410")
    ax_top.text(15.5, ax_top.get_ylim()[1] * 0.92, "Phase 2: recover",
                ha="center", fontsize=9, color="#19612e")

    ax_top.set_title("Township C (338 blocks): per-step trajectory of DRL agent (seed 0)")

    ax_bot = axes[1]
    color_b = "#8e44ad"
    color_a = "#16a085"
    ax_bot.plot(steps, baimu_cnt_chg, "o-", color=color_b, lw=1.8, ms=5,
                label=r"$\Delta$ baimu fang count")
    ax_bot.set_ylabel(r"$\Delta$ baimu fang count", color=color_b)
    ax_bot.tick_params(axis="y", labelcolor=color_b)
    ax_bot.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax_bot.grid(alpha=0.3)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(steps, baimu_area_chg_ha, "^-", color=color_a, lw=1.6, ms=4,
                  label=r"$\Delta$ baimu fang area (ha)")
    ax_bot_r.set_ylabel(r"$\Delta$ baimu fang area (ha)", color=color_a)
    ax_bot_r.tick_params(axis="y", labelcolor=color_a)

    ax_bot.set_xlabel("MDP step (one block selection per step, $\\kappa=5$ swaps)")
    ax_bot.set_xticks(range(0, n_steps + 1, 2))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
