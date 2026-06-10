# -*- coding: utf-8 -*-
"""Area-tolerance transition check for the Paper3 CEUS review.

This script implements the deterministic part of Priority 2 in
``docs/CEUS_REVIEW_ADDITIONAL_EXPERIMENTS.md``. It evaluates Township B under
an area-aware transition engine that:

1. enforces a cumulative farmland-area tolerance around the initial area; and
2. replaces the parcel-level slope comparison with an exact area-weighted mean
   slope improvement check.

The script does not retrain DRL. It runs deterministic baselines plus
Reward-Greedy under the modified engine, then replays the existing Township-B
DRL block histories as a stress test.

Outputs:
    results/derived_analyses/paper3_area_tolerance_check_results.json
    results/tables/paper3_area_tolerance_check_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from baselines_block import run_greedy_global, run_greedy_sequential, run_round_robin  # noqa: E402
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_lookahead_baseline import (  # noqa: E402
    select_lookahead_action,
    snapshot_block_env,
    restore_block_env,
)
from paper3_paths import BLOCK_RESULTS_DIR, DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402


TOWNSHIP_LABEL = "B"
TOWNSHIP_CODE = "500227108"
TOLERANCES_PCT = [0.5, 1.0]
TOTAL_BUDGET = 100
SWAPS_PER_STEP = 5
RESULT_SUBDIR = "township_500227108_v2"

WEIGHTS = dict(
    slope_weight=2000.0,
    cont_weight=500.0,
    baimu_weight=500.0,
    baimu_bonus=20.0,
)

OUT_JSON = DERIVED_RESULTS_DIR / "paper3_area_tolerance_check_results.json"
OUT_TEX = TABLES_DIR / "paper3_area_tolerance_check_table.tex"


def make_env(tolerance_pct: float, reward_weighted: bool = False) -> BlockLevelEnv:
    kwargs = WEIGHTS if reward_weighted else {}
    return BlockLevelEnv(
        TOWNSHIP_CODE,
        total_budget=TOTAL_BUDGET,
        swaps_per_step=SWAPS_PER_STEP,
        area_tolerance_pct=tolerance_pct,
        area_weighted_slope_check=True,
        **kwargs,
    )


def area_metrics(env: BlockLevelEnv) -> dict[str, float]:
    delta_area = env.total_farm_area - env.initial_farm_area
    return {
        "farmland_area_change_ha": delta_area / 10000.0,
        "farmland_area_change_pct": 100.0 * delta_area / (env.initial_farm_area + 1e-8),
    }


def normalize_record(
    env: BlockLevelEnv,
    result: dict[str, Any],
    method: str,
    tolerance_pct: float,
    stage: str,
    seed: int | None = None,
) -> dict[str, Any]:
    record = {
        "township_label": TOWNSHIP_LABEL,
        "township_code": TOWNSHIP_CODE,
        "stage": stage,
        "method": method,
        "area_tolerance_pct": tolerance_pct,
        "area_weighted_slope_check": True,
        "seed": seed,
        "slope_change_pct": float(result["slope_change_pct"]),
        "cont_change": float(result["cont_change"]),
        "baimu_count_change": int(result["baimu_count_change"]),
        "baimu_area_change_ha": float(result["baimu_area_change_ha"]),
        "budget_used": int(result.get("budget_used", result.get("completed_swaps", 0))),
        "n_steps": int(result.get("n_steps", result.get("steps", TOTAL_BUDGET // SWAPS_PER_STEP))),
        "total_reward": float(result["total_reward"]) if "total_reward" in result else None,
    }
    record.update(area_metrics(env))
    return record


def run_reward_greedy(tolerance_pct: float) -> dict[str, Any]:
    env = make_env(tolerance_pct, reward_weighted=True)
    env.reset(seed=0)

    total_reward = 0.0
    block_history: list[dict[str, Any]] = []
    info: dict[str, Any] | None = None

    for step in range(env.max_steps):
        if not env.action_masks().any():
            break

        decision = select_lookahead_action(
            env,
            depth=1,
            beam_width=0,
            snapshot_fn=snapshot_block_env,
            restore_fn=restore_block_env,
        )
        _, reward, terminated, truncated, info = env.step(decision.action)
        total_reward += float(reward)
        block_history.append(
            {
                "step": step + 1,
                "block": int(decision.action),
                "reward": float(reward),
                "swaps": int(info["completed_swaps"]),
                "slope_pct": float(info["slope_change_pct"]),
                "cont": float(info["cont_change"]),
                "baimu_cnt": int(info["baimu_count_change"]),
                "baimu_ha": float(info["baimu_area_change_ha"]),
            }
        )
        if terminated or truncated:
            break

    if info is None:
        raise RuntimeError("Reward-Greedy produced no evaluation steps")

    result = {
        "method": "Reward-Greedy-area-tolerant",
        "slope_change_pct": float(info["slope_change_pct"]),
        "cont_change": float(info["cont_change"]),
        "baimu_count_change": int(info["baimu_count_change"]),
        "baimu_area_change_ha": float(info["baimu_area_change_ha"]),
        "budget_used": int(info["budget_used"]),
        "n_steps": len(block_history),
        "total_reward": total_reward,
        "block_history": block_history,
    }
    return normalize_record(env, result, "Reward-Greedy", tolerance_pct, "deterministic")


def run_drl_replay(tolerance_pct: float, seed: int) -> dict[str, Any]:
    env = make_env(tolerance_pct, reward_weighted=False)
    eval_path = BLOCK_RESULTS_DIR / RESULT_SUBDIR / f"block_eval_seed{seed}.json"
    saved = json.loads(eval_path.read_text(encoding="utf-8"))

    env.reset(seed=0)
    total_reward = 0.0
    info: dict[str, Any] | None = None
    replay_history = []

    for step, entry in enumerate(saved["block_history"], start=1):
        block = int(entry["block"])
        _, reward, terminated, truncated, info = env.step(block)
        total_reward += float(reward)
        replay_history.append(
            {
                "step": step,
                "block": block,
                "reward": float(reward),
                "swaps": int(info["completed_swaps"]),
                "slope_pct": float(info["slope_change_pct"]),
                "cont": float(info["cont_change"]),
                "baimu_cnt": int(info["baimu_count_change"]),
                "baimu_ha": float(info["baimu_area_change_ha"]),
            }
        )
        if terminated or truncated:
            break

    if info is None:
        raise RuntimeError(f"DRL replay seed {seed} produced no evaluation steps")

    result = {
        "method": "DRL-history-replay-area-tolerant",
        "slope_change_pct": float(info["slope_change_pct"]),
        "cont_change": float(info["cont_change"]),
        "baimu_count_change": int(info["baimu_count_change"]),
        "baimu_area_change_ha": float(info["baimu_area_change_ha"]),
        "budget_used": int(info["budget_used"]),
        "n_steps": len(replay_history),
        "total_reward": total_reward,
        "block_history": replay_history,
    }
    return normalize_record(env, result, "DRL-history-replay", tolerance_pct, "stress_test", seed)


def aggregate_drl(records: list[dict[str, Any]], tolerance_pct: float) -> dict[str, Any]:
    fields = [
        "slope_change_pct",
        "cont_change",
        "baimu_count_change",
        "baimu_area_change_ha",
        "farmland_area_change_ha",
        "farmland_area_change_pct",
        "budget_used",
        "n_steps",
        "total_reward",
    ]
    out: dict[str, Any] = {
        "township_label": TOWNSHIP_LABEL,
        "township_code": TOWNSHIP_CODE,
        "stage": "stress_test",
        "method": "DRL-history-replay-mean",
        "area_tolerance_pct": tolerance_pct,
        "area_weighted_slope_check": True,
        "aggregation": "mean_sd_over_5_seeds",
        "n_seeds": len(records),
    }
    for field in fields:
        values = np.array([r[field] for r in records if r[field] is not None], dtype=float)
        out[field] = float(values.mean())
        out[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    out["baimu_count_change"] = float(out["baimu_count_change"])
    return out


def write_table(records: list[dict[str, Any]]) -> None:
    selected = [
        r
        for r in records
        if r["stage"] == "deterministic" or r["method"] == "DRL-history-replay-mean"
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Township B area-tolerance transition check. The modified engine enforces cumulative farmland-area tolerance and uses an exact area-weighted mean-slope improvement check for paired swaps. DRL rows replay existing block histories only and are stress tests, not retrained policies.}",
        r"\label{tab:area_tolerance_check}",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"Tolerance (\%) & Method & Slope (\%) & Cont. & Baimu cnt & Baimu ha & Farm area (\%) & Budget \\",
        r"\midrule",
    ]
    for rec in selected:
        method = rec["method"].replace("DRL-history-replay-mean", "DRL replay mean")
        lines.append(
            f"{rec['area_tolerance_pct']:.1f} & {method} & "
            f"${rec['slope_change_pct']:+.2f}$ & "
            f"${rec['cont_change']:+.3f}$ & "
            f"${rec['baimu_count_change']:+.1f}$ & "
            f"${rec['baimu_area_change_ha']:+.1f}$ & "
            f"${rec['farmland_area_change_pct']:+.2f}$ & "
            f"{rec['budget_used']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records: list[dict[str, Any]] = []
    deterministic = [
        ("Greedy-Global", run_greedy_global),
        ("Greedy-Sequential", run_greedy_sequential),
        ("Round-Robin", run_round_robin),
    ]

    for tol in TOLERANCES_PCT:
        print(f"=== Township B area tolerance {tol:.1f}% ===", flush=True)
        for method, fn in deterministic:
            env = make_env(tol, reward_weighted=False)
            result = fn(env)
            rec = normalize_record(env, result, method, tol, "deterministic")
            records.append(rec)
            print(
                f"  {method}: slope={rec['slope_change_pct']:+.2f}% "
                f"baimu_ha={rec['baimu_area_change_ha']:+.1f} "
                f"farm_area={rec['farmland_area_change_pct']:+.2f}% "
                f"budget={rec['budget_used']}",
                flush=True,
            )

        rec = run_reward_greedy(tol)
        records.append(rec)
        print(
            f"  Reward-Greedy: slope={rec['slope_change_pct']:+.2f}% "
            f"baimu_ha={rec['baimu_area_change_ha']:+.1f} "
            f"farm_area={rec['farmland_area_change_pct']:+.2f}% "
            f"budget={rec['budget_used']}",
            flush=True,
        )

        drl_records = []
        for seed in range(5):
            rec = run_drl_replay(tol, seed)
            records.append(rec)
            drl_records.append(rec)
            print(
                f"  DRL replay seed {seed}: slope={rec['slope_change_pct']:+.2f}% "
                f"baimu_ha={rec['baimu_area_change_ha']:+.1f} "
                f"farm_area={rec['farmland_area_change_pct']:+.2f}% "
                f"budget={rec['budget_used']}",
                flush=True,
            )
        mean_rec = aggregate_drl(drl_records, tol)
        records.append(mean_rec)
        print(
            f"  DRL replay mean: slope={mean_rec['slope_change_pct']:+.2f}% "
            f"baimu_ha={mean_rec['baimu_area_change_ha']:+.1f} "
            f"farm_area={mean_rec['farmland_area_change_pct']:+.2f}% "
            f"budget={mean_rec['budget_used']:.1f}",
            flush=True,
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_table(records)
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
