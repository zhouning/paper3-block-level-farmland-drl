# -*- coding: utf-8 -*-
"""Audit farmland-area drift caused by parcel-count-conserving swaps.

The main environment conserves farmland parcel count, not farmland area. This
script replays the final configurations for baselines, Reward-Greedy, and DRL
and reports the net farmland-area change.

Requires controlled-access parcel geometry through PAPER3_DLTB_PATH.

Outputs:
  results/derived_analyses/paper3_area_drift_results.json
  results/tables/paper3_area_drift_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines_block import (  # noqa: E402
    run_greedy_global,
    run_greedy_sequential,
    run_random_block,
    run_round_robin,
)
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_paths import BLOCK_RESULTS_DIR, DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402

TOWNSHIPS = {
    "A": "500227109",
    "B": "500227108",
    "C": "500227105",
}
RESULT_DIR_OVERRIDES = {
    "500227108": "township_500227108_v2",
}
OUT_JSON = DERIVED_RESULTS_DIR / "paper3_area_drift_results.json"
OUT_TEX = TABLES_DIR / "paper3_area_drift_table.tex"
REWARD_GREEDY_JSON = DERIVED_RESULTS_DIR / "paper3_reward_greedy_results.json"


def _area_record(label: str, code: str, method: str, init_area: float, final_area: float) -> dict:
    delta_ha = (final_area - init_area) / 10000.0
    delta_pct = 100.0 * (final_area - init_area) / (init_area + 1e-12)
    return {
        "township_label": label,
        "township_code": code,
        "method": method,
        "initial_farmland_area_ha": init_area / 10000.0,
        "final_farmland_area_ha": final_area / 10000.0,
        "farmland_area_change_ha": delta_ha,
        "farmland_area_change_pct": delta_pct,
    }


def _run_mutating_baseline(env: BlockLevelEnv, label: str, code: str, method: str, fn) -> dict:
    env.reset(seed=0)
    init_area = float(env.total_farm_area)
    fn(env)
    return _area_record(label, code, method, init_area, float(env.total_farm_area))


def _run_random_mean(env: BlockLevelEnv, label: str, code: str) -> dict:
    records = []
    for seed in range(5):
        env.reset(seed=0)
        init_area = float(env.total_farm_area)
        run_random_block(env, seed=seed)
        records.append(_area_record(label, code, "Random-Block", init_area, float(env.total_farm_area)))
    vals_ha = np.array([r["farmland_area_change_ha"] for r in records], dtype=float)
    vals_pct = np.array([r["farmland_area_change_pct"] for r in records], dtype=float)
    rec = records[0].copy()
    rec["method"] = "Random-Block"
    rec["farmland_area_change_ha"] = float(vals_ha.mean())
    rec["farmland_area_change_ha_std"] = float(vals_ha.std(ddof=1))
    rec["farmland_area_change_pct"] = float(vals_pct.mean())
    rec["farmland_area_change_pct_std"] = float(vals_pct.std(ddof=1))
    rec["final_farmland_area_ha"] = rec["initial_farmland_area_ha"] + rec["farmland_area_change_ha"]
    rec["final_farmland_area_ha_std"] = rec["farmland_area_change_ha_std"]
    rec["aggregation"] = "mean_sd_over_5_seeds"
    rec["per_seed"] = records
    return rec


def _run_reward_greedy_area(env: BlockLevelEnv, label: str, code: str) -> dict:
    reward_greedy = json.loads(REWARD_GREEDY_JSON.read_text(encoding="utf-8"))
    history = reward_greedy[label]["block_history"]

    env.reset(seed=0)
    init_area = float(env.total_farm_area)
    for entry in history:
        env.step(int(entry["block"]))

    return _area_record(label, code, "Reward-Greedy", init_area, float(env.total_farm_area))


def _run_drl_mean(env: BlockLevelEnv, label: str, code: str) -> dict:
    subdir = RESULT_DIR_OVERRIDES.get(code, f"township_{code}")
    records = []
    for seed in range(5):
        eval_path = BLOCK_RESULTS_DIR / subdir / f"block_eval_seed{seed}.json"
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        env.reset(seed=0)
        init_area = float(env.total_farm_area)
        for entry in ev["block_history"]:
            env.step(int(entry["block"]))
        records.append(_area_record(label, code, "DRL", init_area, float(env.total_farm_area)))

    vals_ha = np.array([r["farmland_area_change_ha"] for r in records], dtype=float)
    vals_pct = np.array([r["farmland_area_change_pct"] for r in records], dtype=float)
    rec = records[0].copy()
    rec["method"] = "DRL"
    rec["farmland_area_change_ha"] = float(vals_ha.mean())
    rec["farmland_area_change_ha_std"] = float(vals_ha.std(ddof=1))
    rec["farmland_area_change_pct"] = float(vals_pct.mean())
    rec["farmland_area_change_pct_std"] = float(vals_pct.std(ddof=1))
    rec["final_farmland_area_ha"] = rec["initial_farmland_area_ha"] + rec["farmland_area_change_ha"]
    rec["final_farmland_area_ha_std"] = rec["farmland_area_change_ha_std"]
    rec["aggregation"] = "mean_sd_over_5_seeds"
    rec["per_seed"] = records
    return rec


def _fmt(rec: dict) -> str:
    ha = rec["farmland_area_change_ha"]
    pct = rec["farmland_area_change_pct"]
    if "farmland_area_change_ha_std" in rec:
        return f"${ha:+.1f}\\pm{rec['farmland_area_change_ha_std']:.1f}$ & ${pct:+.2f}\\pm{rec['farmland_area_change_pct_std']:.2f}$"
    return f"${ha:+.1f}$ & ${pct:+.2f}$"


def write_table(records: list[dict]) -> None:
    by_key = {(r["township_label"], r["method"]): r for r in records}
    methods = [
        "Greedy-Global",
        "Greedy-Sequential",
        "Random-Block",
        "Round-Robin",
        "Reward-Greedy",
        "DRL",
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Farmland-area balance audit. The optimization engine conserves farmland parcel count, not farmland area. This post-hoc audit replays each final configuration and reports the net farmland-area change. Random-Block and DRL are reported as mean $\pm$ standard deviation over five seeds; deterministic baselines report a single trajectory. Values are derived from controlled-access parcel geometry and released here only as aggregate diagnostics.}",
        r"\label{tab:area_drift}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Township & Method & $\Delta$ farmland area (ha) & $\Delta$ farmland area (\%) \\",
        r"\midrule",
    ]
    for label in ["A", "B", "C"]:
        for method in methods:
            rec = by_key[(label, method)]
            lines.append(f"{label} & {method} & {_fmt(rec)} \\\\")
        if label != "C":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = []
    for label, code in TOWNSHIPS.items():
        print(f"=== Township {label} ({code}) ===", flush=True)
        env = BlockLevelEnv(code, total_budget=100, swaps_per_step=5)
        records.append(_run_mutating_baseline(env, label, code, "Greedy-Global", run_greedy_global))
        records.append(_run_mutating_baseline(env, label, code, "Greedy-Sequential", run_greedy_sequential))
        records.append(_run_random_mean(env, label, code))
        records.append(_run_mutating_baseline(env, label, code, "Round-Robin", run_round_robin))
        records.append(_run_reward_greedy_area(env, label, code))
        records.append(_run_drl_mean(env, label, code))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_table(records)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
