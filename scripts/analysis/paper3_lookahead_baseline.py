# -*- coding: utf-8 -*-
"""Limited-lookahead baseline for Paper 3.

This script tests whether a shallow deterministic planner can reproduce the
multi-step advantage attributed to the trained DRL scheduler. It uses the same
BlockLevelEnv reward and the same connectivity-aware within-block execution
engine as the manuscript's Reward-Greedy baseline, but evaluates short action
sequences instead of only the next block.

Run examples:
    python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
    python scripts/analysis/paper3_lookahead_baseline.py --depth 2 --beam-width 12
    python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8

Outputs:
    results/derived_analyses/paper3_lookahead_d<depth>_b<beam>[_suffix]_results.json
    results/tables/paper3_lookahead_d<depth>_b<beam>[_suffix]_table_fragment.tex
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

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

WEIGHTS = dict(
    slope_weight=2000.0,
    cont_weight=500.0,
    baimu_weight=500.0,
    baimu_bonus=20.0,
)


@dataclass(frozen=True)
class LookaheadDecision:
    action: int
    score: float
    sequence: tuple[int, ...]


SnapshotFn = Callable[[Any], Any]
RestoreFn = Callable[[Any, Any], None]


def _sanitize_suffix(suffix: str) -> str:
    cleaned = []
    for ch in suffix.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_"}:
            cleaned.append("_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def build_output_paths(depth: int, beam_width: int, suffix: str) -> tuple[Path, Path]:
    beam_label = "all" if beam_width == 0 else str(beam_width)
    stem = f"paper3_lookahead_d{depth}_b{beam_label}"
    suffix_clean = _sanitize_suffix(suffix)
    if suffix_clean:
        stem = f"{stem}_{suffix_clean}"
    return (
        DERIVED_RESULTS_DIR / f"{stem}_results.json",
        TABLES_DIR / f"{stem}_table_fragment.tex",
    )


def snapshot_block_env(env: BlockLevelEnv) -> dict[str, Any]:
    """Capture mutable BlockLevelEnv state for deterministic rollouts."""
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


def restore_block_env(env: BlockLevelEnv, snap: dict[str, Any]) -> None:
    """Restore BlockLevelEnv mutable state captured by snapshot_block_env."""
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


def _valid_actions(env: Any) -> np.ndarray:
    mask = np.asarray(env.action_masks(), dtype=bool)
    return np.where(mask)[0]


def _rank_actions_by_immediate_reward(
    env: Any,
    actions: np.ndarray,
    beam_width: int,
    snapshot_fn: SnapshotFn,
    restore_fn: RestoreFn,
) -> list[tuple[int, float, bool, bool]]:
    root = snapshot_fn(env)
    scored: list[tuple[int, float, bool, bool]] = []

    for action in actions:
        restore_fn(env, root)
        _, reward, terminated, truncated, _ = env.step(int(action))
        scored.append((int(action), float(reward), bool(terminated), bool(truncated)))

    restore_fn(env, root)
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    if beam_width == 0:
        return scored
    return scored[:beam_width]


def _evaluate_from_state(
    env: Any,
    depth: int,
    beam_width: int,
    snapshot_fn: SnapshotFn,
    restore_fn: RestoreFn,
) -> tuple[float, tuple[int, ...]]:
    if depth <= 0:
        return 0.0, ()

    actions = _valid_actions(env)
    if len(actions) == 0:
        return 0.0, ()

    root = snapshot_fn(env)
    candidates = _rank_actions_by_immediate_reward(
        env, actions, beam_width, snapshot_fn, restore_fn
    )

    best_score = -np.inf
    best_sequence: tuple[int, ...] = ()

    for action, immediate, terminated, truncated in candidates:
        restore_fn(env, root)
        _, reward, is_terminal, is_truncated, _ = env.step(action)
        if terminated != is_terminal or truncated != is_truncated:
            raise RuntimeError("Non-deterministic rollout detected during lookahead")

        if is_terminal or is_truncated:
            future_score, future_sequence = 0.0, ()
        else:
            future_score, future_sequence = _evaluate_from_state(
                env, depth - 1, beam_width, snapshot_fn, restore_fn
            )

        score = float(reward) + future_score
        sequence = (action,) + future_sequence
        if score > best_score:
            best_score = score
            best_sequence = sequence

    restore_fn(env, root)
    return best_score, best_sequence


def select_lookahead_action(
    env: Any,
    depth: int,
    beam_width: int,
    snapshot_fn: SnapshotFn = snapshot_block_env,
    restore_fn: RestoreFn = restore_block_env,
) -> LookaheadDecision:
    """Select the first action from the best finite-lookahead sequence."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if beam_width < 0:
        raise ValueError("beam_width must be >= 0; use 0 to evaluate all valid actions")

    root = snapshot_fn(env)
    score, sequence = _evaluate_from_state(
        env, depth, beam_width, snapshot_fn, restore_fn
    )
    restore_fn(env, root)

    if not sequence:
        raise RuntimeError("No valid lookahead action is available")
    return LookaheadDecision(action=sequence[0], score=score, sequence=sequence)


def run_lookahead_baseline(
    township_code: str,
    depth: int,
    beam_width: int,
) -> dict[str, Any]:
    env = BlockLevelEnv(
        township_code,
        total_budget=100,
        swaps_per_step=5,
        **WEIGHTS,
    )
    env.reset(seed=0)

    block_history: list[dict[str, Any]] = []
    total_reward = 0.0
    info: dict[str, Any] | None = None

    for step in range(env.max_steps):
        if len(_valid_actions(env)) == 0:
            break

        decision = select_lookahead_action(
            env,
            depth=depth,
            beam_width=beam_width,
            snapshot_fn=snapshot_block_env,
            restore_fn=restore_block_env,
        )
        _, reward, terminated, truncated, info = env.step(decision.action)
        total_reward += float(reward)
        block_history.append(
            {
                "step": step + 1,
                "block": decision.action,
                "lookahead_score": decision.score,
                "lookahead_sequence": list(decision.sequence),
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
        raise RuntimeError("Lookahead baseline produced no evaluation steps")

    return {
        "method": f"lookahead-depth{depth}-beam{'all' if beam_width == 0 else beam_width}",
        "township_code": township_code,
        "depth": depth,
        "beam_width": beam_width,
        "total_reward": float(total_reward),
        "slope_change_pct": float(info["slope_change_pct"]),
        "cont_change": float(info["cont_change"]),
        "baimu_count_change": int(info["baimu_count_change"]),
        "baimu_area_change_ha": float(info["baimu_area_change_ha"]),
        "budget_used": int(info["budget_used"]),
        "n_steps": len(block_history),
        "block_history": block_history,
    }


def write_outputs(results: dict[str, dict[str, Any]], out_json: Path, out_tex: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = []
    for label in ["A", "B", "C"]:
        if label not in results:
            continue
        r = results[label]
        lines.append(
            f" & Lookahead d={r['depth']} b={r['beam_width']} & "
            f"${r['slope_change_pct']:+.2f}$ & ${r['cont_change']:+.3f}$ & "
            f"${r['baimu_count_change']:+d}$ & ${r['baimu_area_change_ha']:+.1f}$ \\\\"
        )
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret_township_b_result(results: dict[str, dict[str, Any]]) -> str:
    """Return manuscript-facing interpretation guidance for Township B."""
    if "B" not in results:
        return (
            "Township B was not run. Run Township B before revising the scheduler "
            "claim, because B is the critical case where DRL differs from one-step "
            "Reward-Greedy planning."
        )

    result = results["B"]
    baimu_area = float(result["baimu_area_change_ha"])
    depth = int(result.get("depth", -1))
    beam_width = int(result.get("beam_width", -1))
    beam_label = "all valid actions" if beam_width == 0 else f"beam width {beam_width}"

    if baimu_area > 0:
        return (
            f"Township B lookahead depth {depth} ({beam_label}) reaches positive "
            f"baimu-fang area ({baimu_area:+.1f} ha). This indicates that "
            "finite-depth search can reproduce the qualitative Township-B effect; "
            "revise the manuscript to frame the scheduler contribution as "
            "multi-step planning rather than RL-specific learning."
        )

    if baimu_area < 0:
        return (
            f"Township B lookahead depth {depth} ({beam_label}) remains negative "
            f"on baimu-fang area ({baimu_area:+.1f} ha). This does not reproduce "
            "the DRL Township-B effect and strengthens the current claim that the "
            "learned scheduler avoids a trap missed by one-step and tested "
            "finite-depth deterministic planning."
        )

    return (
        f"Township B lookahead depth {depth} ({beam_label}) ends exactly at zero "
        "baimu-fang area change. Treat this as ambiguous: report it as neutral "
        "and run a depth-3 sensitivity check before strengthening or weakening "
        "the learned-scheduler claim."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper3 limited-lookahead baseline")
    parser.add_argument("--township", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=12,
        help="Immediate-reward beam width; use 0 to evaluate all valid actions.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional output filename suffix for sensitivity runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = ["A", "B", "C"] if args.township == "all" else [args.township]
    results: dict[str, dict[str, Any]] = {}

    for label in labels:
        code = TOWNSHIPS[label]
        print(
            f"=== Township {label} ({code}) - lookahead depth={args.depth}, "
            f"beam={args.beam_width} ===",
            flush=True,
        )
        result = run_lookahead_baseline(code, args.depth, args.beam_width)
        results[label] = result
        print(
            f"  slope={result['slope_change_pct']:+.2f}%  "
            f"cont={result['cont_change']:+.4f}  "
            f"baimu_cnt={result['baimu_count_change']:+d}  "
            f"baimu_ha={result['baimu_area_change_ha']:+.1f}  "
            f"R={result['total_reward']:.1f}  steps={result['n_steps']}",
            flush=True,
        )

    out_json, out_tex = build_output_paths(args.depth, args.beam_width, args.suffix)
    write_outputs(results, out_json, out_tex)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_tex}")
    print("\nInterpretation guidance:")
    print(interpret_township_b_result(results))


if __name__ == "__main__":
    main()
