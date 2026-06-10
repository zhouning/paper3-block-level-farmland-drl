# -*- coding: utf-8 -*-
"""Kappa (swaps-per-step) ablation for block-level DRL.

Trains MaskablePPO with varying kappa to quantify how the macro/micro power
split affects multi-objective performance. Budget is held fixed at 100 paired
swaps; episode length H = 100/kappa varies accordingly.

Output:
  results/derived_analyses/paper3_kappa_ablation_results.json
  results/tables/paper3_kappa_ablation_table.tex
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_paths import DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

TOWNSHIPS = {
    "A": ("500227109", "Township A", "78 blocks"),
    "B": ("500227108", "Township B", "132 blocks"),
    "C": ("500227105", "Township C", "338 blocks"),
}

TOTAL_BUDGET = 100
DEFAULT_KAPPAS = [1, 3, 5, 10, 20]
DEFAULT_TIMESTEPS = 50_000  # plateau by ~50K per Section 3.8

WEIGHTS = dict(slope_weight=2000.0, cont_weight=500.0,
               baimu_weight=500.0, baimu_bonus=20.0)


def mask_fn(env):
    return env.action_masks()


def parse_kappas(raw: str) -> list[int]:
    kappas = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not kappas:
        raise argparse.ArgumentTypeError("At least one kappa value is required.")
    invalid = [k for k in kappas if k <= 0 or k > TOTAL_BUDGET]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"kappa values must be positive and no larger than {TOTAL_BUDGET}: {invalid}"
        )
    return kappas


def output_paths(suffix: str | None) -> tuple[Path, Path]:
    if suffix:
        stem = f"paper3_kappa_ablation_{suffix}"
        return (
            DERIVED_RESULTS_DIR / f"{stem}_results.json",
            TABLES_DIR / f"{stem}_table.tex",
        )
    return (
        DERIVED_RESULTS_DIR / "paper3_kappa_ablation_results.json",
        TABLES_DIR / "paper3_kappa_ablation_table.tex",
    )


def train_and_eval_one(
    township_code: str,
    township_label: str,
    kappa: int,
    seed: int,
    timesteps: int,
    device: str,
) -> dict:
    print(f"\n=== kappa={kappa} (H={TOTAL_BUDGET // kappa}), seed={seed} ===", flush=True)
    t0 = time.time()
    env = BlockLevelEnv(township_code, total_budget=TOTAL_BUDGET,
                        swaps_per_step=kappa, **WEIGHTS)
    n_blocks = env.n_blocks
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
        device=device,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    t_train = time.time() - t0

    # Deterministic evaluation
    obs, info = env.reset(seed=0)
    total_r = 0.0
    while True:
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += float(r)
        if terminated or truncated:
            break

    out = {
        "township_label": township_label,
        "township_code": township_code,
        "n_blocks": n_blocks,
        "kappa": kappa,
        "H_steps": TOTAL_BUDGET // kappa,
        "seed": seed,
        "timesteps": timesteps,
        "train_seconds": t_train,
        "total_reward": total_r,
        "slope_change_pct": info["slope_change_pct"],
        "cont_change": info["cont_change"],
        "baimu_count_change": info["baimu_count_change"],
        "baimu_area_change_ha": info["baimu_area_change_ha"],
        "budget_used": info["budget_used"],
    }
    print(
        f"  trained in {t_train:.1f}s; slope={out['slope_change_pct']:+.2f}%, "
        f"cont={out['cont_change']:+.3f}, baimu_cnt={out['baimu_count_change']:+d}, "
        f"baimu_ha={out['baimu_area_change_ha']:+.1f}, R={out['total_reward']:.1f}",
        flush=True,
    )
    return out


def write_table(results: list[dict], out_tex: Path) -> None:
    if not results:
        raise ValueError("No results to write.")

    first = results[0]
    township_label = first["township_label"]
    n_blocks = first["n_blocks"]
    timesteps = first["timesteps"]
    seed = first["seed"]
    rewards = [r["total_reward"] for r in results]
    best_idx = int(np.argmax(rewards))
    label = "tab:kappa_ablation"
    if out_tex.stem != "paper3_kappa_ablation_table":
        label_suffix = out_tex.stem.removeprefix("paper3_kappa_ablation_")
        label_suffix = label_suffix.removesuffix("_table").replace("-", "_")
        label = f"tab:kappa_ablation_{label_suffix}"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Sensitivity of the block-level DRL agent to the "
        rf"swaps-per-step parameter $\kappa$ on {township_label} ({n_blocks} blocks). "
        rf"The total paired-swap budget is fixed at 100; the MDP horizon is "
        rf"$H = 100/\kappa$. Each $\kappa$ is trained for {timesteps:,} timesteps "
        rf"with seed {seed} on CPU and evaluated deterministically. ``Budget used'' "
        rf"is the number of paired swaps actually committed in the deterministic "
        rf"episode out of the 100-swap cap.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$\kappa$ & $H$ & Slope (\%) & Cont. & Baimu cnt & Baimu ha & Budget used & Reward \\")
    lines.append(r"\midrule")
    for idx, r in enumerate(results):
        reward = f"{r['total_reward']:.1f}"
        if idx == best_idx:
            reward = rf"\mathbf{{{reward}}}"
        lines.append(
            f"{r['kappa']} & {r['H_steps']} & "
            f"${r['slope_change_pct']:+.2f}$ & ${r['cont_change']:+.3f}$ & "
            f"${r['baimu_count_change']:+d}$ & ${r['baimu_area_change_ha']:+.1f}$ & "
            f"{r['budget_used']}/100 & ${reward}$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--township", choices=sorted(TOWNSHIPS), default="A")
    parser.add_argument("--kappas", type=parse_kappas,
                        default=DEFAULT_KAPPAS,
                        help="Comma-separated kappa values, e.g. 3,5,10.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--suffix", default=None,
                        help="Optional output suffix, e.g. b_seed0.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    township_code, township_label, _ = TOWNSHIPS[args.township]
    out_json, out_tex = output_paths(args.suffix)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for k in args.kappas:
        results.append(
            train_and_eval_one(
                township_code=township_code,
                township_label=township_label,
                kappa=k,
                seed=args.seed,
                timesteps=args.timesteps,
                device=args.device,
            )
        )
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    write_table(results, out_tex)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
