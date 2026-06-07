# -*- coding: utf-8 -*-
"""Kappa (swaps-per-step) ablation on Township A.

Trains MaskablePPO with kappa in {1, 3, 5, 10, 20} on the small township
to quantify how the macro/micro power split affects multi-objective
performance. Budget is held fixed at 100 paired swaps; episode length
H = 100/kappa varies accordingly.

Output:
  results/derived_analyses/paper3_kappa_ablation_results.json
  results/tables/paper3_kappa_ablation_table.tex
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from block_level_env import BlockLevelEnv  # noqa: E402
from paper3_paths import DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

TOWNSHIP_CODE = "500227109"  # Township A
TOTAL_BUDGET = 100
KAPPAS = [1, 3, 5, 10, 20]
TIMESTEPS = 50_000  # plateau by ~50K per Section 3.8

WEIGHTS = dict(slope_weight=2000.0, cont_weight=500.0,
               baimu_weight=500.0, baimu_bonus=20.0)

OUT_JSON = DERIVED_RESULTS_DIR / "paper3_kappa_ablation_results.json"
OUT_TEX = TABLES_DIR / "paper3_kappa_ablation_table.tex"


def mask_fn(env):
    return env.action_masks()


def train_and_eval_one(kappa: int, seed: int = 0) -> dict:
    print(f"\n=== kappa={kappa} (H={TOTAL_BUDGET // kappa}), seed={seed} ===", flush=True)
    t0 = time.time()
    env = BlockLevelEnv(TOWNSHIP_CODE, total_budget=TOTAL_BUDGET,
                        swaps_per_step=kappa, **WEIGHTS)
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
        device="cpu",
    )
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
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
        "kappa": kappa,
        "H_steps": TOTAL_BUDGET // kappa,
        "seed": seed,
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


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for k in KAPPAS:
        results.append(train_and_eval_one(k))
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity of the block-level DRL agent to the swaps-per-step parameter $\kappa$ on Township A. "
                 r"The total paired-swap budget is fixed at 100; the MDP horizon is $H = 100/\kappa$. "
                 r"Each $\kappa$ is trained for 50{,}000 timesteps with seed 0 on CPU and evaluated deterministically.}")
    lines.append(r"\label{tab:kappa_ablation}")
    lines.append(r"\begin{tabular}{rrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$\kappa$ & $H$ & Slope (\%) & Cont. & Baimu cnt & Baimu ha & Reward \\")
    lines.append(r"\midrule")
    for r in results:
        lines.append(
            f"{r['kappa']} & {r['H_steps']} & "
            f"${r['slope_change_pct']:+.2f}$ & ${r['cont_change']:+.3f}$ & "
            f"${r['baimu_count_change']:+d}$ & ${r['baimu_area_change_ha']:+.1f}$ & "
            f"{r['total_reward']:.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
