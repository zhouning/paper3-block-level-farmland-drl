# -*- coding: utf-8 -*-
"""Post-hoc compactness analysis of baimu fang formed by DRL on each township.

For each (township, seed) we:
  1. Load the saved block_history from block_eval_seed*.json.
  2. Replay the block selection sequence through BlockLevelEnv to recover
     final land_use deterministically (the greedy engine is fully deterministic
     given a block sequence and the initial state).
  3. Identify final farmland connected components via the same BFS logic.
  4. For each component with area >= BAIMU_THRESHOLD_M2 (the *raw* baimu fang
     count), compute the isoperimetric quotient IPQ = 4 * pi * A / P^2, where A
     and P are the area and perimeter of the dissolved farmland polygon. IPQ=1
     for a circle; lower means more elongated / fragmented shape.
  5. Report raw vs qualified counts under increasing IPQ thresholds (0.10, 0.20,
     0.30) so reviewers can see how morphology filtering changes the headline
     result.

Output:
  results/derived_analyses/paper3_compactness_results.json
  results/tables/paper3_compactness_table.tex   (LaTeX-ready table fragment)

Run:
  python paper3_compactness_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

# Make sure src/ modules can be imported when running from the repository.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from block_level_env import BlockLevelEnv, FARMLAND, BAIMU_THRESHOLD_M2  # noqa: E402
from paper3_paths import BLOCK_RESULTS_DIR, DERIVED_RESULTS_DIR, TABLES_DIR  # noqa: E402

TOWNSHIPS = {
    "A": "500227109",
    "B": "500227108",
    "C": "500227105",
}
RESULT_ROOT = BLOCK_RESULTS_DIR
# Township B uses the v2 retrained run that matches the manuscript's reported numbers
# (slope -1.64% etc.); A and C use the standard run.
RESULT_DIR_OVERRIDES = {
    "500227108": "township_500227108_v2",
}
SEEDS = list(range(5))
IPQ_THRESHOLDS = [0.10, 0.20, 0.30]

OUT_JSON = DERIVED_RESULTS_DIR / "paper3_compactness_results.json"
OUT_TEX = TABLES_DIR / "paper3_compactness_table.tex"


def replay_episode(env: BlockLevelEnv, history: list[dict]) -> None:
    """Drive env.step with the saved block sequence."""
    env.reset(seed=0)
    for entry in history:
        action = int(entry["block"])
        env.step(action)


def farmland_components(env: BlockLevelEnv) -> list[np.ndarray]:
    """Return list of arrays of parcel indices, one per connected farmland component."""
    visited = np.zeros(env.n_parcels, dtype=bool)
    comps: list[np.ndarray] = []
    for s in range(env.n_parcels):
        if visited[s] or env.land_use[s] != FARMLAND:
            continue
        comp: list[int] = []
        q = deque([s])
        visited[s] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for u in env.adjacency[v]:
                if not visited[u] and env.land_use[u] == FARMLAND:
                    visited[u] = True
                    q.append(u)
        comps.append(np.asarray(comp, dtype=np.intp))
    return comps


def load_township_geometries(township_code: str) -> gpd.GeoDataFrame:
    """Load the same swappable GeoDataFrame BlockLevelEnv used, in the projected CRS."""
    from block_level_env import DLTB_PATH, PROJ_CRS, _classify_type

    gdf = gpd.read_file(DLTB_PATH, where=f"QSDWDM LIKE '{township_code}%'")
    gdf["type_code"] = gdf["DLBM"].apply(_classify_type)
    gdf_swap = gdf[gdf["type_code"].isin([FARMLAND, 2])].copy()  # 2 == FOREST
    gdf_swap = gdf_swap.reset_index(drop=True)
    return gdf_swap.to_crs(PROJ_CRS)


def compactness_for_components(
    comps: list[np.ndarray],
    env: BlockLevelEnv,
    geoms_proj: gpd.GeoDataFrame,
) -> list[dict]:
    """For each component compute area, perimeter, IPQ. Filter to baimu fang."""
    out = []
    for cid, parcels in enumerate(comps):
        area_m2 = float(env.areas[parcels].sum())
        if area_m2 < BAIMU_THRESHOLD_M2:
            continue
        # Dissolve polygons in the projected CRS (so perimeter is in metres).
        polys = list(geoms_proj.geometry.iloc[parcels].values)
        dissolved = unary_union(polys)
        perim_m = float(dissolved.length)
        if perim_m <= 0.0:
            ipq = 0.0
        else:
            ipq = 4.0 * math.pi * area_m2 / (perim_m ** 2)
        out.append({
            "comp_id": cid,
            "n_parcels": int(len(parcels)),
            "area_m2": area_m2,
            "area_ha": area_m2 / 10000.0,
            "perimeter_m": perim_m,
            "ipq": ipq,
        })
    return out


def analyze_one(township_label: str, township_code: str, seed: int) -> dict:
    subdir = RESULT_DIR_OVERRIDES.get(township_code, f"township_{township_code}")
    eval_path = RESULT_ROOT / subdir / f"block_eval_seed{seed}.json"
    with eval_path.open() as f:
        ev = json.load(f)
    history = ev["block_history"]

    env = BlockLevelEnv(township_code, total_budget=100, swaps_per_step=5)
    geoms_proj = load_township_geometries(township_code)
    if len(geoms_proj) != env.n_parcels:
        raise RuntimeError(
            f"Parcel count mismatch on {township_code}: env has {env.n_parcels}, "
            f"loaded geometries have {len(geoms_proj)}"
        )

    # Initial-state IPQ (for fairness baseline; same on every seed but cheap to record)
    env.reset(seed=0)
    init_comps = farmland_components(env)
    init_records = compactness_for_components(init_comps, env, geoms_proj)
    init_qualified = {f"init_ipq_ge_{t:.2f}": sum(1 for r in init_records if r["ipq"] >= t)
                      for t in IPQ_THRESHOLDS}

    # Replay agent's block selection sequence
    replay_episode(env, history)
    final_baimu_count, final_baimu_area = env._count_baimu_fang()

    comps = farmland_components(env)
    baimu_records = compactness_for_components(comps, env, geoms_proj)

    qualified = {f"ipq_ge_{t:.2f}": sum(1 for r in baimu_records if r["ipq"] >= t)
                 for t in IPQ_THRESHOLDS}

    return {
        "township_label": township_label,
        "township_code": township_code,
        "seed": seed,
        "saved_baimu_count_change": ev["baimu_count_change"],
        "raw_final_baimu_count": int(final_baimu_count),
        "raw_final_baimu_area_ha": final_baimu_area / 10000.0,
        "n_baimu_records": len(baimu_records),
        "n_init_baimu_records": len(init_records),
        "ipq_values": [r["ipq"] for r in baimu_records],
        "areas_ha": [r["area_ha"] for r in baimu_records],
        "init_ipq_values": [r["ipq"] for r in init_records],
        "init_areas_ha": [r["area_ha"] for r in init_records],
        **qualified,
        **init_qualified,
    }


def main() -> None:
    all_records: list[dict] = []
    for label, code in TOWNSHIPS.items():
        for seed in SEEDS:
            print(f"=== Township {label} ({code}) seed={seed} ===", flush=True)
            try:
                rec = analyze_one(label, code, seed)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                continue
            all_records.append(rec)
            print(
                f"  raw final baimu count = {rec['raw_final_baimu_count']}, "
                f"qualified at IPQ>=0.10/0.20/0.30 = "
                f"{rec['ipq_ge_0.10']}/{rec['ipq_ge_0.20']}/{rec['ipq_ge_0.30']}",
                flush=True,
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\nWrote per-seed records to {OUT_JSON}")

    # Aggregate per township: means over 5 seeds for raw + each IPQ threshold.
    df = pd.DataFrame(all_records)
    if df.empty:
        print("No successful runs; skipping table.")
        return
    summary = df.groupby("township_label").agg(
        raw_count_mean=("raw_final_baimu_count", "mean"),
        raw_count_std=("raw_final_baimu_count", "std"),
        raw_area_ha_mean=("raw_final_baimu_area_ha", "mean"),
        ipq010_mean=("ipq_ge_0.10", "mean"),
        ipq020_mean=("ipq_ge_0.20", "mean"),
        ipq030_mean=("ipq_ge_0.30", "mean"),
        init_ipq010_mean=("init_ipq_ge_0.10", "mean"),
        init_ipq020_mean=("init_ipq_ge_0.20", "mean"),
        init_ipq030_mean=("init_ipq_ge_0.30", "mean"),
    ).reset_index()

    # Recover initial baimu counts (before optimization) from any eval file's saved
    # baimu_count_change so we can express deltas: change = final - initial.
    # initial_baimu_count is constant per township; pull it from baselines or env.
    init_baimu = {}
    for label, code in TOWNSHIPS.items():
        env = BlockLevelEnv(code, total_budget=100, swaps_per_step=5)
        init_baimu[label] = int(env.initial_baimu_count)

    print("\nInitial baimu counts:", init_baimu)

    # Build LaTeX table — include initial qualified count as a fairness baseline
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Post-hoc compactness audit of \textit{baimu fang} formed by the block-level DRL agent. ")
    lines.append(r"For each township and each evaluation seed we recover the final farmland configuration ")
    lines.append(r"by replaying the agent's block selection sequence, identify all connected farmland ")
    lines.append(r"components with area $\geq$6.67\,ha (the raw \textit{baimu fang} count), and compute ")
    lines.append(r"the isoperimetric quotient $\mathrm{IPQ} = 4\pi A / P^2$ on the dissolved farmland polygon ")
    lines.append(r"of each component (IPQ$=$1 for a circle; lower means more elongated/fragmented shape). ")
    lines.append(r"The ``Init.\ IPQ'' columns report the same compactness counts on the \emph{initial} land-use ")
    lines.append(r"configuration before any optimization, providing a fairness baseline: the Queen-contiguity ")
    lines.append(r"connectivity definition combined with real cadastral parcel shapes typically yields ")
    lines.append(r"low-IPQ components even prior to consolidation, so the post-DRL counts at each ")
    lines.append(r"$\tau$ should be read against the corresponding initial value rather than against an ideal of $\tau{=}1$. ")
    lines.append(r"DRL values are means over 5 seeds.}")
    lines.append(r"\label{tab:compactness}")
    lines.append(r"\begin{tabular}{lrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{Initial state} & \multicolumn{5}{c}{After DRL optimization (mean over 5 seeds)} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-10}")
    lines.append(r"Township & Init.\ raw & Init.\ IPQ$\geq$0.10 & Init.\ IPQ$\geq$0.20 & Init.\ IPQ$\geq$0.30 & "
                 r"Final raw & Final IPQ$\geq$0.10 & Final IPQ$\geq$0.20 & Final IPQ$\geq$0.30 & $\Delta$ raw \\")
    lines.append(r"\midrule")
    for _, row in summary.iterrows():
        lab = row["township_label"]
        init = init_baimu[lab]
        # Initial qualified counts are deterministic; pull from any seed.
        seed0 = df[df["township_label"] == lab].iloc[0]
        init_010 = int(seed0["init_ipq_ge_0.10"])
        init_020 = int(seed0["init_ipq_ge_0.20"])
        init_030 = int(seed0["init_ipq_ge_0.30"])
        raw_m = row["raw_count_mean"]
        d_raw = raw_m - init
        f010 = row["ipq010_mean"]
        f020 = row["ipq020_mean"]
        f030 = row["ipq030_mean"]
        lines.append(
            f"{lab} & {init} & {init_010} & {init_020} & {init_030} & "
            f"{raw_m:.1f} & {f010:.1f} & {f020:.1f} & {f030:.1f} & {d_raw:+.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {OUT_TEX}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
