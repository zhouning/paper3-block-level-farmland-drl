# macOS Lookahead Experiment Checklist

This checklist is for the Paper3 robustness run on a macOS machine with access
to the restricted parcel geometry. It does not retrain the DRL models.

## Goal

Run the finite-depth lookahead baseline for Township B to test whether the
reported DRL advantage can be reproduced by deterministic multi-step search
under the same reward and within-block transition model as Reward-Greedy.

The primary run is exact depth-2 search:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
```

`--beam-width 0` means that all valid first-step and second-step block choices
are evaluated. This is the required run for the manuscript audit.

## 1. Pull the Updated Repository

```bash
git clone https://github.com/zhouning/paper3-block-level-farmland-drl.git
cd paper3-block-level-farmland-drl
```

If the repository already exists on the macOS machine:

```bash
cd paper3-block-level-farmland-drl
git pull origin main
```

## 2. Create a Python Environment

Use Python 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `geopandas`, `rasterio`, or `pyproj` fails to install from wheels on the Mac,
install the geospatial system libraries first and rerun `pip install`:

```bash
brew install gdal proj geos
pip install -r requirements.txt
```

## 3. Point to the Restricted Data

The script requires the controlled-access parcel geometry with slope
attributes:

```bash
export PAPER3_DLTB_PATH=/absolute/path/to/DLTB_with_slope.gpkg
```

Before running the experiment, confirm that the path is visible:

```bash
python -c "import os, pathlib; p=pathlib.Path(os.environ['PAPER3_DLTB_PATH']); print(p); print(p.exists())"
```

The second line must print `True`.

## 4. Run the Required Experiment

Run Township B exact depth-2 first:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
```

Expected outputs:

```text
results/derived_analyses/paper3_lookahead_d2_ball_results.json
results/tables/paper3_lookahead_d2_ball_table_fragment.tex
```

Keep the terminal text printed under `Interpretation guidance:`. That message
determines how the manuscript should frame the Township-B scheduler result.

## 5. Optional Sensitivity Run

Only run this if the exact depth-2 result is close to zero or otherwise
ambiguous:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

Expected optional outputs:

```text
results/derived_analyses/paper3_lookahead_d3_b8_b_depth3_sensitivity_results.json
results/tables/paper3_lookahead_d3_b8_b_depth3_sensitivity_table_fragment.tex
```

Depth-3 with `--beam-width 8` is a pruned sensitivity analysis, not an exact
global-search result.

## 6. Send Back

Send these files and the terminal `Interpretation guidance:` block:

```text
results/derived_analyses/paper3_lookahead_d2_ball_results.json
results/tables/paper3_lookahead_d2_ball_table_fragment.tex
```

If the optional depth-3 run was used, send its JSON and TEX files too.

## 7. Manuscript Interpretation

- If exact depth-2 Township B reaches positive `baimu_fang` area close to the
  DRL result, revise the manuscript to frame the contribution as multi-step
  planning rather than RL-specific learning.
- If exact depth-2 Township B remains negative while DRL is positive, the
  revised claim is stronger: one-step and tested two-step deterministic
  planning do not reproduce the learned scheduler's Township-B behavior.
- If exact depth-2 is near zero, treat it as ambiguous and use the depth-3
  sensitivity result only as supporting evidence.
