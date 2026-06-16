# Reproducibility Guide

## 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS users can replace the activation command with:

```bash
source .venv/bin/activate
```

## 2. Validate Result-Level Reproducibility

The following commands use included result artifacts and regenerate key figures
or derived tables without raw geospatial data:

```bash
python scripts/analysis/paper3_pareto_figures.py
python scripts/analysis/paper3_cross_township_bars.py
python scripts/analysis/plot_training_curves.py
python scripts/analysis/plot_training_curves_108.py
python scripts/analysis/plot_training_curves_105.py
python scripts/analysis/plot_block_construction.py
python scripts/analysis/plot_framework_diagram.py
```

Generated figures are written to `figures/`. The current manuscript figures
are exported as high-resolution PNG files suitable for journal review.

Derived analyses that instantiate the real block environment require
restricted parcel geometry unless their aggregate outputs are already present
in `results/derived_analyses/`.

The area-balance audit replays final configurations against controlled-access
parcel geometry:

```bash
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_area_drift_audit.py
```

It writes `results/derived_analyses/paper3_area_drift_results.json` and
`results/tables/paper3_area_drift_table.tex`.

The limited-lookahead robustness baseline also instantiates the real block
environment and therefore requires the same controlled-access parcel geometry.
Use `--beam-width 0` for the primary depth-2 Township-B check. This disables
immediate-reward pruning and evaluates all valid second-step actions.

```bash
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

Configuration-specific outputs include
`results/derived_analyses/paper3_lookahead_d2_ball_results.json` and
`results/tables/paper3_lookahead_d2_ball_table_fragment.tex`. The controlled
Township-B outputs reported in the Land Use Policy manuscript are included.
Use Township B first when rerunning because it is the critical case for
separating learned scheduling from one-step reward-greedy planning.

The lookahead action-selection logic can be tested without restricted data:

```bash
python -m unittest tests.test_paper3_lookahead_baseline -v
```

For the exact macOS run sequence on a machine with restricted parcel geometry,
see `docs/MAC_LOOKAHEAD_EXPERIMENT.md`.

## 3. Re-run Training With Authorized Raw Data

Full raw-data retraining requires local access to restricted TNLS parcel data
with slope attributes. After setting the environment variables described in
`DATASET.md`, run:

```bash
python scripts/training/colab_train_all.py --township 500227109
python scripts/training/colab_retrain_108.py
python scripts/training/colab_train_all.py --township 500227105
```

The final B-medium results are written to
`results/blocks/township_500227108_v2`.

## 4. Reproduce the Manuscript Package

The active anonymous Land Use Policy manuscript is stored under
`manuscript/lup_anonymous/`. Editable LaTeX source is stored in
`manuscript/latex_source/`.

The anonymized LUP-oriented submission package for code-review mirroring is
stored under `submission/lup_anonymous/`. Author-identifying title-page and
cover-letter files are intentionally excluded from this repository copy.

For double-anonymized review, use:

`https://anonymous.4open.science/r/block-level-farmland-drl-8456/`

The public GitHub repository is:

`https://github.com/zhouning/paper3-block-level-farmland-drl`

Do not use the public GitHub URL in reviewer-facing manuscript text before the
review process is complete.

## 5. Expected Limitations

- Raw parcel geometries and DEM rasters are not in the public repository.
- Public figure/table regeneration works from included derived artifacts.
- Exact full retraining from source parcels requires controlled raw-data access.
- Limited-lookahead baseline reruns require controlled parcel geometry; the
  included Township-B outputs document the controlled-data runs used in the
  LUP manuscript.
