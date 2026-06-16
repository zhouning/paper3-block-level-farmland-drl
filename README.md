# Paper 3: Block-Level Farmland Consolidation Scenario Screening

This repository supports the manuscript:

**Policy-Relevant Block Abstraction for Sequential Farmland Consolidation Scenario Screening**

The code and derived artifacts implement a block-level decision-support
framework for early-stage farmland consolidation scenario screening. The
manuscript is prepared for **Land Use Policy** and frames the method as a
policy-facing tool for comparing constrained consolidation scenarios, not as a
general claim that reinforcement learning dominates heuristic planning.

## Double-Blind Review

Use the anonymous review mirror in submissions and reviewer-facing documents:

`https://anonymous.4open.science/r/block-level-farmland-drl-8456/`

The corresponding public GitHub repository is:

`https://github.com/zhouning/paper3-block-level-farmland-drl`

Do not use the public GitHub URL in the double-anonymized manuscript or in
reviewer-facing fields before the review process is complete.

## Contents

- `src/`: block-level environment, baselines, parcel/block scoring helpers, and
  training/evaluation entry points.
- `scripts/analysis/`: figure, table, reward-greedy, limited-lookahead,
  compactness, area-balance, area-tolerance, trajectory, and kappa-ablation
  scripts.
- `scripts/training/`: Colab/A100 training scripts for the three townships.
- `results/blocks/`: five-seed block-level DRL outputs for the three study
  townships.
- `results/derived_analyses/` and `results/tables/`: downstream analysis JSON
  files and LaTeX table fragments used by the manuscript.
- `figures/`: high-resolution manuscript figures regenerated at 600 dpi.
- `manuscript/lup_anonymous/`: current anonymous Land Use Policy manuscript
  source and PDF.
- `manuscript/latex_source/`: editable anonymous LaTeX source package.
- `submission/lup_anonymous/`: anonymized LUP-oriented submission package for
  code-review mirroring, including highlights, figure captions, high-resolution
  figure files, declarations, and anonymous LaTeX source. Author-identifying
  title-page and cover-letter files are intentionally excluded from this
  repository copy.
- `notebooks/`: sanitized Colab provenance notebooks recovered from Drive.
- `archives/`: legacy non-restricted result archives retained for provenance.

Historical drafts and planning notes, if present under
`manuscript/development_archive/` or `docs/`, are retained only for provenance
and are not the active submission target.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS users can replace the activation command with:

```bash
source .venv/bin/activate
```

Regenerate key manuscript figures from included result artifacts:

```bash
python scripts/analysis/paper3_pareto_figures.py
python scripts/analysis/paper3_cross_township_bars.py
python scripts/analysis/plot_training_curves.py
python scripts/analysis/plot_training_curves_108.py
python scripts/analysis/plot_training_curves_105.py
python scripts/analysis/plot_framework_diagram.py
python scripts/analysis/plot_block_construction.py
```

Generated figures are written to `figures/`.

## Controlled-Data Analyses

The original Third National Land Survey parcel geometry is restricted and is
not redistributed here. Analyses that instantiate the real block environment
from parcel geometry require authorized local data access:

```bash
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_area_drift_audit.py
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

The aggregate outputs used in the manuscript are included under
`results/derived_analyses/` and `results/tables/`.

## Data Availability Boundary

Raw TNLS parcel geometries, geodatabases, shapefiles, and DEM-derived rasters
are excluded. The repository includes derived block-level metrics, training
logs, model artifacts, evaluation outputs, figure-generation scripts, and
synthetic tests sufficient to validate the reported tables and figures without
redistributing restricted raw parcels. See `DATASET.md` and
`restricted_data_manifest/TNLS_RESTRICTED_DATA.md`.

## Verification

Run the tests that do not require restricted data:

```bash
python -m unittest tests.test_paper3_lookahead_baseline -v
```

For a fuller reproduction checklist, see `REPRODUCIBILITY.md`.
