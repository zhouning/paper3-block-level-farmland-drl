# Paper3 Block-Level Farmland DRL

Repository for Paper3:

**From Parcels to Blocks: Rescaling Deep Reinforcement Learning for Farmland
Consolidation Planning through Spatial Abstraction**

This repository collects the Paper3 code, anonymous CEUS manuscript files,
figures, derived block-level results, trained model artifacts, and
reproducibility documentation.

## Double-Blind Review Note

This public GitHub repository is useful for long-term project management and
post-review reproducibility. For CEUS double-blind review, copy the same
contents to an anonymous review repository and use that anonymous URL in the
submission system.

## Contents

- `src/`: block-level environment, baselines, parcel policy, preprocessing
  helpers, and training/evaluation entry points.
- `scripts/analysis/`: figure, table, ablation, reward-greedy, compactness,
  area-balance, and trajectory analysis scripts.
- `scripts/training/`: Colab/A100 training scripts for the three townships.
- `results/blocks/`: five-seed block-level DRL outputs for townships 109, 108
  v2, and 105.
- `results/derived_analyses/` and `results/tables/`: downstream analysis JSON
  and LaTeX table fragments.
- `figures/`: figure files used by the manuscript.
- `manuscript/`: anonymous CEUS manuscript and editable LaTeX source.
- `submission/ceus_anonymous/`: anonymous CEUS review package.
- `notebooks/`: sanitized Google Colab run notebooks recovered from Drive.
- `archives/`: legacy Google Drive result archives retained for provenance.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Regenerate selected manuscript figures from included result artifacts:

```bash
python scripts/analysis/paper3_pareto_figures.py
python scripts/analysis/plot_training_curves.py
python scripts/analysis/plot_training_curves_108.py
python scripts/analysis/plot_training_curves_105.py
```

Generated figures are written to `figures/`.

The farmland-area balance audit reported in the revised manuscript requires
controlled-access parcel geometry:

```bash
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_area_drift_audit.py
```

Its aggregate outputs are included as
`results/derived_analyses/paper3_area_drift_results.json` and
`results/tables/paper3_area_drift_table.tex`.

## Data Availability Boundary

The original Third National Land Survey parcel geometry, geodatabases,
shapefiles, and DEM-derived rasters are restricted and are not redistributed
here. The repository includes derived block-level metrics, training logs,
trained models, evaluation outputs, and figure-generation scripts sufficient to
validate the reported tables and figures. Full retraining from raw parcels
requires controlled access to the original geospatial data. See `DATASET.md`
and `restricted_data_manifest/TNLS_RESTRICTED_DATA.md`.

## Google Drive Recovery

Additional Paper3 files were checked in a local Google Drive mirror.
Non-restricted artifacts recovered from that location are included under
`notebooks/` and `results/google_drive_artifacts/`. The private
`paper3_colab.zip` archive is not included because it contained restricted raw
geospatial data.
