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
python scripts/analysis/plot_pareto_all.py
python scripts/analysis/plot_training_curves.py
python scripts/analysis/plot_training_curves_108.py
python scripts/analysis/plot_training_curves_105.py
python scripts/analysis/plot_block_construction.py
python scripts/analysis/plot_framework_diagram.py
```

Derived analyses that instantiate the real block environment require the
restricted parcel geometry unless their outputs are already present in
`results/derived_analyses/`.

The area-balance audit added for the revised manuscript replays final
configurations against the controlled-access parcel geometry:

```bash
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg python scripts/analysis/paper3_area_drift_audit.py
```

It writes `results/derived_analyses/paper3_area_drift_results.json` and
`results/tables/paper3_area_drift_table.tex`.

## 3. Re-run Training With Authorized Raw Data

Full raw-data retraining requires local access to the restricted TNLS parcel
data with slope attributes. After setting the environment variables described
in `DATASET.md`, run:

```bash
python scripts/training/colab_train_all.py --township 500227109
python scripts/training/colab_retrain_108.py
python scripts/training/colab_train_all.py --township 500227105
```

The final B-medium results are written to
`results/blocks/township_500227108_v2`.

## 4. Reproduce Manuscript Package

The anonymous CEUS package is stored under `submission/ceus_anonymous/`.
Editable LaTeX source is also stored in `manuscript/latex_source/`. The source
archive in the submission folder is provided for journal upload if editable
source is requested.

## 5. Expected Limitations

- Raw parcel geometries and DEM rasters are not in the public repository.
- Public figure/table regeneration works from included derived artifacts.
- Exact full retraining from source parcels requires controlled raw-data access.
- For CEUS double-blind review, use an anonymous mirror rather than this public
  GitHub repository.
