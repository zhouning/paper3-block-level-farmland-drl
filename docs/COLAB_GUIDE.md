# Google Colab Reproducibility Guide

This guide documents the A100/Colab workflow used for the Paper3 block-level
DRL experiments. The public repository contains code, derived block features,
trained models, training logs, evaluation JSON files, figures, and manuscript
artifacts. Original cadastral geometry and DEM rasters are restricted and are
not included.

## Repository Setup

Clone the repository and install the Python dependencies:

```bash
git clone <REPOSITORY_URL>
cd paper3-block-level-farmland-drl
pip install -r requirements.txt
```

For Google Colab, install the same core dependencies:

```python
!pip install -q stable-baselines3==2.2.1 sb3-contrib==2.2.1 gymnasium==0.29.1
!pip install -q geopandas libpysal shimmy tqdm tensorboard
```

## Included Results

The repository already includes the derived block-level experiment results:

- `results/blocks/township_500227109`: A-small, five seeds.
- `results/blocks/township_500227108_v2`: B-medium, five seeds, rebalanced
  reward version used in the final manuscript.
- `results/blocks/township_500227105`: C-large, five seeds.
- `results/block_construction_audit`: block construction outputs and maps.
- `results/derived_analyses`: compactness, reward-greedy, kappa ablation, and
  trajectory analyses.

These artifacts allow reviewers to regenerate manuscript tables and figures
without access to the restricted raw geospatial layers.

## Re-running Training

From the repository root:

```bash
python scripts/training/colab_train_all.py --township 500227109
python scripts/training/colab_train_all.py --township 500227105
python scripts/training/colab_retrain_108.py
```

The 108 script writes to `results/blocks/township_500227108_v2`, matching the
version used in the manuscript tables. The default configuration trains five
seeds for 200,000 timesteps per township.

Estimated A100 runtime:

| Township | Blocks | Approximate runtime |
|---|---:|---:|
| 500227109 | 78 | 25 minutes |
| 500227108 v2 | 132 | 50 minutes |
| 500227105 | 338 | 2 hours |

## Regenerating Figures

```bash
python scripts/analysis/plot_training_curves.py
python scripts/analysis/plot_training_curves_108.py
python scripts/analysis/plot_training_curves_105.py
python scripts/analysis/paper3_pareto_figures.py
python scripts/analysis/plot_block_construction.py
python scripts/analysis/plot_framework_diagram.py
```

Generated figures are written to `figures/`.

## Limited-Lookahead Robustness Baseline

The manuscript identifies Township B as the critical case where DRL differs
from one-step Reward-Greedy planning. If controlled-access parcel geometry is
available on Colab, run the exact depth-2 check first:

```bash
export PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 2 --beam-width 0
```

The `--beam-width 0` setting evaluates all valid first- and second-step block
choices, avoiding immediate-reward beam pruning. If the depth-2 result is
ambiguous, run a depth-3 sensitivity check with pruning:

```bash
python scripts/analysis/paper3_lookahead_baseline.py --township B --depth 3 --beam-width 8 --suffix b_depth3_sensitivity
```

These runs write configuration-specific JSON and LaTeX fragments under
`results/derived_analyses/` and `results/tables/`.

## Restricted Raw Data

The public repository intentionally excludes original cadastral geometry,
geodatabase files, shapefiles, DEM rasters, and raster-derived NumPy arrays.
If controlled-access raw data are available, set these environment variables
before rerunning preprocessing:

```bash
export PAPER3_RESTRICTED_DATA_DIR=/path/to/restricted_data
export PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg
export PAPER3_TNLS_GDB_PATH=/path/to/GDB.gdb
export PAPER3_XIANGZHEN_PATH=/path/to/xiangzhen.shp
```

On Windows PowerShell, use `$env:PAPER3_DLTB_PATH = "..."` syntax.

## Original Colab Run Notebooks

Sanitized provenance notebooks from Google Drive are stored in `notebooks/`.
They have execution outputs and Colab authorship metadata removed. They refer
to the original private `paper3_colab.zip`, which is not public because it
contained restricted geospatial data. Use the scripts above for the public
reproducibility workflow.
