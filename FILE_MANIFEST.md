# File Manifest

## Code

- `src/block_level_env.py`: block-level MDP environment.
- `src/baselines_block.py`: greedy, random, and round-robin baselines.
- `src/train_block.py`: MaskablePPO block-level training utilities.
- `src/parcel_scoring_policy.py`: custom parcel/block scoring policy.
- `src/block_definition.py` and `src/block_definition_all.py`: block
  construction pipeline.
- `src/paper3_paths.py`: repository-relative path configuration.
- `scripts/training/`: Colab/A100 training entry points.
- `scripts/preprocessing/`: DEM slope and parcel feature preprocessing.
- `scripts/analysis/`: figure, table, and robustness analysis scripts.

## Results

- `results/blocks/`: final five-seed township result folders.
- `results/block_construction_audit/`: construction audit outputs and maps.
- `results/derived_analyses/`: JSON outputs for derived analyses.
- `results/tables/`: LaTeX table fragments used by the manuscript.
- `results/google_drive_artifacts/`: small non-restricted artifacts recovered
  from Google Drive.

## Manuscript and Submission

- `manuscript/ceus_anonymous/`: latest anonymous CEUS manuscript source and PDF.
- `manuscript/latex_source/`: editable anonymous LaTeX source package.
- `manuscript/development_archive/`: historical Paper3 manuscript drafts.
- `submission/ceus_anonymous/`: anonymous reviewer-facing CEUS upload package.

## Figures and Notebooks

- `figures/`: manuscript figures and diagnostic figures.
- `notebooks/`: sanitized Colab provenance notebooks recovered from Drive.

## Archives

- `archives/paper3_105_results-20260309T133921Z-3-001.zip`: legacy 105 result
  archive from Google Drive.
- `archives/paper3_108_results-20260309T040107Z-3-001.zip`: legacy 108 result
  archive from Google Drive.

The private `paper3_colab.zip` archive is intentionally excluded because it
contained restricted raw geospatial data.
