# Dataset and Artifact Boundary

## Included Public Artifacts

The repository includes non-raw, derived artifacts needed to validate the
reported Paper3 results:

- block compositions, block features, and parcel-to-block mapping CSV files;
- five-seed training logs, trained MaskablePPO model zip files, and evaluation
  JSON files for the final block-level experiments;
- block construction audit outputs and maps;
- derived analysis outputs for reward-greedy, compactness, kappa ablation, and
  trajectory analyses;
- manuscript figure PNGs and LaTeX table fragments;
- sanitized Colab provenance notebooks and small timing metadata recovered from
  Google Drive.

## Excluded Restricted Inputs

The following raw or intermediate geospatial files are intentionally excluded:

- Third National Land Survey parcel geometry and attribute databases;
- `.gpkg`, `.gdb`, `.shp`, `.dbf`, `.shx`, and related vector sidecar files;
- DEM rasters and raster-derived arrays such as `.tif`, `.npy`, and `.npz`;
- private Colab packages containing restricted data, including
  `paper3_colab.zip`.

These files are not required to verify the included tables and figures, but
they are required for full raw-data preprocessing and exact retraining from
source parcels.

## Environment Variables for Controlled-Access Reruns

If you have authorized access to the raw data, point the code to local copies:

```bash
PAPER3_RESTRICTED_DATA_DIR=/path/to/restricted_data
PAPER3_DLTB_PATH=/path/to/DLTB_with_slope.gpkg
PAPER3_TNLS_GDB_PATH=/path/to/GDB.gdb
PAPER3_XIANGZHEN_PATH=/path/to/xiangzhen.shp
PAPER3_DEM_INTERMEDIATE_DIR=/path/to/dem_intermediate
```

On Windows PowerShell:

```powershell
$env:PAPER3_DLTB_PATH = "D:\path\to\DLTB_with_slope.gpkg"
```

All default path definitions are centralized in `src/paper3_paths.py`.

## Final Township Result Sets

- `township_500227109`: A-small final five-seed result set.
- `township_500227108_v2`: B-medium final five-seed result set used in the
  manuscript after reward rebalancing.
- `township_500227105`: C-large final five-seed result set.

The older non-v2 500227108 Google Drive run is retained only as provenance in
`archives/` and `results/google_drive_artifacts/`.
