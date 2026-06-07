"""Repository-relative paths for the Paper3 reproducibility package.

Raw TNLS cadastral geometry is restricted and is not included in this
repository. Set the environment variables below when rerunning preprocessing
or training from controlled-access source data.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

RESULTS_DIR = Path(os.getenv("PAPER3_RESULTS_DIR", REPO_ROOT / "results"))
BLOCK_RESULTS_DIR = Path(os.getenv("PAPER3_BLOCK_RESULTS_DIR", RESULTS_DIR / "blocks"))
DERIVED_RESULTS_DIR = Path(
    os.getenv("PAPER3_DERIVED_RESULTS_DIR", RESULTS_DIR / "derived_analyses")
)
TABLES_DIR = Path(os.getenv("PAPER3_TABLES_DIR", RESULTS_DIR / "tables"))
FIGURES_DIR = Path(os.getenv("PAPER3_FIGURES_DIR", REPO_ROOT / "figures"))

RESTRICTED_DATA_DIR = Path(
    os.getenv("PAPER3_RESTRICTED_DATA_DIR", REPO_ROOT / "restricted_data")
)
DEM_ANALYSIS_DIR = Path(
    os.getenv("PAPER3_DEM_ANALYSIS_DIR", RESTRICTED_DATA_DIR / "dem_slope_analysis")
)
DEM_INTERMEDIATE_DIR = Path(
    os.getenv("PAPER3_DEM_INTERMEDIATE_DIR", DEM_ANALYSIS_DIR / "intermediate")
)
DEM_OUTPUT_DIR = Path(os.getenv("PAPER3_DEM_OUTPUT_DIR", DEM_ANALYSIS_DIR / "output"))

DLTB_PATH = Path(os.getenv("PAPER3_DLTB_PATH", DEM_OUTPUT_DIR / "DLTB_with_slope.gpkg"))
GDB_PATH = Path(os.getenv("PAPER3_TNLS_GDB_PATH", RESTRICTED_DATA_DIR / "GDB.gdb"))
XIANGZHEN_PATH = Path(os.getenv("PAPER3_XIANGZHEN_PATH", RESTRICTED_DATA_DIR / "xiangzhen.shp"))

PARCEL_FEATURES_DIR = Path(
    os.getenv("PAPER3_PARCEL_FEATURES_DIR", RESULTS_DIR / "parcel_features")
)
ADJACENCY_DIR = Path(os.getenv("PAPER3_ADJACENCY_DIR", RESULTS_DIR / "adjacency_cache"))

DEM_NPY_PATH = Path(os.getenv("PAPER3_DEM_NPY_PATH", DEM_INTERMEDIATE_DIR / "dem_full_tile.npy"))
SLOPE_NPY_PATH = Path(
    os.getenv("PAPER3_SLOPE_NPY_PATH", DEM_INTERMEDIATE_DIR / "slope_degrees.npy")
)
DEM_TIF_PATH = Path(
    os.getenv(
        "PAPER3_DEM_TIF_PATH",
        DEM_INTERMEDIATE_DIR / "Copernicus_DSM_COG_10_N29_00_E106_00_DEM.tif",
    )
)
