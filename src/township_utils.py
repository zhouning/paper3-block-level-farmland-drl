"""Shared utilities for township data extraction and land use classification.

Provides consistent data loading, DLBM-code-based classification, and
helper functions used across Tasks 1-4 of the real cadastral data pipeline.
"""

import numpy as np
import geopandas as gpd
from paper3_paths import (
    DLTB_PATH as DEFAULT_DLTB_PATH,
    DEM_NPY_PATH as DEFAULT_DEM_NPY_PATH,
    DEM_TIF_PATH as DEFAULT_DEM_TIF_PATH,
    SLOPE_NPY_PATH as DEFAULT_SLOPE_NPY_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DLBM land use codes (Third National Land Survey standard)
FARMLAND_CODES = {'011', '012', '013'}   # paddy, irrigated, dry land
FOREST_CODES = {'031', '032', '033'}     # forest, shrub, other forest
ORCHARD_CODES = {'021', '022', '023'}    # orchard, tea, rubber

# Land use type constants (matching PoC convention)
OTHER = 0
FARMLAND = 1
FOREST = 2

# Target townships for experiments
TARGET_TOWNSHIPS = {
    '500227109': {'name': 'Small township', 'expected_parcels': 3607},
    '500227108': {'name': 'Medium township', 'expected_parcels': 6274},
    '500227105': {'name': 'Large township', 'expected_parcels': 12950},
}

# Data paths. Raw/geometry files are restricted and must be supplied locally.
GPKG_PATH = str(DEFAULT_DLTB_PATH)
DEM_NPY_PATH = str(DEFAULT_DEM_NPY_PATH)
SLOPE_NPY_PATH = str(DEFAULT_SLOPE_NPY_PATH)
DEM_TIF_PATH = str(DEFAULT_DEM_TIF_PATH)

# DEM geotransform (verified in dem_slope_zonal.py)
DEM_ORIGIN_LON = 106.0
DEM_ORIGIN_LAT = 30.0
DEM_PIXEL_SIZE = 1.0 / 3600  # degrees

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def load_county(gpkg_path=GPKG_PATH):
    """Load the full county dataset from GPKG.

    Returns GeoDataFrame with CRS overridden to EPSG:4326 (verified CGCS2000).
    """
    gdf = gpd.read_file(gpkg_path, layer='DLTB')
    gdf = gdf.set_crs('EPSG:4326', allow_override=True)
    return gdf


def load_township(township_code, gpkg_path=GPKG_PATH, reproject=False):
    """Load parcels for a single township.

    Args:
        township_code: 9-digit township code (e.g., '500227109')
        gpkg_path: Path to GPKG with slope data
        reproject: If True, reproject to EPSG:4523 (CGCS2000 GK Zone 35, meters)

    Returns:
        GeoDataFrame with index reset to 0..N-1
    """
    gdf = load_county(gpkg_path)
    mask = gdf['QSDWDM'].str[:9] == township_code
    gdf_t = gdf[mask].copy()
    gdf_t = gdf_t.reset_index(drop=True)

    if reproject:
        gdf_t = gdf_t.to_crs(epsg=4523)

    return gdf_t


def classify_land_use(gdf):
    """Add classification columns based on DLBM codes.

    Adds:
        - 'category': str ('Farmland', 'Forest', 'Orchard', 'Other')
        - 'type_code': int (OTHER=0, FARMLAND=1, FOREST=2)

    Returns modified GeoDataFrame (in-place).
    """
    def _classify(dlbm):
        if dlbm in FARMLAND_CODES:
            return 'Farmland'
        elif dlbm in FOREST_CODES:
            return 'Forest'
        elif dlbm in ORCHARD_CODES:
            return 'Orchard'
        else:
            return 'Other'

    def _type_code(dlbm):
        if dlbm in FARMLAND_CODES:
            return FARMLAND
        elif dlbm in FOREST_CODES:
            return FOREST
        else:
            return OTHER

    gdf['category'] = gdf['DLBM'].apply(_classify)
    gdf['type_code'] = gdf['DLBM'].apply(_type_code).astype(np.int8)
    return gdf


def get_swappable_mask(gdf):
    """Return boolean mask of farmland + forest parcels (swappable set).

    Requires 'type_code' column (call classify_land_use first).
    """
    return (gdf['type_code'] == FARMLAND) | (gdf['type_code'] == FOREST)


def get_swappable_indices(gdf):
    """Return integer indices of swappable (farmland + forest) parcels.

    Requires 'type_code' column.
    """
    mask = get_swappable_mask(gdf)
    return np.where(mask)[0]


def print_township_summary(gdf, township_code):
    """Print formatted summary of township data."""
    classify_land_use(gdf) if 'type_code' not in gdf.columns else None

    n = len(gdf)
    n_farm = int((gdf['type_code'] == FARMLAND).sum())
    n_forest = int((gdf['type_code'] == FOREST).sum())
    n_swap = n_farm + n_forest

    print(f"  Township: {township_code}")
    print(f"  Total parcels: {n:,}")
    print(f"  Farmland: {n_farm:,} ({100*n_farm/n:.1f}%)")
    print(f"  Forest: {n_forest:,} ({100*n_forest/n:.1f}%)")
    print(f"  Swappable: {n_swap:,} ({100*n_swap/n:.1f}%)")

    if 'slope_mean' in gdf.columns:
        farm_slope = gdf.loc[gdf['type_code'] == FARMLAND, 'slope_mean']
        forest_slope = gdf.loc[gdf['type_code'] == FOREST, 'slope_mean']
        print(f"  Farmland avg slope: {farm_slope.mean():.2f} deg")
        print(f"  Forest avg slope: {forest_slope.mean():.2f} deg")

    if 'TBMJ' in gdf.columns:
        areas = gdf['TBMJ']
        print(f"  Area range: {areas.min():.0f} - {areas.max():.0f} m2")
        print(f"  Area median: {areas.median():.0f} m2, mean: {areas.mean():.0f} m2")


if __name__ == '__main__':
    print("=" * 60)
    print("Township Data Summary")
    print("=" * 60)

    for code, info in TARGET_TOWNSHIPS.items():
        print(f"\n--- {info['name']} ({code}) ---")
        gdf = load_township(code)
        classify_land_use(gdf)
        print_township_summary(gdf, code)
        print(f"  Expected: {info['expected_parcels']}, Actual: {len(gdf)}")
