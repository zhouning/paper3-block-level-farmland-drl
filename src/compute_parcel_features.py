"""Compute multi-source spatial features for real cadastral parcels.

Extracts additional features from DEM (aspect, elevation) and geometry
(shape index) using the same STRtree zonal statistics approach as the
slope pipeline. Outputs per-township feature tables for DRL training.

Features computed:
  - slope_mean, slope_max: already in GPKG (from dem_slope_zonal.py)
  - elevation_mean: mean DEM elevation per parcel
  - aspect_sin, aspect_cos: circular aspect encoded as sin/cos
  - shape_index: perimeter / (2 * sqrt(pi * area))

Usage:
    python compute_parcel_features.py                      # all townships
    python compute_parcel_features.py --township 500227109  # single township
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.ndimage import map_coordinates
from shapely import points as make_points
from shapely import STRtree
from pyproj import Geod

from township_utils import (
    load_county, load_township, classify_land_use, get_swappable_indices,
    TARGET_TOWNSHIPS, DEM_NPY_PATH, SLOPE_NPY_PATH,
    DEM_ORIGIN_LON, DEM_ORIGIN_LAT, DEM_PIXEL_SIZE,
)
from paper3_paths import DEM_INTERMEDIATE_DIR, PARCEL_FEATURES_DIR

OUTPUT_DIR = PARCEL_FEATURES_DIR
INTERMEDIATE_DIR = DEM_INTERMEDIATE_DIR


# ---------------------------------------------------------------------------
# Raster computation
# ---------------------------------------------------------------------------

def compute_aspect_raster(dem):
    """Compute aspect in degrees [0, 360) from DEM.

    Uses numpy gradient to compute dz/dx and dz/dy, then aspect as
    atan2(-dz_dy, -dz_dx) following standard GIS convention:
    0=North, 90=East, 180=South, 270=West.
    """
    # Compute pixel sizes at center latitude (same as slope pipeline)
    lat_center = DEM_ORIGIN_LAT - dem.shape[0] / 2 * DEM_PIXEL_SIZE
    geod = Geod(ellps='WGS84')
    ps = DEM_PIXEL_SIZE

    _, _, dy_m = geod.inv(
        DEM_ORIGIN_LON, lat_center - ps / 2,
        DEM_ORIGIN_LON, lat_center + ps / 2
    )
    _, _, dx_m = geod.inv(
        DEM_ORIGIN_LON - ps / 2, lat_center,
        DEM_ORIGIN_LON + ps / 2, lat_center
    )

    dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)

    # Aspect: 0=North, 90=East, 180=South, 270=West
    aspect_rad = np.arctan2(-dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad) % 360
    aspect_deg = aspect_deg.astype(np.float32)

    return aspect_deg


def zonal_stats_strtree(raster, parcels_gdf, stat_names=('mean',)):
    """Generic zonal statistics using STRtree pixel-parcel matching.

    Args:
        raster: 2D numpy array (n_rows, n_cols)
        parcels_gdf: GeoDataFrame in WGS84 CRS
        stat_names: tuple of aggregation functions (e.g., ('mean', 'max'))

    Returns:
        DataFrame with columns '{stat}' for each stat_name, indexed by parcel idx
    """
    n_parcels = len(parcels_gdf)

    # Build pixel grid covering study area bounds
    lon_min, lat_min, lon_max, lat_max = parcels_gdf.total_bounds

    col_min = max(0, int((lon_min - DEM_ORIGIN_LON) / DEM_PIXEL_SIZE) - 1)
    col_max = min(raster.shape[1] - 1, int((lon_max - DEM_ORIGIN_LON) / DEM_PIXEL_SIZE) + 1)
    row_min = max(0, int((DEM_ORIGIN_LAT - lat_max) / DEM_PIXEL_SIZE) - 1)
    row_max = min(raster.shape[0] - 1, int((DEM_ORIGIN_LAT - lat_min) / DEM_PIXEL_SIZE) + 1)

    cols = np.arange(col_min, col_max + 1)
    rows = np.arange(row_min, row_max + 1)

    col_grid, row_grid = np.meshgrid(cols, rows)
    col_flat = col_grid.ravel()
    row_flat = row_grid.ravel()

    lon_centers = DEM_ORIGIN_LON + (col_flat + 0.5) * DEM_PIXEL_SIZE
    lat_centers = DEM_ORIGIN_LAT - (row_flat + 0.5) * DEM_PIXEL_SIZE

    coords = np.column_stack([lon_centers, lat_centers])
    pixel_points = make_points(coords)

    # STRtree query
    tree = STRtree(parcels_gdf.geometry.values)
    pixel_idx, parcel_idx = tree.query(pixel_points, predicate='intersects')

    # Get raster values for matched pixels
    matched_rows = row_flat[pixel_idx]
    matched_cols = col_flat[pixel_idx]
    values = raster[matched_rows, matched_cols]

    # Aggregate per parcel
    df = pd.DataFrame({'parcel_idx': parcel_idx, 'value': values})
    df = df.dropna(subset=['value'])

    agg_dict = {name: name for name in stat_names}
    stats = df.groupby('parcel_idx')['value'].agg(list(stat_names))

    # Centroid fallback for unmatched parcels
    matched_set = set(stats.index)
    unmatched = [i for i in range(n_parcels) if i not in matched_set]

    if unmatched:
        centroids = parcels_gdf.geometry.iloc[unmatched].centroid
        cx = centroids.x.values
        cy = centroids.y.values

        sample_col = (cx - DEM_ORIGIN_LON) / DEM_PIXEL_SIZE - 0.5
        sample_row = (DEM_ORIGIN_LAT - cy) / DEM_PIXEL_SIZE - 0.5

        sampled = map_coordinates(
            raster.astype(np.float64),
            [sample_row, sample_col],
            order=1, mode='nearest'
        )

        fallback_rows = []
        for i, idx in enumerate(unmatched):
            row = {name: sampled[i] for name in stat_names}
            row['parcel_idx'] = idx
            fallback_rows.append(row)

        fallback_df = pd.DataFrame(fallback_rows).set_index('parcel_idx')
        stats = pd.concat([stats, fallback_df]).sort_index()

    return stats


# ---------------------------------------------------------------------------
# Shape features
# ---------------------------------------------------------------------------

def compute_shape_index(gdf_proj):
    """Compute shape irregularity index for each parcel.

    shape_index = perimeter / (2 * sqrt(pi * area))
    Value = 1.0 for perfect circle, higher for more irregular shapes.

    Args:
        gdf_proj: GeoDataFrame in projected CRS (meters)
    """
    areas = gdf_proj.geometry.area.values
    perimeters = gdf_proj.geometry.length.values

    # Avoid division by zero for degenerate parcels
    areas = np.maximum(areas, 1e-6)
    shape_idx = perimeters / (2.0 * np.sqrt(np.pi * areas))
    return shape_idx.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_table(gdf, township_code=None):
    """Build complete feature table for a set of parcels.

    Returns DataFrame with columns: BSM, DLBM, type_code, slope_mean, slope_max,
    elevation_mean, aspect_sin, aspect_cos, shape_index, TBMJ
    """
    n = len(gdf)
    print(f"  Computing features for {n:,} parcels...")

    # Ensure classification
    if 'type_code' not in gdf.columns:
        classify_land_use(gdf)

    # Start building feature table
    features = pd.DataFrame(index=range(n))
    features['BSM'] = gdf['BSM'].values
    features['DLBM'] = gdf['DLBM'].values
    features['type_code'] = gdf['type_code'].values
    features['TBMJ'] = gdf['TBMJ'].values
    features['slope_mean'] = gdf['slope_mean'].values
    features['slope_max'] = gdf['slope_max'].values

    # --- Elevation ---
    print("    Computing elevation zonal stats...")
    t0 = time.time()
    dem = np.load(DEM_NPY_PATH)
    elev_stats = zonal_stats_strtree(dem, gdf, stat_names=('mean',))
    features['elevation_mean'] = elev_stats['mean'].values
    print(f"      Elevation: mean={features['elevation_mean'].mean():.1f}m, "
          f"range=[{features['elevation_mean'].min():.0f}, {features['elevation_mean'].max():.0f}]m "
          f"({time.time()-t0:.1f}s)")

    # --- Aspect ---
    print("    Computing aspect...")
    t0 = time.time()

    # Load or compute aspect raster
    aspect_path = INTERMEDIATE_DIR / 'aspect_degrees.npy'
    if aspect_path.exists():
        aspect_deg = np.load(aspect_path)
        print(f"      Loaded cached aspect raster")
    else:
        aspect_deg = compute_aspect_raster(dem)
        np.save(aspect_path, aspect_deg)
        print(f"      Saved aspect raster: {aspect_path}")

    aspect_stats = zonal_stats_strtree(aspect_deg, gdf, stat_names=('mean',))
    aspect_mean = aspect_stats['mean'].values

    # Encode circular aspect as sin/cos
    aspect_rad = np.radians(aspect_mean)
    features['aspect_sin'] = np.sin(aspect_rad).astype(np.float32)
    features['aspect_cos'] = np.cos(aspect_rad).astype(np.float32)
    print(f"      Aspect sin/cos computed ({time.time()-t0:.1f}s)")

    del dem, aspect_deg  # free memory

    # --- Shape Index ---
    print("    Computing shape index...")
    t0 = time.time()
    gdf_proj = gdf.to_crs(epsg=4523)  # project to meters
    features['shape_index'] = compute_shape_index(gdf_proj)
    print(f"      Shape index: mean={features['shape_index'].mean():.2f}, "
          f"range=[{features['shape_index'].min():.2f}, {features['shape_index'].max():.2f}] "
          f"({time.time()-t0:.1f}s)")

    # --- Normalization stats ---
    norm_stats = {}
    for col in ['slope_mean', 'slope_max', 'elevation_mean', 'aspect_sin',
                'aspect_cos', 'shape_index', 'TBMJ']:
        vals = features[col].values.astype(float)
        norm_stats[col] = {
            'min': float(np.nanmin(vals)),
            'max': float(np.nanmax(vals)),
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
        }

    # Check for NaN
    n_nan = features.isna().sum().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values found. Filling with column means.")
        features = features.fillna(features.mean(numeric_only=True))

    return features, norm_stats


def main():
    parser = argparse.ArgumentParser(description='Compute parcel features')
    parser.add_argument('--township', type=str, default=None,
                        help='Single township code (default: all targets)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_start = time.time()

    if args.township:
        codes = [args.township]
    else:
        codes = list(TARGET_TOWNSHIPS.keys())

    all_norm_stats = {}

    for code in codes:
        info = TARGET_TOWNSHIPS.get(code, {})
        name = info.get('name', code)
        print(f"\n{'='*60}")
        print(f"Processing: {name} ({code})")
        print(f"{'='*60}")

        gdf = load_township(code)
        classify_land_use(gdf)

        features, norm_stats = build_feature_table(gdf, township_code=code)
        all_norm_stats[code] = norm_stats

        # Save township features
        csv_path = OUTPUT_DIR / f'township_{code}_features.csv'
        features.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  Saved: {csv_path} ({csv_path.stat().st_size/1024:.0f} KB)")

        # Print summary
        print(f"\n  Feature summary ({len(features):,} parcels):")
        for col in ['slope_mean', 'elevation_mean', 'aspect_sin', 'aspect_cos', 'shape_index']:
            vals = features[col].values
            print(f"    {col:20s}: mean={vals.mean():8.3f}, "
                  f"std={vals.std():8.3f}, "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]")

        # Land use breakdown
        for cat_code, cat_name in [(1, 'Farmland'), (2, 'Forest')]:
            mask = features['type_code'] == cat_code
            if mask.any():
                sl = features.loc[mask, 'slope_mean'].mean()
                el = features.loc[mask, 'elevation_mean'].mean()
                si = features.loc[mask, 'shape_index'].mean()
                print(f"    {cat_name:10s}: slope={sl:.2f}, elev={el:.0f}m, shape_idx={si:.2f}")

    # Save normalization stats
    stats_path = OUTPUT_DIR / 'feature_summary.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_norm_stats, f, indent=2)
    print(f"\nNormalization stats: {stats_path}")

    dt = time.time() - t_start
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == '__main__':
    main()
