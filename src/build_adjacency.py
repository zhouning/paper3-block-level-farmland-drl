"""Build topology-based spatial adjacency graphs for real cadastral parcels.

Uses libpysal Queen contiguity (shared edges + vertices) to build adjacency
lists for each target township. Output format matches land_use_env_v7.py:
list[np.ndarray] where adjacency[i] = array of neighbor indices.

Fallback: shapely STRtree + touches predicate if libpysal unavailable.

Usage:
    python build_adjacency.py                     # all target townships
    python build_adjacency.py --township 500227109  # single township
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import geopandas as gpd

from township_utils import (
    load_township, classify_land_use, print_township_summary,
    TARGET_TOWNSHIPS, GPKG_PATH,
)
from paper3_paths import ADJACENCY_DIR

OUTPUT_DIR = str(ADJACENCY_DIR)


def build_adjacency_libpysal(gdf):
    """Build adjacency using libpysal Queen contiguity weights.

    Queen contiguity detects shared edges AND vertices (more inclusive than
    Rook which only detects shared edges).

    Returns:
        list[np.ndarray]: adjacency[i] = sorted array of neighbor indices for parcel i
    """
    from libpysal.weights import Queen

    t0 = time.time()
    w = Queen.from_dataframe(gdf, use_index=False)
    dt = time.time() - t0
    print(f"  libpysal Queen contiguity built in {dt:.1f}s")

    n = len(gdf)
    adjacency = []
    for i in range(n):
        nbrs = w.neighbors.get(i, [])
        adjacency.append(np.array(sorted(nbrs), dtype=np.intp))

    return adjacency


def build_adjacency_strtree(gdf):
    """Fallback: STRtree + touches predicate for adjacency.

    Slower than libpysal but requires only shapely.
    """
    from shapely import STRtree

    t0 = time.time()
    tree = STRtree(gdf.geometry.values)
    n = len(gdf)
    adjacency = []

    for i in range(n):
        geom = gdf.geometry.iloc[i]
        # query for candidates that touch (shared boundary)
        candidates = tree.query(geom, predicate='touches')
        # Also check intersects but not equals (for shared-vertex cases)
        if len(candidates) == 0:
            candidates_int = tree.query(geom, predicate='intersects')
            candidates = np.array([c for c in candidates_int if c != i], dtype=np.intp)
        else:
            candidates = np.array([c for c in candidates if c != i], dtype=np.intp)
        adjacency.append(np.sort(candidates))

        if (i + 1) % 2000 == 0:
            print(f"    STRtree adjacency: {i+1}/{n} parcels...")

    dt = time.time() - t0
    print(f"  STRtree adjacency built in {dt:.1f}s")
    return adjacency


def save_adjacency(adjacency, output_path, metadata=None):
    """Save adjacency as compressed numpy archive (.npz).

    Format: adj_0, adj_1, ..., adj_N-1 arrays + metadata JSON string.
    """
    save_dict = {}
    for i, arr in enumerate(adjacency):
        save_dict[f'adj_{i}'] = arr

    if metadata:
        save_dict['metadata'] = np.array([json.dumps(metadata)])

    np.savez_compressed(output_path, **save_dict)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def load_adjacency(npz_path):
    """Load adjacency from .npz file. Returns list[np.ndarray]."""
    data = np.load(npz_path, allow_pickle=True)

    # Determine number of parcels from keys
    adj_keys = sorted([k for k in data.files if k.startswith('adj_')],
                      key=lambda x: int(x.split('_')[1]))
    adjacency = [data[k] for k in adj_keys]
    return adjacency


def verify_adjacency(gdf, adjacency, n_samples=50):
    """Spot-check adjacency against actual geometry.

    Returns dict with verification results.
    """
    import random

    n = len(gdf)
    assert len(adjacency) == n, f"Adjacency length {len(adjacency)} != {n} parcels"

    # Basic statistics
    nbr_counts = np.array([len(a) for a in adjacency])
    n_isolated = int((nbr_counts == 0).sum())

    results = {
        'n_parcels': n,
        'avg_neighbors': float(nbr_counts.mean()),
        'min_neighbors': int(nbr_counts.min()),
        'max_neighbors': int(nbr_counts.max()),
        'median_neighbors': float(np.median(nbr_counts)),
        'n_isolated': n_isolated,
        'pct_isolated': float(100 * n_isolated / n),
    }

    # Symmetry check
    asymmetric_count = 0
    for i in range(n):
        for j in adjacency[i]:
            if i not in adjacency[j]:
                asymmetric_count += 1
    results['asymmetric_pairs'] = asymmetric_count

    # Spot-check: verify neighbors actually touch/intersect
    sample_ids = random.sample(range(n), min(n_samples, n))
    false_positives = 0
    false_negatives_checked = 0
    total_checked = 0

    for i in sample_ids:
        geom_i = gdf.geometry.iloc[i]
        for j in adjacency[i]:
            total_checked += 1
            if not geom_i.intersects(gdf.geometry.iloc[j]):
                false_positives += 1

    results['spot_check_samples'] = len(sample_ids)
    results['spot_check_total_pairs'] = total_checked
    results['spot_check_false_positives'] = false_positives
    results['verification_passed'] = (
        asymmetric_count == 0 and false_positives == 0
    )

    return results


def process_township(township_code, output_dir=OUTPUT_DIR, use_strtree_fallback=False):
    """Build adjacency for a single township. Save results."""
    info = TARGET_TOWNSHIPS.get(township_code, {})
    name = info.get('name', township_code)
    print(f"\n{'='*60}")
    print(f"Building adjacency: {name} ({township_code})")
    print(f"{'='*60}")

    # Load and classify
    t0 = time.time()
    gdf = load_township(township_code, reproject=True)
    classify_land_use(gdf)
    print_township_summary(gdf, township_code)

    # Build adjacency
    if use_strtree_fallback:
        adjacency = build_adjacency_strtree(gdf)
    else:
        try:
            adjacency = build_adjacency_libpysal(gdf)
        except Exception as e:
            print(f"  libpysal failed: {e}")
            print(f"  Falling back to STRtree...")
            adjacency = build_adjacency_strtree(gdf)

    # Verify
    print("\n  Verifying adjacency...")
    vresults = verify_adjacency(gdf, adjacency)
    print(f"    Avg neighbors: {vresults['avg_neighbors']:.1f}")
    print(f"    Min/Max neighbors: {vresults['min_neighbors']}/{vresults['max_neighbors']}")
    print(f"    Isolated parcels: {vresults['n_isolated']} ({vresults['pct_isolated']:.1f}%)")
    print(f"    Asymmetric pairs: {vresults['asymmetric_pairs']}")
    print(f"    Spot-check false positives: {vresults['spot_check_false_positives']}/{vresults['spot_check_total_pairs']}")
    print(f"    PASSED: {vresults['verification_passed']}")

    # Save adjacency
    adj_path = os.path.join(output_dir, f'township_{township_code}_adj.npz')
    metadata = {
        'township_code': township_code,
        'n_parcels': len(gdf),
        'method': 'strtree' if use_strtree_fallback else 'libpysal_queen',
        'verification': vresults,
    }
    save_adjacency(adjacency, adj_path, metadata)

    # Save township parcels subset as GPKG (with reset index)
    parcels_path = os.path.join(output_dir, f'township_{township_code}_parcels.gpkg')
    gdf_save = gdf.to_crs(epsg=4326)  # save in WGS84 for compatibility
    gdf_save.to_file(parcels_path, driver='GPKG')
    size_mb = os.path.getsize(parcels_path) / (1024 * 1024)
    print(f"  Saved: {parcels_path} ({size_mb:.1f} MB)")

    dt = time.time() - t0
    print(f"  Total time: {dt:.1f}s")

    return vresults


def main():
    parser = argparse.ArgumentParser(description='Build adjacency graphs for townships')
    parser.add_argument('--township', type=str, default=None,
                        help='Single township code (default: all target townships)')
    parser.add_argument('--strtree', action='store_true',
                        help='Force STRtree fallback instead of libpysal')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.township:
        codes = [args.township]
    else:
        codes = list(TARGET_TOWNSHIPS.keys())

    all_results = {}
    t_start = time.time()

    for code in codes:
        vresults = process_township(code, OUTPUT_DIR, use_strtree_fallback=args.strtree)
        all_results[code] = vresults

    # Save verification summary
    summary_path = os.path.join(OUTPUT_DIR, 'adjacency_verification.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nVerification summary: {summary_path}")

    dt_total = time.time() - t_start
    print(f"\nTotal time: {dt_total:.1f}s")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Township':<12} {'Parcels':>8} {'AvgNbr':>8} {'Isolated':>10} {'Passed':>8}")
    print(f"{'-'*60}")
    for code, r in all_results.items():
        print(f"{code:<12} {r['n_parcels']:>8} {r['avg_neighbors']:>8.1f} "
              f"{r['n_isolated']:>10} {'YES' if r['verification_passed'] else 'NO':>8}")


if __name__ == '__main__':
    main()
