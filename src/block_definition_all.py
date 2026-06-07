# -*- coding: utf-8 -*-
"""
Batch Block Definition for All Bishan Townships (Paper 4 prep).

Extends Paper 3's block_definition.py to all 13 Bishan townships.
Reuses the same pipeline: DLTB barriers -> connected components -> AgglomerativeClustering.

Usage:
    python block_definition_all.py                     # all 13 townships
    python block_definition_all.py --skip-existing     # skip already processed
    python block_definition_all.py --township 500227102  # single new township
    python block_definition_all.py --visualize         # also generate block maps
    python block_definition_all.py --summary           # just print summary of all townships
"""

import os
import sys
import json
import argparse
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from block_definition import (
    load_township, build_swappable_adjacency, find_connected_components,
    subdivide_large_components, compute_block_features,
    save_results, visualize_blocks, DLTB_PATH, OUTPUT_DIR, PROJ_CRS
)

# All 13 Bishan townships with their codes and labels
ALL_TOWNSHIPS = {
    '500227001': 'T01-Bishan (Street)',
    '500227002': 'T02-Qinggang (Street)',
    '500227100': 'T03-Hechuan',
    '500227101': 'T04-Laifeng',
    '500227102': 'T05-Guangpu',
    '500227103': 'T06-Daxing',
    '500227104': 'T07-Zhengxing',
    '500227105': 'T08-Dalukou (Large)',
    '500227106': 'T09-Hebian',
    '500227107': 'T10-Shihe',
    '500227108': 'T11-Baxian (Medium)',
    '500227109': 'T12-Jianlong (Small)',
    '500227200': 'T13-Qinglonghu',
}

# Paper 3's original 3 townships (already processed)
PAPER3_TOWNSHIPS = {'500227109', '500227108', '500227105'}


def has_existing_results(township_code):
    """Check if block definition results already exist for a township."""
    out_dir = os.path.join(OUTPUT_DIR, f'township_{township_code}')
    required = ['block_compositions.json', 'block_features.json', 'parcel_block_mapping.csv']
    return all(os.path.exists(os.path.join(out_dir, f)) for f in required)


def define_blocks_safe(township_code, label, min_parcels=3, min_area_ha=0.5, max_parcels=30):
    """Run block definition with error handling per township."""
    print(f"\n{'='*60}")
    print(f"  Block Definition: {label} ({township_code})")
    print(f"{'='*60}")

    t0 = time.time()

    try:
        # Step 1: Load and classify
        print("  Step 1: Loading parcels...")
        gdf = load_township(township_code)
        n_total = len(gdf)
        n_farm = (gdf['category'] == 'farmland').sum()
        n_forest = (gdf['category'] == 'forest').sum()
        n_barrier = (gdf['category'] == 'barrier').sum()
        n_other = (gdf['category'] == 'other').sum()
        print(f"    Total: {n_total}  |  Farmland: {n_farm}  |  Forest: {n_forest}  |  Barrier: {n_barrier}  |  Other: {n_other}")

        # Step 2: Extract swappable parcels
        gdf_swappable = gdf[gdf['category'].isin(['farmland', 'forest'])].copy()
        gdf_swappable = gdf_swappable.reset_index(drop=True)
        n_swap = len(gdf_swappable)
        print(f"  Step 2: {n_swap} swappable parcels")

        if n_swap < min_parcels:
            print(f"  SKIP: Too few swappable parcels ({n_swap} < {min_parcels})")
            return None

        # Step 3: Build adjacency
        print("  Step 3: Building adjacency...")
        adjacency = build_swappable_adjacency(gdf_swappable)

        # Step 4: Connected components
        print("  Step 4: Finding connected components...")
        components = find_connected_components(adjacency, n_swap)
        print(f"    Raw components: {len(components)}")

        # Project for area/clustering
        gdf_proj = gdf_swappable.to_crs(PROJ_CRS)

        # Step 5: Subdivide large components
        print("  Step 5: Subdividing large components...")
        subdivided = subdivide_large_components(components, adjacency, gdf_proj,
                                                max_parcels=max_parcels)
        print(f"    After subdivision: {len(subdivided)} blocks")

        # Step 6: Filter by size
        valid_blocks = []
        dropped = 0
        for comp in subdivided:
            area_ha = gdf_proj.iloc[comp].geometry.area.sum() / 10000
            if len(comp) >= min_parcels and area_ha >= min_area_ha:
                valid_blocks.append(comp)
            else:
                dropped += len(comp)
        print(f"    Valid blocks: {len(valid_blocks)}  |  Dropped parcels: {dropped}")

        # Step 7: Assign block IDs
        gdf_swappable['block_id'] = -1
        for block_id, parcel_indices in enumerate(valid_blocks):
            for idx in parcel_indices:
                gdf_swappable.iloc[idx, gdf_swappable.columns.get_loc('block_id')] = block_id

        # Step 8: Compute block features
        print("  Step 8: Computing block features...")
        block_features = []
        for block_id, parcel_indices in enumerate(valid_blocks):
            feat = compute_block_features(gdf_proj, parcel_indices, block_id)
            block_features.append(feat)

        elapsed = time.time() - t0

        # Summary
        areas = [b['total_area_ha'] for b in block_features]
        n_parcels_list = [b['n_parcels'] for b in block_features]
        baimu = sum(1 for a in areas if a >= 6.67)
        print(f"\n  === Summary ({elapsed:.1f}s) ===")
        print(f"    Blocks: {len(block_features)}")
        print(f"    Parcels/block: min={min(n_parcels_list)}, med={np.median(n_parcels_list):.0f}, max={max(n_parcels_list)}")
        print(f"    Area (ha): min={min(areas):.2f}, med={np.median(areas):.2f}, max={max(areas):.2f}")
        print(f"    Baimu fang (>=6.67ha): {baimu}")

        return {
            'township_code': township_code,
            'label': label,
            'gdf_swappable': gdf_swappable,
            'block_features': block_features,
            'valid_blocks': valid_blocks,
            'n_total': n_total,
            'n_swappable': n_swap,
            'n_blocks': len(block_features),
            'n_baimu': baimu,
            'elapsed': elapsed,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary_table(results):
    """Print a summary table of all townships."""
    print(f"\n{'='*90}")
    print(f"  BISHAN COUNTY BLOCK DEFINITION SUMMARY ({len(results)} townships)")
    print(f"{'='*90}")
    print(f"  {'Code':<12} {'Label':<22} {'Total':>6} {'Swap':>6} {'Blocks':>6} {'Med.P':>5} {'Med.Ha':>7} {'Baimu':>5} {'Time':>6}")
    print(f"  {'-'*12} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*5} {'-'*6}")

    total_blocks = 0
    total_baimu = 0
    total_parcels = 0

    for r in sorted(results, key=lambda x: x['township_code']):
        areas = [b['total_area_ha'] for b in r['block_features']]
        n_parcels_list = [b['n_parcels'] for b in r['block_features']]
        med_p = int(np.median(n_parcels_list)) if n_parcels_list else 0
        med_a = np.median(areas) if areas else 0
        print(f"  {r['township_code']:<12} {r['label']:<22} {r['n_total']:>6} {r['n_swappable']:>6} "
              f"{r['n_blocks']:>6} {med_p:>5} {med_a:>7.2f} {r['n_baimu']:>5} {r['elapsed']:>5.1f}s")
        total_blocks += r['n_blocks']
        total_baimu += r['n_baimu']
        total_parcels += r['n_swappable']

    print(f"  {'-'*12} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*5} {'-'*6}")
    print(f"  {'TOTAL':<12} {'':22} {'':>6} {total_parcels:>6} {total_blocks:>6} {'':>5} {'':>7} {total_baimu:>5}")
    print()


def load_existing_summary(township_code):
    """Load summary from existing results (for --summary mode)."""
    out_dir = os.path.join(OUTPUT_DIR, f'township_{township_code}')
    try:
        with open(os.path.join(out_dir, 'block_features.json'), 'r') as f:
            block_features = json.load(f)
        with open(os.path.join(out_dir, 'block_compositions.json'), 'r') as f:
            compositions = json.load(f)

        n_swappable = 0
        for parcels in compositions.values():
            n_swappable = max(n_swappable, max(parcels) + 1 if parcels else 0)

        areas = [b['total_area_ha'] for b in block_features]
        baimu = sum(1 for a in areas if a >= 6.67)

        return {
            'township_code': township_code,
            'label': ALL_TOWNSHIPS[township_code],
            'block_features': block_features,
            'valid_blocks': [compositions[str(i)] for i in range(len(block_features))],
            'n_total': 0,  # unknown from saved data
            'n_swappable': n_swappable,
            'n_blocks': len(block_features),
            'n_baimu': baimu,
            'elapsed': 0,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Batch block definition for all Bishan townships')
    parser.add_argument('--township', type=str, default=None, help='Single township code')
    parser.add_argument('--skip-existing', action='store_true', help='Skip townships with existing results')
    parser.add_argument('--visualize', action='store_true', help='Generate block maps')
    parser.add_argument('--summary', action='store_true', help='Print summary only (no processing)')
    parser.add_argument('--min-parcels', type=int, default=3)
    parser.add_argument('--min-area', type=float, default=0.5)
    parser.add_argument('--max-parcels', type=int, default=30)
    args = parser.parse_args()

    # Determine which townships to process
    if args.township:
        if args.township not in ALL_TOWNSHIPS:
            print(f"Error: Unknown township {args.township}")
            print(f"Available: {', '.join(sorted(ALL_TOWNSHIPS.keys()))}")
            sys.exit(1)
        target_codes = [args.township]
    else:
        target_codes = sorted(ALL_TOWNSHIPS.keys())

    # Summary-only mode
    if args.summary:
        results = []
        for tc in target_codes:
            r = load_existing_summary(tc)
            if r:
                results.append(r)
            else:
                print(f"  {tc} ({ALL_TOWNSHIPS[tc]}): no results found")
        if results:
            print_summary_table(results)
        return

    # Processing mode
    print(f"Target: {len(target_codes)} townships")
    if args.skip_existing:
        existing = [tc for tc in target_codes if has_existing_results(tc)]
        if existing:
            print(f"Skipping {len(existing)} with existing results: {', '.join(existing)}")
            target_codes = [tc for tc in target_codes if tc not in existing]
        print(f"Processing: {len(target_codes)} townships")

    if not target_codes:
        print("Nothing to process.")
        return

    results = []
    for tc in target_codes:
        label = ALL_TOWNSHIPS[tc]
        result = define_blocks_safe(tc, label,
                                    min_parcels=args.min_parcels,
                                    min_area_ha=args.min_area,
                                    max_parcels=args.max_parcels)
        if result is None:
            continue

        # Save results
        save_results(tc, result['gdf_swappable'], result['block_features'], result['valid_blocks'])

        # Visualize if requested
        if args.visualize:
            visualize_blocks(tc, result['gdf_swappable'], result['block_features'])

        results.append(result)

    # Also load existing results for full summary
    if not args.township:
        for tc in sorted(ALL_TOWNSHIPS.keys()):
            if tc not in [r['township_code'] for r in results]:
                r = load_existing_summary(tc)
                if r:
                    results.append(r)

    if results:
        print_summary_table(results)

    # Save county-wide summary JSON
    summary_path = os.path.join(OUTPUT_DIR, 'county_summary.json')
    summary = {}
    for r in results:
        areas = [b['total_area_ha'] for b in r['block_features']]
        n_list = [b['n_parcels'] for b in r['block_features']]
        summary[r['township_code']] = {
            'label': r['label'],
            'n_swappable': r['n_swappable'],
            'n_blocks': r['n_blocks'],
            'n_baimu': r['n_baimu'],
            'median_parcels_per_block': int(np.median(n_list)) if n_list else 0,
            'median_area_ha': float(np.median(areas)) if areas else 0,
            'total_area_ha': float(sum(areas)) if areas else 0,
        }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"County summary saved: {summary_path}")


if __name__ == '__main__':
    main()
