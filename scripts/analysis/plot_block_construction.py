# -*- coding: utf-8 -*-
"""Generate block construction illustration for Paper 3."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paper3_paths import FIGURES_DIR  # noqa: E402

np.random.seed(42)

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

# Generate a Voronoi-like grid of irregular parcels using random points
from scipy.spatial import Voronoi

# Create random points in a region
n_pts = 80
pts = np.random.rand(n_pts, 2) * 8
# Add boundary points
border = np.array([[x, y] for x in np.linspace(-1, 9, 20) for y in [-1, 9]]
                  + [[x, y] for x in [-1, 9] for y in np.linspace(-1, 9, 20)])
all_pts = np.vstack([pts, border])
vor = Voronoi(all_pts)

# Assign land use types
# farmland=0 (green), forest=1 (brown), road=2 (gray), water=3 (blue)
land_type = np.zeros(n_pts, dtype=int)
for i in range(n_pts):
    r = np.random.rand()
    if r < 0.05:  # road
        land_type[i] = 2
    elif r < 0.1:  # water
        land_type[i] = 3
    elif r < 0.4:  # forest
        land_type[i] = 1
    else:  # farmland
        land_type[i] = 0

# Create road/water as barriers in specific positions
for i in range(n_pts):
    x, y = pts[i]
    # Horizontal road
    if 3.5 < y < 4.2 and x > 1 and x < 7:
        land_type[i] = 2
    # Vertical stream
    if 5.0 < x < 5.7 and y > 1 and y < 7:
        land_type[i] = 3

colors_land = {'farmland': '#66BB6A', 'forest': '#8D6E63', 'road': '#9E9E9E', 'water': '#42A5F5'}
cmap_list = [colors_land['farmland'], colors_land['forest'], colors_land['road'], colors_land['water']]

def draw_voronoi(ax, vor, pts, n_pts, colors_arr, title, show_edges=True, edge_color='#666666', edge_lw=0.5):
    """Draw Voronoi cells colored by array."""
    for i in range(n_pts):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        polygon = [vor.vertices[v] for v in region]
        polygon = np.array(polygon)
        # Clip to bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, 8)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, 8)
        poly = plt.Polygon(polygon, facecolor=colors_arr[i],
                          edgecolor=edge_color if show_edges else colors_arr[i],
                          linewidth=edge_lw, alpha=0.85)
        ax.add_patch(poly)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

# ── Panel 1: Raw parcels by land use ──
ax = axes[0]
raw_colors = [cmap_list[land_type[i]] for i in range(n_pts)]
draw_voronoi(ax, vor, pts, n_pts, raw_colors, 'Step 1: Raw Parcels\n(by land use)')
# Legend
legend_patches = [
    mpatches.Patch(color=colors_land['farmland'], label='Farmland'),
    mpatches.Patch(color=colors_land['forest'], label='Forest'),
    mpatches.Patch(color=colors_land['road'], label='Road'),
    mpatches.Patch(color=colors_land['water'], label='Water'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7, framealpha=0.9)

# ── Panel 2: Swappable only (barriers removed) ──
ax = axes[1]
swap_colors = []
for i in range(n_pts):
    if land_type[i] in (2, 3):  # barrier
        swap_colors.append('#FFFFFF')  # white = removed
    else:
        swap_colors.append(cmap_list[land_type[i]])
draw_voronoi(ax, vor, pts, n_pts, swap_colors,
             'Step 2: Swappable Only\n(barriers removed)', edge_lw=0.3)
# Draw X marks on barriers
for i in range(n_pts):
    if land_type[i] in (2, 3):
        ax.plot(pts[i, 0], pts[i, 1], 'x', color='red', markersize=5, markeredgewidth=1.5)

# ── Panel 3: Connected components ──
ax = axes[2]
# Simple connected component simulation: parcels on same side of barriers get same component
comp_id = np.zeros(n_pts, dtype=int)
comp_colors_map = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC',
                   '#26C6DA', '#FF7043', '#9CCC65']
for i in range(n_pts):
    if land_type[i] in (2, 3):
        comp_id[i] = -1
        continue
    x, y = pts[i]
    # Assign to quadrant based on road/stream
    if y < 3.85:
        if x < 5.35:
            comp_id[i] = 0  # bottom-left
        else:
            comp_id[i] = 1  # bottom-right
    else:
        if x < 5.35:
            comp_id[i] = 2  # top-left
        else:
            comp_id[i] = 3  # top-right

comp_colors = []
for i in range(n_pts):
    if comp_id[i] == -1:
        comp_colors.append('#FFFFFF')
    else:
        comp_colors.append(comp_colors_map[comp_id[i]])
draw_voronoi(ax, vor, pts, n_pts, comp_colors,
             'Step 3: Connected\nComponents (BFS)', edge_lw=0.5)
# Label components
for cid, label_pos in [(0, (2.5, 1.8)), (1, (6.5, 1.8)), (2, (2.5, 6.0)), (3, (6.5, 6.0))]:
    ax.text(label_pos[0], label_pos[1], f'C{cid+1}', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor=comp_colors_map[cid], alpha=0.8))

# ── Panel 4: Agglomerative clustering (subdivide large components) ──
ax = axes[3]
# Subdivide each quadrant into ~2-3 sub-blocks
block_id = np.zeros(n_pts, dtype=int)
block_colors_map = ['#E53935', '#F44336', '#FF5722',  # reds for comp0
                    '#1E88E5', '#2196F3', '#42A5F5',  # blues for comp1
                    '#43A047', '#66BB6A', '#81C784',  # greens for comp2
                    '#FB8C00', '#FFA726', '#FFB74D']   # oranges for comp3
bid = 0
for i in range(n_pts):
    if land_type[i] in (2, 3):
        block_id[i] = -1
        continue
    x, y = pts[i]
    base = comp_id[i] * 3
    # Sub-divide within quadrant
    if comp_id[i] == 0:  # bottom-left
        if x < 2.5:
            block_id[i] = base
        elif y < 2.0:
            block_id[i] = base + 1
        else:
            block_id[i] = base + 2
    elif comp_id[i] == 1:  # bottom-right
        if y < 2.0:
            block_id[i] = base
        else:
            block_id[i] = base + 1
    elif comp_id[i] == 2:  # top-left
        if x < 2.5:
            block_id[i] = base
        elif y > 6.0:
            block_id[i] = base + 1
        else:
            block_id[i] = base + 2
    else:  # top-right
        if y > 6.0:
            block_id[i] = base
        else:
            block_id[i] = base + 1

blk_colors = []
for i in range(n_pts):
    if block_id[i] == -1:
        blk_colors.append('#FFFFFF')
    else:
        blk_colors.append(block_colors_map[block_id[i] % len(block_colors_map)])
draw_voronoi(ax, vor, pts, n_pts, blk_colors,
             'Step 4: Constrained\nClustering', edge_lw=0.8, edge_color='#333333')

# ── Panel 5: Final blocks with bold boundaries ──
ax = axes[4]
# Same as panel 4 but with thicker block boundaries and cleaner look
draw_voronoi(ax, vor, pts, n_pts, blk_colors,
             'Step 5: Final Blocks\n(~20 parcels each)', edge_lw=0.3, edge_color='#999999')
# Add block labels
block_positions = {}
for i in range(n_pts):
    if block_id[i] == -1:
        continue
    bid = block_id[i]
    if bid not in block_positions:
        block_positions[bid] = []
    block_positions[bid].append(pts[i])
for bid, positions in block_positions.items():
    positions = np.array(positions)
    cx, cy = positions.mean(axis=0)
    ax.text(cx, cy, f'B{bid+1}', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor=block_colors_map[bid % len(block_colors_map)],
                     alpha=0.85, edgecolor='black', linewidth=0.8))

# Add legend for step 5
ax.text(4, -0.5, 'Barriers (roads, water) → natural block boundaries',
        fontsize=8, ha='center', va='center', style='italic', color='#666666')

# ── Arrows between panels ──
for i in range(4):
    fig.text(0.107 + 0.196 * (i + 1), 0.5, '→', fontsize=24, fontweight='bold',
             ha='center', va='center', color='#1565C0')

plt.suptitle('Block Construction: Hybrid Barrier Segmentation + Constrained Agglomerative Clustering',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out_path = FIGURES_DIR / 'paper3_block_construction.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {out_path}")
