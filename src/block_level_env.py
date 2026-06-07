# -*- coding: utf-8 -*-
"""
Block-Level MDP Environment for Paper 3 (Direction 2: 百亩方 Formation).

The DRL agent selects which block to invest consolidation resources in.
The environment internally runs slope-greedy swaps within the selected block.
"Macroscopic Planning, Microscopic Execution."

Direction 2 adds 百亩方 (contiguous farmland ≥6.67ha) formation as a key
reward signal, creating genuine sequential dependencies through spatial
synergy between adjacent blocks.

Compatible with sb3-contrib MaskablePPO.

Usage:
    env = BlockLevelEnv('500227109', total_budget=100, swaps_per_step=5)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
"""

import os
import json
import numpy as np
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from paper3_paths import BLOCK_RESULTS_DIR, DLTB_PATH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_DIR = str(BLOCK_RESULTS_DIR)
PROJ_CRS = 'EPSG:32648'

FARMLAND = 1
FOREST = 2

FARMLAND_PREFIXES = ('011', '012', '013')
FOREST_PREFIXES = ('031', '032', '033')

# Per-block feature count (17: 13 original + 4 adjacency/investment features)
K_BLOCK = 17
# Global feature count (9: 6 original + 3 百亩方/investment features)
K_GLOBAL = 9

# 百亩方 threshold: 100 亩 = 6.67 hectares = 66,700 m²
BAIMU_THRESHOLD_M2 = 66700.0


def _classify_type(dlbm):
    if dlbm.startswith(FARMLAND_PREFIXES):
        return FARMLAND
    elif dlbm.startswith(FOREST_PREFIXES):
        return FOREST
    return 0


class BlockLevelEnv(gym.Env):
    """Block-level land use optimization environment with 百亩方 formation reward.

    State:
        Per-block features (n_blocks x K_BLOCK) + global stats (K_GLOBAL).
        Flattened into a single vector for compatibility with MaskablePPO.

    Action:
        Discrete(n_blocks) — select which block to invest in.

    Step mechanics:
        1. Agent selects block_id
        2. Environment runs up to `swaps_per_step` slope-greedy paired swaps
           within that block
        3. Budget consumed; reward reflects slope + contiguity + 百亩方 changes

    Episode:
        Fixed length = total_budget // swaps_per_step.
        Terminates early if no blocks have remaining swap potential.

    Sequential dependency:
        百亩方 formation creates spatial synergy — investing in adjacent blocks
        can merge farmland patches across boundaries, forming large contiguous
        farmland areas that neither block alone could achieve.
    """

    metadata = {"render_modes": []}

    def __init__(self, township_code, total_budget=100, swaps_per_step=5,
                 slope_weight=1000.0, cont_weight=500.0,
                 baimu_weight=2000.0, baimu_bonus=50.0,
                 baimu_threshold_m2=BAIMU_THRESHOLD_M2,
                 gamma_conn=1.0, delta_conn=0.5):
        super().__init__()

        self.township_code = township_code
        self.total_budget = total_budget
        self.swaps_per_step = swaps_per_step
        self.max_steps = total_budget // swaps_per_step
        self.slope_weight = slope_weight
        self.cont_weight = cont_weight
        self.baimu_weight = baimu_weight
        self.baimu_bonus = baimu_bonus
        self.baimu_threshold_m2 = baimu_threshold_m2
        self.gamma_conn = gamma_conn  # connectivity bonus for forest→farmland
        self.delta_conn = delta_conn  # connectivity penalty for farmland→forest

        # Load parcel data, adjacency, block compositions, and block adjacency
        self._load_data()

        # Spaces
        self.n_blocks = len(self.block_parcels)
        self.action_space = spaces.Discrete(self.n_blocks)
        obs_dim = self.n_blocks * K_BLOCK + K_GLOBAL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize state
        self.land_use = self.initial_types.copy()
        self.swapped = np.zeros(self.n_parcels, dtype=bool)
        self.budget_used = 0
        self.step_count = 0
        self.swaps_in_block = np.zeros(self.n_blocks, dtype=np.int32)

        # Per-block available counters (for fast action masking)
        self._block_farm_avail = np.zeros(self.n_blocks, dtype=np.int32)
        self._block_forest_avail = np.zeros(self.n_blocks, dtype=np.int32)
        self._init_block_counters()

        # Compute initial metrics (slope, contiguity, 百亩方)
        self._compute_metrics_full()
        self.baimu_count, self.baimu_total_area = self._count_baimu_fang()
        self._cache_initial_state()

        print(f"  Obs dim: {obs_dim}, Action dim: {self.n_blocks}")
        print(f"  Initial avg farmland slope: {self.avg_farmland_slope:.4f}")
        print(f"  Initial contiguity: {self.contiguity:.4f}")
        print(f"  Initial 百亩方: {self.baimu_count} patches, "
              f"{self.baimu_total_area/10000:.1f} ha total")
        print(f"  Block adjacency: median {np.median([len(a) for a in self.block_adj]):.0f} neighbors")
        print(f"  Connectivity-aware greedy: γ={self.gamma_conn}, δ={self.delta_conn}")

    # ==================================================================
    # Data loading
    # ==================================================================

    def _load_data(self):
        """Load parcel data, build adjacency, load block compositions, build block adjacency."""
        print(f"BlockLevelEnv: Loading township {self.township_code}...")

        # Load parcels from DLTB
        gdf = gpd.read_file(str(DLTB_PATH), where=f"QSDWDM LIKE '{self.township_code}%'")
        gdf['type_code'] = gdf['DLBM'].apply(_classify_type)

        # Filter to swappable (farmland + forest)
        gdf_swap = gdf[gdf['type_code'].isin([FARMLAND, FOREST])].copy()
        gdf_swap = gdf_swap.reset_index(drop=True)
        self.n_parcels = len(gdf_swap)

        # Project for area computation
        gdf_proj = gdf_swap.to_crs(PROJ_CRS)

        # Extract arrays
        self.initial_types = gdf_swap['type_code'].values.astype(np.int8)
        self.slopes = gdf_swap['slope_mean'].values.astype(np.float64)
        self.areas = gdf_proj.geometry.area.values.astype(np.float64)

        # Slope normalization params
        self.slope_min = float(self.slopes.min())
        self.slope_max = float(self.slopes.max())
        self.slope_range = self.slope_max - self.slope_min + 1e-8

        # Build adjacency (Queen contiguity among swappable parcels)
        print(f"  Building adjacency ({self.n_parcels} swappable parcels)...")
        self._build_adjacency(gdf_swap)

        # Load block compositions
        block_dir = os.path.join(BLOCK_DIR, f'township_{self.township_code}')
        with open(os.path.join(block_dir, 'block_compositions.json')) as f:
            compositions = json.load(f)

        self.block_parcels = []
        for i in range(len(compositions)):
            self.block_parcels.append(np.array(compositions[str(i)], dtype=np.intp))

        # Block-level static attributes
        self.block_areas = np.array([self.areas[bp].sum() for bp in self.block_parcels])
        self.max_block_area = self.block_areas.max() + 1e-8

        with open(os.path.join(block_dir, 'block_features.json')) as f:
            saved_feats = json.load(f)
        self.block_compactness = np.array([b['compactness'] for b in saved_feats],
                                          dtype=np.float32)

        # Parcel-to-block mapping
        self.parcel_to_block = np.full(self.n_parcels, -1, dtype=np.int32)
        for bid, parcels in enumerate(self.block_parcels):
            self.parcel_to_block[parcels] = bid

        n_assigned = int((self.parcel_to_block >= 0).sum())
        print(f"  {len(self.block_parcels)} blocks, {n_assigned}/{self.n_parcels} parcels assigned")

        # Build block adjacency graph
        self._build_block_adjacency()

        print(f"  Budget: {self.total_budget} swaps, {self.swaps_per_step}/step, "
              f"{self.max_steps} steps")

    def _build_adjacency(self, gdf_swap):
        """Build adjacency lists via libpysal Queen contiguity."""
        try:
            from libpysal.weights import Queen
            w = Queen.from_dataframe(gdf_swap, use_index=False)
            self.adjacency = [np.array(w.neighbors[i], dtype=np.intp)
                              for i in range(self.n_parcels)]
        except Exception as e:
            print(f"  libpysal failed ({e}), using spatial index fallback")
            from shapely.strtree import STRtree
            geoms = gdf_swap.geometry.values
            tree = STRtree(geoms)
            self.adjacency = []
            for i in range(self.n_parcels):
                cands = tree.query(geoms[i], predicate='intersects')
                self.adjacency.append(np.array([j for j in cands if j != i], dtype=np.intp))

        self.total_nbr_count = np.array(
            [len(self.adjacency[i]) for i in range(self.n_parcels)], dtype=np.float32
        )

    def _build_block_adjacency(self):
        """Build block-level adjacency graph from parcel adjacency.

        Two blocks are adjacent if any parcel in block A neighbors
        any parcel in block B. This is the key structure enabling
        cross-boundary 百亩方 formation.
        """
        n_blocks = len(self.block_parcels)
        adj_sets = [set() for _ in range(n_blocks)]

        for i in range(self.n_parcels):
            bi = self.parcel_to_block[i]
            if bi < 0:
                continue
            for j in self.adjacency[i]:
                bj = self.parcel_to_block[j]
                if bj >= 0 and bj != bi:
                    adj_sets[bi].add(bj)

        self.block_adj = [np.array(sorted(s), dtype=np.intp) for s in adj_sets]
        self.block_n_adj = np.array([len(a) for a in self.block_adj], dtype=np.int32)

    def _init_block_counters(self):
        """Initialize per-block available parcel counters."""
        for b, parcels in enumerate(self.block_parcels):
            types = self.land_use[parcels]
            self._block_farm_avail[b] = int(((types == FARMLAND) & ~self.swapped[parcels]).sum())
            self._block_forest_avail[b] = int(((types == FOREST) & ~self.swapped[parcels]).sum())

    # ==================================================================
    # Metrics (area-weighted slope, contiguity, 百亩方)
    # ==================================================================

    def _compute_metrics_full(self):
        """Compute slope/contiguity metrics from scratch."""
        fm = self.land_use == FARMLAND
        self.n_farmland = int(fm.sum())
        self.n_forest = int((self.land_use == FOREST).sum())

        self.total_weighted_slope = float((self.slopes[fm] * self.areas[fm]).sum())
        self.total_farm_area = float(self.areas[fm].sum())

        self.farmland_nbr_count = np.zeros(self.n_parcels, dtype=np.int32)
        for i in range(self.n_parcels):
            nbrs = self.adjacency[i]
            if len(nbrs) > 0:
                self.farmland_nbr_count[i] = int((self.land_use[nbrs] == FARMLAND).sum())
        self.total_farmland_adj = int(self.farmland_nbr_count[fm].sum())

    def _count_baimu_fang(self):
        """Count 百亩方 patches via BFS on farmland connected components.

        A 百亩方 is a contiguous farmland patch with total area >= threshold.
        Uses parcel-level adjacency (crosses block boundaries).

        Returns:
            (baimu_count, baimu_total_area): count and total area (m²)
        """
        visited = np.zeros(self.n_parcels, dtype=bool)
        baimu_count = 0
        baimu_total_area = 0.0

        for start in range(self.n_parcels):
            if visited[start] or self.land_use[start] != FARMLAND:
                continue

            # BFS to find connected farmland component
            component_area = 0.0
            queue = deque([start])
            visited[start] = True

            while queue:
                node = queue.popleft()
                component_area += self.areas[node]
                for nbr in self.adjacency[node]:
                    if not visited[nbr] and self.land_use[nbr] == FARMLAND:
                        visited[nbr] = True
                        queue.append(nbr)

            if component_area >= self.baimu_threshold_m2:
                baimu_count += 1
                baimu_total_area += component_area

        return baimu_count, baimu_total_area

    @property
    def avg_farmland_slope(self):
        return self.total_weighted_slope / max(self.total_farm_area, 1e-8)

    @property
    def contiguity(self):
        return self.total_farmland_adj / max(self.n_farmland, 1)

    # ==================================================================
    # Incremental swap updates
    # ==================================================================

    def _swap_to_forest(self, k):
        """Farmland -> Forest at parcel k. O(degree) incremental update."""
        self.total_farmland_adj -= self.farmland_nbr_count[k]
        self.total_weighted_slope -= self.slopes[k] * self.areas[k]
        self.total_farm_area -= self.areas[k]

        self.land_use[k] = FOREST
        self.n_farmland -= 1
        self.n_forest += 1

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] -= 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj -= 1

    def _swap_to_farmland(self, k):
        """Forest -> Farmland at parcel k. O(degree) incremental update."""
        self.land_use[k] = FARMLAND
        self.n_farmland += 1
        self.n_forest -= 1
        self.total_weighted_slope += self.slopes[k] * self.areas[k]
        self.total_farm_area += self.areas[k]

        self.total_farmland_adj += self.farmland_nbr_count[k]

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] += 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj += 1

    # ==================================================================
    # Greedy execution engine (microscopic execution)
    # ==================================================================

    def _execute_greedy_in_block(self, block_id, max_swaps):
        """Run connectivity-aware greedy paired swaps within a single block.

        Farmland removal: score = slope - δ * farmland_nbr_count
            → prefer steep AND isolated farmland (preserves contiguity)
        Forest conversion: score = slope - γ * farmland_nbr_count
            → prefer gentle AND well-connected forest (grows contiguity)

        When γ=δ=0, reduces to pure slope-greedy (Paper 2 baseline).

        Returns:
            int: number of completed paired swaps
        """
        parcels = self.block_parcels[block_id]
        completed = 0

        for _ in range(max_swaps):
            types = self.land_use[parcels]
            avail = ~self.swapped[parcels]

            farm_mask = (types == FARMLAND) & avail
            forest_mask = (types == FOREST) & avail

            if not farm_mask.any() or not forest_mask.any():
                break

            farm_local = np.where(farm_mask)[0]
            forest_local = np.where(forest_mask)[0]

            farm_idx = parcels[farm_local]
            forest_idx = parcels[forest_local]

            # Farmland to remove: maximize slope - δ * connectivity
            # High slope + few farmland neighbors → remove first
            farm_scores = (self.slopes[farm_idx]
                           - self.delta_conn * self.farmland_nbr_count[farm_idx])
            best_farm = farm_idx[np.argmax(farm_scores)]

            # Forest to convert: minimize slope - γ * connectivity
            # Low slope + many farmland neighbors → convert first
            forest_scores = (self.slopes[forest_idx]
                             - self.gamma_conn * self.farmland_nbr_count[forest_idx])
            best_forest = forest_idx[np.argmin(forest_scores)]

            # Only swap if beneficial (slope gap still positive)
            if self.slopes[best_farm] <= self.slopes[best_forest]:
                break

            # Execute paired swap
            self._swap_to_forest(best_farm)
            self._swap_to_farmland(best_forest)
            self.swapped[best_farm] = True
            self.swapped[best_forest] = True
            completed += 1

            # Update block counters
            self._block_farm_avail[block_id] -= 1
            self._block_forest_avail[block_id] -= 1

        return completed

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def _get_block_features(self):
        """Compute per-block feature matrix (n_blocks x K_BLOCK=17).

        Features per block:
            [0]  avg_farm_slope_norm
            [1]  avg_forest_slope_norm
            [2]  slope_gap_norm
            [3]  best_swap_gain_norm
            [4]  farm_slope_std_norm
            [5]  top_farm_slope_norm
            [6]  bottom_forest_slope_norm
            [7]  farm_area_frac
            [8]  forest_area_frac
            [9]  swap_potential
            [10] invested_frac (swaps done / swaps_per_step)
            [11] compactness
            [12] block_area_norm
            [13] n_adj_invested_frac   — fraction of adjacent blocks invested
            [14] adj_farm_area_norm    — farmland area in adjacent blocks (norm)
            [15] block_farm_area_norm  — current farmland area in this block (norm)
            [16] is_invested           — 1.0 if block has been invested in
        """
        features = np.zeros((self.n_blocks, K_BLOCK), dtype=np.float32)

        # Pre-compute current farmland area per block for adjacency features
        block_farm_areas = np.zeros(self.n_blocks, dtype=np.float64)
        for b in range(self.n_blocks):
            parcels = self.block_parcels[b]
            fm = self.land_use[parcels] == FARMLAND
            block_farm_areas[b] = self.areas[parcels[fm]].sum() if fm.any() else 0.0

        for b in range(self.n_blocks):
            parcels = self.block_parcels[b]
            types = self.land_use[parcels]
            avail = ~self.swapped[parcels]

            fm = (types == FARMLAND) & avail
            ff = (types == FOREST) & avail

            # Area-weighted slopes within block
            if fm.any():
                farm_s = self.slopes[parcels[fm]]
                farm_a = self.areas[parcels[fm]]
                avg_farm = np.average(farm_s, weights=farm_a)
                farm_area = farm_a.sum()
                farm_std = farm_s.std() if len(farm_s) > 1 else 0.0
                top_farm = farm_s.max()
            else:
                avg_farm = 0.0
                farm_area = 0.0
                farm_std = 0.0
                top_farm = 0.0

            if ff.any():
                for_s = self.slopes[parcels[ff]]
                for_a = self.areas[parcels[ff]]
                avg_for = np.average(for_s, weights=for_a)
                forest_area = for_a.sum()
                bottom_forest = for_s.min()
            else:
                avg_for = 0.0
                forest_area = 0.0
                bottom_forest = 0.0

            # Best swap gain: what the greedy engine would achieve next
            best_gain = top_farm - bottom_forest if (fm.any() and ff.any()) else 0.0

            features[b, 0] = (avg_farm - self.slope_min) / self.slope_range
            features[b, 1] = (avg_for - self.slope_min) / self.slope_range
            features[b, 2] = (avg_farm - avg_for) / self.slope_range
            features[b, 3] = best_gain / self.slope_range
            features[b, 4] = farm_std / self.slope_range
            features[b, 5] = (top_farm - self.slope_min) / self.slope_range
            features[b, 6] = (bottom_forest - self.slope_min) / self.slope_range
            features[b, 7] = farm_area / self.max_block_area
            features[b, 8] = forest_area / self.max_block_area
            features[b, 9] = min(self._block_farm_avail[b],
                                 self._block_forest_avail[b]) / max(len(parcels), 1)
            features[b, 10] = self.swaps_in_block[b] / max(self.swaps_per_step, 1)
            features[b, 11] = self.block_compactness[b]
            features[b, 12] = self.block_areas[b] / self.max_block_area

            # Adjacency/investment features (Direction 2)
            adj_blocks = self.block_adj[b]
            if len(adj_blocks) > 0:
                n_adj_invested = int((self.swaps_in_block[adj_blocks] > 0).sum())
                features[b, 13] = n_adj_invested / len(adj_blocks)
                features[b, 14] = block_farm_areas[adj_blocks].sum() / (
                    self.max_block_area * len(adj_blocks))
            else:
                features[b, 13] = 0.0
                features[b, 14] = 0.0

            features[b, 15] = block_farm_areas[b] / self.max_block_area
            features[b, 16] = 1.0 if self.swaps_in_block[b] > 0 else 0.0

        return features

    def _get_global_features(self):
        """Compute global feature vector (K_GLOBAL=9).

        Features:
            [0] budget_remaining_frac
            [1] global_slope_norm
            [2] global_cont_norm
            [3] step_frac
            [4] slope_improvement
            [5] cont_improvement
            [6] baimu_count_norm       — current 百亩方 count (normalized)
            [7] baimu_area_frac        — 百亩方 area / total farmland area
            [8] blocks_invested_frac   — fraction of blocks invested in
        """
        cur_slope = self.avg_farmland_slope
        cur_cont = self.contiguity
        n_invested = int((self.swaps_in_block > 0).sum())

        return np.array([
            1.0 - self.step_count / self.max_steps,
            (cur_slope - self.slope_min) / self.slope_range,
            cur_cont / 10.0,
            self.step_count / self.max_steps,
            (self.initial_slope - cur_slope) / (abs(self.initial_slope) + 1e-8),
            (cur_cont - self.initial_cont) / (abs(self.initial_cont) + 1e-8),
            self.baimu_count / max(self.n_blocks / 10.0, 1.0),
            self.baimu_total_area / max(self.total_farm_area, 1e-8),
            n_invested / self.n_blocks,
        ], dtype=np.float32)

    def _get_obs(self):
        """Build flat observation vector: [block_features | global_features]."""
        block_feats = self._get_block_features()   # (n_blocks, K_BLOCK)
        global_feats = self._get_global_features()  # (K_GLOBAL,)
        return np.concatenate([block_feats.ravel(), global_feats])

    def action_masks(self):
        """Boolean mask over blocks: True if block has swap potential."""
        return (self._block_farm_avail > 0) & (self._block_forest_avail > 0)

    def _cache_initial_state(self):
        """Cache initial state for fast reset."""
        self.initial_slope = self.avg_farmland_slope
        self.initial_cont = self.contiguity
        self.initial_baimu_count = self.baimu_count
        self.initial_baimu_area = self.baimu_total_area
        self.initial_farm_area = self.total_farm_area
        self._init_cache = {
            'land_use': self.land_use.copy(),
            'n_farmland': self.n_farmland,
            'n_forest': self.n_forest,
            'total_weighted_slope': self.total_weighted_slope,
            'total_farm_area': self.total_farm_area,
            'farmland_nbr_count': self.farmland_nbr_count.copy(),
            'total_farmland_adj': self.total_farmland_adj,
            'block_farm_avail': self._block_farm_avail.copy(),
            'block_forest_avail': self._block_forest_avail.copy(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        c = self._init_cache
        self.land_use = c['land_use'].copy()
        self.n_farmland = c['n_farmland']
        self.n_forest = c['n_forest']
        self.total_weighted_slope = c['total_weighted_slope']
        self.total_farm_area = c['total_farm_area']
        self.farmland_nbr_count = c['farmland_nbr_count'].copy()
        self.total_farmland_adj = c['total_farmland_adj']
        self._block_farm_avail = c['block_farm_avail'].copy()
        self._block_forest_avail = c['block_forest_avail'].copy()

        self.swapped[:] = False
        self.budget_used = 0
        self.step_count = 0
        self.swaps_in_block[:] = 0

        # Reset 百亩方 tracking
        self.baimu_count = self.initial_baimu_count
        self.baimu_total_area = self.initial_baimu_area

        self.prev_slope = self.initial_slope
        self.prev_cont = self.initial_cont
        self.prev_baimu_count = self.initial_baimu_count
        self.prev_baimu_area = self.initial_baimu_area

        obs = self._get_obs()
        info = {
            'avg_slope': self.avg_farmland_slope,
            'contiguity': self.contiguity,
            'baimu_count': self.baimu_count,
            'baimu_area_ha': self.baimu_total_area / 10000.0,
            'budget_used': 0,
        }
        return obs, info

    def step(self, action):
        block_id = int(action)

        # Microscopic execution: greedy swaps within selected block
        completed = self._execute_greedy_in_block(block_id, self.swaps_per_step)
        self.budget_used += completed
        self.swaps_in_block[block_id] += completed
        self.step_count += 1

        # Recompute 百亩方 (BFS over all farmland, O(N_parcels))
        self.baimu_count, self.baimu_total_area = self._count_baimu_fang()

        # Compute reward: incremental improvements
        cur_slope = self.avg_farmland_slope
        cur_cont = self.contiguity

        slope_delta = (self.prev_slope - cur_slope) / (abs(self.initial_slope) + 1e-8)
        cont_delta = (cur_cont - self.prev_cont) / (abs(self.initial_cont) + 1e-8)
        baimu_area_delta = ((self.baimu_total_area - self.prev_baimu_area)
                            / (self.initial_farm_area + 1e-8))
        baimu_new_count = max(0, self.baimu_count - self.prev_baimu_count)

        reward = (self.slope_weight * slope_delta
                  + self.cont_weight * cont_delta
                  + self.baimu_weight * baimu_area_delta
                  + self.baimu_bonus * baimu_new_count)

        # Small penalty for wasted steps (selected block with 0 actual swaps)
        if completed == 0:
            reward -= 1.0

        self.prev_slope = cur_slope
        self.prev_cont = cur_cont
        self.prev_baimu_count = self.baimu_count
        self.prev_baimu_area = self.baimu_total_area

        # Termination
        terminated = self.step_count >= self.max_steps
        if not terminated:
            if not self.action_masks().any():
                terminated = True

        info = {
            'avg_slope': cur_slope,
            'contiguity': cur_cont,
            'baimu_count': self.baimu_count,
            'baimu_area_ha': self.baimu_total_area / 10000.0,
            'budget_used': self.budget_used,
            'completed_swaps': completed,
            'block_selected': block_id,
            'step': self.step_count,
            'slope_change_pct': 100.0 * (cur_slope - self.initial_slope) / (
                abs(self.initial_slope) + 1e-8),
            'cont_change': cur_cont - self.initial_cont,
            'baimu_count_change': self.baimu_count - self.initial_baimu_count,
            'baimu_area_change_ha': (self.baimu_total_area - self.initial_baimu_area) / 10000.0,
        }

        return self._get_obs(), float(reward), terminated, False, info


# ======================================================================
# Quick test
# ======================================================================

if __name__ == '__main__':
    import sys
    import time
    tc = sys.argv[1] if len(sys.argv) > 1 else '500227109'
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sps = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    env = BlockLevelEnv(tc, total_budget=budget, swaps_per_step=sps)
    obs, info = env.reset()
    print(f"\nObs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Valid actions: {env.action_masks().sum()} / {env.n_blocks}")

    # Benchmark 百亩方 counting
    t0 = time.time()
    for _ in range(100):
        env._count_baimu_fang()
    t_baimu = (time.time() - t0) / 100
    print(f"百亩方 BFS time: {t_baimu*1000:.1f} ms/call")

    # Run random episode
    total_reward = 0
    while True:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = np.random.choice(valid)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\nRandom episode complete:")
    print(f"  Steps: {info['step']}, Budget used: {info['budget_used']}")
    print(f"  Slope: {info['slope_change_pct']:+.2f}%")
    print(f"  Contiguity change: {info['cont_change']:+.4f}")
    print(f"  百亩方 count change: {info['baimu_count_change']:+d}")
    print(f"  百亩方 area change: {info['baimu_area_change_ha']:+.2f} ha")
    print(f"  Total reward: {total_reward:.2f}")
