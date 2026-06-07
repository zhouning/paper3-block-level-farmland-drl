"""Gymnasium environment for land use optimization on real cadastral data.

Adapts land_use_env_v7.py for:
  - DLBM-code land use classification (Third National Land Survey standard)
  - Variable parcel areas (TBMJ field, not uniform grids)
  - Pre-computed adjacency graphs (from build_adjacency.py)
  - Extended feature vectors (K_PARCEL=10 with aspect, elevation, shape index)
  - Pre-computed feature tables (from compute_parcel_features.py)

Same reward structure as v7: slope reduction + contiguity improvement
- count penalty + pair bonus. Same incremental O(1) metric updates.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from township_utils import FARMLAND, FOREST, OTHER
from build_adjacency import load_adjacency


# Per-parcel feature count (expanded from 6 to 10)
K_PARCEL = 10
# Global feature count (same as v7)
K_GLOBAL = 8

# Reward weights (same as v7)
SLOPE_REWARD_WEIGHT = 1000.0
CONT_REWARD_WEIGHT = 500.0
COUNT_PENALTY_WEIGHT = 500.0
PAIR_BONUS = 1.0


class RealDataLandUseEnv(gym.Env):
    """Land use optimization environment for real cadastral data.

    Key differences from LandUseOptEnv (v7):
    - Loads from pre-computed features CSV + adjacency NPZ (not shapefile)
    - Classification by DLBM code, not DLMC text
    - 10 per-parcel features (vs 6 in v7)
    - Variable parcel areas from TBMJ field
    - Adjacency pre-computed via libpysal Queen contiguity
    """

    metadata = {"render_modes": []}

    def __init__(self, features_csv, adjacency_npz,
                 max_conversions=200,
                 count_penalty_weight=None,
                 slope_reward_weight=None,
                 cont_reward_weight=None,
                 enforce_pairs=False,
                 k_parcel=None,
                 k_global=None):
        """Initialize environment.

        Args:
            features_csv: Path to township feature CSV from compute_parcel_features.py
            adjacency_npz: Path to adjacency NPZ from build_adjacency.py
            max_conversions: Number of swap steps per episode
            count_penalty_weight: Override for count penalty (default: 500)
            slope_reward_weight: Override for slope reward weight (default: 1000)
            cont_reward_weight: Override for contiguity reward weight (default: 500)
            enforce_pairs: If True, alternate farmland/forest action masks
            k_parcel: Override per-parcel feature count (default: 10)
            k_global: Override global feature count (default: 8)
        """
        super().__init__()

        self.count_penalty_weight = count_penalty_weight or COUNT_PENALTY_WEIGHT
        self.slope_reward_weight = slope_reward_weight if slope_reward_weight is not None else SLOPE_REWARD_WEIGHT
        self.cont_reward_weight = cont_reward_weight if cont_reward_weight is not None else CONT_REWARD_WEIGHT
        self.enforce_pairs = enforce_pairs
        self.k_parcel = k_parcel or K_PARCEL
        self.k_global = k_global or K_GLOBAL

        # Load features
        print(f"Loading features: {features_csv}")
        df = pd.read_csv(features_csv)
        self.n_parcels = len(df)
        self.max_steps = max_conversions

        # Extract attributes
        self.slopes = df['slope_mean'].values.astype(np.float64)
        self.areas = df['TBMJ'].values.astype(np.float64)

        # Classify parcels from type_code column
        self.initial_types = df['type_code'].values.astype(np.int8)

        # Identify swappable parcels (farmland or forest)
        self.swappable_indices = np.where(
            (self.initial_types == FARMLAND) | (self.initial_types == FOREST)
        )[0]
        self.n_swappable = len(self.swappable_indices)

        # Normalize slopes to [0, 1]
        self.slope_min = float(self.slopes.min())
        self.slope_max_val = float(self.slopes.max())
        self.slope_range = self.slope_max_val - self.slope_min + 1e-8
        self.slopes_norm = ((self.slopes - self.slope_min) / self.slope_range).astype(np.float32)

        # Normalize areas to [0, 1]
        area_min = float(self.areas.min())
        area_max = float(self.areas.max())
        area_range = area_max - area_min + 1e-8
        self.areas_norm = ((self.areas - area_min) / area_range).astype(np.float32)

        # Extended features: normalize elevation, shape_index
        elev = df['elevation_mean'].values.astype(np.float64)
        elev_min, elev_max = elev.min(), elev.max()
        self.elevation_norm = ((elev - elev_min) / (elev_max - elev_min + 1e-8)).astype(np.float32)

        # Aspect sin/cos are already in [-1, 1]
        self.aspect_sin = df['aspect_sin'].values.astype(np.float32)
        self.aspect_cos = df['aspect_cos'].values.astype(np.float32)

        # Shape index: normalize
        si = df['shape_index'].values.astype(np.float64)
        si_min, si_max = si.min(), si.max()
        self.shape_index_norm = ((si - si_min) / (si_max - si_min + 1e-8)).astype(np.float32)

        # Load pre-computed adjacency
        print(f"Loading adjacency: {adjacency_npz}")
        self.adjacency = load_adjacency(adjacency_npz)
        assert len(self.adjacency) == self.n_parcels, \
            f"Adjacency size {len(self.adjacency)} != {self.n_parcels} parcels"

        # Total neighbor count per parcel
        self.total_nbr_count = np.array(
            [len(self.adjacency[i]) for i in range(self.n_parcels)], dtype=np.float32
        )

        # Pre-compute static per-parcel features for swappable parcels
        si_idx = self.swappable_indices
        self._static_slopes_norm = self.slopes_norm[si_idx].copy()
        self._static_areas_norm = self.areas_norm[si_idx].copy()
        self._static_elevation_norm = self.elevation_norm[si_idx].copy()
        self._static_aspect_sin = self.aspect_sin[si_idx].copy()
        self._static_aspect_cos = self.aspect_cos[si_idx].copy()
        self._static_shape_index_norm = self.shape_index_norm[si_idx].copy()

        # Pre-compute neighbor average slope (static)
        self._nbr_avg_slope_norm = np.zeros(self.n_parcels, dtype=np.float32)
        for i in range(self.n_parcels):
            nbrs = self.adjacency[i]
            if len(nbrs) > 0:
                self._nbr_avg_slope_norm[i] = self.slopes_norm[nbrs].mean()
        self._static_nbr_avg_slope = self._nbr_avg_slope_norm[si_idx].copy()

        # Pre-compute initial metrics
        self.land_use = self.initial_types.copy()
        self._compute_metrics_full()
        self._cache = {
            'n_farmland': self.n_farmland,
            'n_forest': self.n_forest,
            'total_farmland_slope': self.total_farmland_slope,
            'farmland_nbr_count': self.farmland_nbr_count.copy(),
            'total_farmland_adj': self.total_farmland_adj,
        }

        # Define spaces
        self.action_space = spaces.Discrete(self.n_swappable)
        obs_dim = self.n_swappable * self.k_parcel + self.k_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Print summary
        init_slope = self._cache['total_farmland_slope'] / max(self._cache['n_farmland'], 1)
        init_cont = self._cache['total_farmland_adj'] / max(self._cache['n_farmland'], 1)
        print(f"Environment initialized (real data):")
        print(f"  Total parcels: {self.n_parcels:,}")
        print(f"  Swappable: {self.n_swappable:,} "
              f"(farmland={self._cache['n_farmland']:,}, forest={self._cache['n_forest']:,})")
        print(f"  Initial avg farmland slope: {init_slope:.4f}")
        print(f"  Initial farmland contiguity: {init_cont:.4f}")
        print(f"  Per-parcel features: {self.k_parcel}, Global features: {self.k_global}")
        print(f"  Observation dim: {obs_dim:,}, Action dim: {self.n_swappable:,}")
        print(f"  Max steps/episode: {self.max_steps}")
        print(f"  Area range: {self.areas.min():.0f} - {self.areas.max():.0f} m2")

        # Track converted parcels (prevents undo within episode)
        self._converted = np.zeros(self.n_swappable, dtype=bool)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics_full(self):
        """Compute all metrics from scratch (used once at init)."""
        fm = self.land_use == FARMLAND
        self.n_farmland = int(fm.sum())
        self.n_forest = int((self.land_use == FOREST).sum())
        self.total_farmland_slope = float(self.slopes[fm].sum())

        self.farmland_nbr_count = np.zeros(self.n_parcels, dtype=np.int32)
        for i in range(self.n_parcels):
            nbrs = self.adjacency[i]
            if len(nbrs) > 0:
                self.farmland_nbr_count[i] = int((self.land_use[nbrs] == FARMLAND).sum())

        self.total_farmland_adj = int(self.farmland_nbr_count[fm].sum())

    @property
    def avg_farmland_slope(self):
        return self.total_farmland_slope / max(self.n_farmland, 1)

    @property
    def contiguity(self):
        return self.total_farmland_adj / max(self.n_farmland, 1)

    # ------------------------------------------------------------------
    # Incremental swap updates (same as v7)
    # ------------------------------------------------------------------

    def _swap_to_forest(self, k):
        """Convert parcel k: farmland -> forest. Update metrics incrementally."""
        self.total_farmland_adj -= self.farmland_nbr_count[k]
        self.total_farmland_slope -= self.slopes[k]

        self.land_use[k] = FOREST
        self.n_farmland -= 1
        self.n_forest += 1

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] -= 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj -= 1

    def _swap_to_farmland(self, k):
        """Convert parcel k: forest -> farmland. Update metrics incrementally."""
        self.land_use[k] = FARMLAND
        self.n_farmland += 1
        self.n_forest -= 1
        self.total_farmland_slope += self.slopes[k]

        self.total_farmland_adj += self.farmland_nbr_count[k]

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] += 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj += 1

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build observation: per-parcel features (N*K) + global features (G).

        K_PARCEL=10 features per swappable parcel:
          [0] slope_mean_norm     -- normalized slope
          [1] type_binary         -- 1.0 if farmland, 0.0 if forest (dynamic)
          [2] nbr_farmland_ratio  -- fraction of neighbors that are farmland (dynamic)
          [3] nbr_avg_slope       -- average slope of neighbors (static)
          [4] area_norm           -- normalized parcel area
          [5] slope_vs_mean       -- (parcel_slope - avg_farmland_slope) / |avg| (dynamic)
          [6] elevation_norm      -- normalized elevation (static)
          [7] aspect_sin          -- sin(aspect) (static)
          [8] aspect_cos          -- cos(aspect) (static)
          [9] shape_index_norm    -- normalized shape irregularity (static)
        """
        si = self.swappable_indices
        avg_sl = self.avg_farmland_slope

        # Per-parcel features
        f_slope = self._static_slopes_norm
        f_type = (self.land_use[si] == FARMLAND).astype(np.float32)
        f_nbr_ratio = self.farmland_nbr_count[si].astype(np.float32) / np.maximum(self.total_nbr_count[si], 1.0)
        f_nbr_slope = self._static_nbr_avg_slope
        f_area = self._static_areas_norm
        f_slope_vs = ((self.slopes[si] - avg_sl) / (abs(avg_sl) + 1e-8)).astype(np.float32)
        f_elev = self._static_elevation_norm
        f_asp_sin = self._static_aspect_sin
        f_asp_cos = self._static_aspect_cos
        f_shape = self._static_shape_index_norm

        per_parcel = np.column_stack([
            f_slope, f_type, f_nbr_ratio, f_nbr_slope, f_area, f_slope_vs,
            f_elev, f_asp_sin, f_asp_cos, f_shape,
        ])

        # Global features (K_GLOBAL=8, same as v7)
        cont = self.contiguity
        farmland_dev = (self.n_farmland - self.initial_n_farmland_count) / self.initial_n_farmland_count

        if self.enforce_pairs:
            phase_indicator = float(self.step_count % 2)
        else:
            phase_indicator = farmland_dev

        global_f = np.array([
            (avg_sl - self.slope_min) / self.slope_range,
            cont / 10.0,
            phase_indicator,
            self.step_count / self.max_steps,
            self.n_farmland / self.n_parcels,
            self.n_forest / self.n_parcels,
            (avg_sl - self.initial_avg_slope) / (abs(self.initial_avg_slope) + 1e-8),
            (cont - self.initial_contiguity) / (abs(self.initial_contiguity) + 1e-8),
        ], dtype=np.float32)

        return np.concatenate([per_parcel.ravel(), global_f])

    def action_masks(self):
        """Return boolean mask of valid actions."""
        si = self.swappable_indices
        unconverted = ~self._converted

        if self.enforce_pairs:
            if self.step_count % 2 == 0:
                mask = (self.land_use[si] == FARMLAND) & unconverted
            else:
                mask = (self.land_use[si] == FOREST) & unconverted
        else:
            mask = ((self.land_use[si] == FARMLAND) | (self.land_use[si] == FOREST)) & unconverted

        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.land_use = self.initial_types.copy()
        self.n_farmland = self._cache['n_farmland']
        self.n_forest = self._cache['n_forest']
        self.total_farmland_slope = self._cache['total_farmland_slope']
        self.farmland_nbr_count = self._cache['farmland_nbr_count'].copy()
        self.total_farmland_adj = self._cache['total_farmland_adj']

        self.step_count = 0
        self.completed_conversions = 0
        self.completed_pairs = 0
        self._converted[:] = False

        self.initial_avg_slope = self.avg_farmland_slope
        self.initial_contiguity = self.contiguity
        self.initial_n_farmland_count = self.n_farmland
        self.prev_avg_slope = self.initial_avg_slope
        self.prev_contiguity = self.initial_contiguity

        info = {
            'avg_slope': self.avg_farmland_slope,
            'contiguity': self.contiguity,
            'completed_conversions': 0,
            'completed_pairs': 0,
            'farmland_change': 0,
        }
        return self._get_obs(), info

    def step(self, action):
        action = int(action)
        parcel_idx = self.swappable_indices[action]

        self._converted[action] = True

        if self.land_use[parcel_idx] == FARMLAND:
            self._swap_to_forest(parcel_idx)
        else:
            self._swap_to_farmland(parcel_idx)

        self.step_count += 1
        self.completed_conversions += 1

        # Compute reward
        avg_sl = self.avg_farmland_slope
        cont = self.contiguity

        slope_r = (self.prev_avg_slope - avg_sl) / (abs(self.initial_avg_slope) + 1e-8)
        cont_r = (cont - self.prev_contiguity) / (abs(self.initial_contiguity) + 1e-8)
        count_dev = abs(self.n_farmland - self.initial_n_farmland_count) / self.initial_n_farmland_count

        reward = (self.slope_reward_weight * slope_r
                  + self.cont_reward_weight * cont_r
                  - self.count_penalty_weight * count_dev * count_dev)

        if self.n_farmland == self.initial_n_farmland_count:
            reward += PAIR_BONUS
            self.completed_pairs += 1

        self.prev_avg_slope = avg_sl
        self.prev_contiguity = cont

        terminated = self.step_count >= self.max_steps
        if not terminated:
            mask = self.action_masks()
            if not mask.any():
                terminated = True

        farmland_change = self.n_farmland - self.initial_n_farmland_count
        info = {
            'avg_slope': self.avg_farmland_slope,
            'contiguity': self.contiguity,
            'completed_conversions': self.completed_conversions,
            'completed_pairs': self.completed_pairs,
            'farmland_change': farmland_change,
            'slope_r': float(slope_r),
            'cont_r': float(cont_r),
            'count_dev2': float(count_dev * count_dev),
        }

        return self._get_obs(), float(reward), terminated, False, info
