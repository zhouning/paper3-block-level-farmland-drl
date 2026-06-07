"""Gymnasium environment for land use optimization with area-weighted constraint.

Extends RealDataLandUseEnv to replace count-based farmland conservation
with area-based conservation. This prevents the problem where swapping
a large farmland parcel out for a tiny forest parcel in causes massive
farmland area loss despite FC=0.

Key changes from land_use_env_real.py:
  - Tracks total_farmland_area (sum of areas for farmland parcels)
  - area_dev replaces count_dev in reward
  - K_GLOBAL becomes 9 (adds area_deviation feature)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from township_utils import FARMLAND, FOREST, OTHER
from build_adjacency import load_adjacency


# Per-parcel feature count (same as real env)
K_PARCEL = 10
# Global feature count (expanded from 8 to 9 with area deviation)
K_GLOBAL = 9

# Reward weights
SLOPE_REWARD_WEIGHT = 1000.0
CONT_REWARD_WEIGHT = 500.0
AREA_PENALTY_WEIGHT = 500.0    # area deviation penalty (replaces count penalty)
PAIR_BONUS = 1.0


class RealDataLandUseEnvV2(gym.Env):
    """Land use optimization environment with area-weighted constraint.

    Same as RealDataLandUseEnv except:
    - Penalty is on AREA deviation, not count deviation
    - Tracks total_farmland_area for area conservation
    - K_GLOBAL=9 (adds area_deviation as global feature)
    """

    metadata = {"render_modes": []}

    def __init__(self, features_csv, adjacency_npz,
                 max_conversions=200,
                 area_penalty_weight=None,
                 enforce_pairs=False,
                 k_parcel=None,
                 k_global=None):
        super().__init__()

        self.area_penalty_weight = area_penalty_weight or AREA_PENALTY_WEIGHT
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
        self.initial_types = df['type_code'].values.astype(np.int8)

        # Swappable parcels
        self.swappable_indices = np.where(
            (self.initial_types == FARMLAND) | (self.initial_types == FOREST)
        )[0]
        self.n_swappable = len(self.swappable_indices)

        # Normalize slopes
        self.slope_min = float(self.slopes.min())
        self.slope_max_val = float(self.slopes.max())
        self.slope_range = self.slope_max_val - self.slope_min + 1e-8
        self.slopes_norm = ((self.slopes - self.slope_min) / self.slope_range).astype(np.float32)

        # Normalize areas
        area_min = float(self.areas.min())
        area_max = float(self.areas.max())
        area_range = area_max - area_min + 1e-8
        self.areas_norm = ((self.areas - area_min) / area_range).astype(np.float32)

        # Extended features
        elev = df['elevation_mean'].values.astype(np.float64)
        elev_min, elev_max = elev.min(), elev.max()
        self.elevation_norm = ((elev - elev_min) / (elev_max - elev_min + 1e-8)).astype(np.float32)
        self.aspect_sin = df['aspect_sin'].values.astype(np.float32)
        self.aspect_cos = df['aspect_cos'].values.astype(np.float32)
        si = df['shape_index'].values.astype(np.float64)
        si_min, si_max = si.min(), si.max()
        self.shape_index_norm = ((si - si_min) / (si_max - si_min + 1e-8)).astype(np.float32)

        # Load adjacency
        print(f"Loading adjacency: {adjacency_npz}")
        self.adjacency = load_adjacency(adjacency_npz)

        # Neighbor counts
        self.total_nbr_count = np.array(
            [len(self.adjacency[i]) for i in range(self.n_parcels)], dtype=np.float32
        )

        # Pre-compute static features
        si_idx = self.swappable_indices
        self._static_slopes_norm = self.slopes_norm[si_idx].copy()
        self._static_areas_norm = self.areas_norm[si_idx].copy()
        self._static_elevation_norm = self.elevation_norm[si_idx].copy()
        self._static_aspect_sin = self.aspect_sin[si_idx].copy()
        self._static_aspect_cos = self.aspect_cos[si_idx].copy()
        self._static_shape_index_norm = self.shape_index_norm[si_idx].copy()

        self._nbr_avg_slope_norm = np.zeros(self.n_parcels, dtype=np.float32)
        for i in range(self.n_parcels):
            nbrs = self.adjacency[i]
            if len(nbrs) > 0:
                self._nbr_avg_slope_norm[i] = self.slopes_norm[nbrs].mean()
        self._static_nbr_avg_slope = self._nbr_avg_slope_norm[si_idx].copy()

        # Compute initial metrics including total farmland area
        self.land_use = self.initial_types.copy()
        self._compute_metrics_full()
        self._cache = {
            'n_farmland': self.n_farmland,
            'n_forest': self.n_forest,
            'total_farmland_slope': self.total_farmland_slope,
            'farmland_nbr_count': self.farmland_nbr_count.copy(),
            'total_farmland_adj': self.total_farmland_adj,
            'total_farmland_area': self.total_farmland_area,
        }

        # Define spaces
        self.action_space = spaces.Discrete(self.n_swappable)
        obs_dim = self.n_swappable * self.k_parcel + self.k_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Summary
        init_slope = self._cache['total_farmland_slope'] / max(self._cache['n_farmland'], 1)
        init_cont = self._cache['total_farmland_adj'] / max(self._cache['n_farmland'], 1)
        init_area_ha = self._cache['total_farmland_area'] / 10000
        print(f"Environment initialized (real data, area-weighted):")
        print(f"  Total parcels: {self.n_parcels:,}")
        print(f"  Swappable: {self.n_swappable:,} "
              f"(farmland={self._cache['n_farmland']:,}, forest={self._cache['n_forest']:,})")
        print(f"  Initial avg farmland slope: {init_slope:.4f}")
        print(f"  Initial farmland contiguity: {init_cont:.4f}")
        print(f"  Initial farmland area: {init_area_ha:.1f} ha")
        print(f"  Per-parcel features: {self.k_parcel}, Global features: {self.k_global}")
        print(f"  Observation dim: {obs_dim:,}, Action dim: {self.n_swappable:,}")

        self._converted = np.zeros(self.n_swappable, dtype=bool)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics_full(self):
        fm = self.land_use == FARMLAND
        self.n_farmland = int(fm.sum())
        self.n_forest = int((self.land_use == FOREST).sum())
        self.total_farmland_slope = float(self.slopes[fm].sum())
        self.total_farmland_area = float(self.areas[fm].sum())

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

    @property
    def area_deviation(self):
        """Fractional change in total farmland area from initial."""
        return abs(self.total_farmland_area - self.initial_total_farmland_area) / \
               (self.initial_total_farmland_area + 1e-8)

    # ------------------------------------------------------------------
    # Incremental swap updates (with area tracking)
    # ------------------------------------------------------------------

    def _swap_to_forest(self, k):
        self.total_farmland_adj -= self.farmland_nbr_count[k]
        self.total_farmland_slope -= self.slopes[k]
        self.total_farmland_area -= self.areas[k]  # NEW: area tracking

        self.land_use[k] = FOREST
        self.n_farmland -= 1
        self.n_forest += 1

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] -= 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj -= 1

    def _swap_to_farmland(self, k):
        self.land_use[k] = FARMLAND
        self.n_farmland += 1
        self.n_forest -= 1
        self.total_farmland_slope += self.slopes[k]
        self.total_farmland_area += self.areas[k]  # NEW: area tracking

        self.total_farmland_adj += self.farmland_nbr_count[k]

        for j in self.adjacency[k]:
            self.farmland_nbr_count[j] += 1
            if self.land_use[j] == FARMLAND:
                self.total_farmland_adj += 1

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self):
        si = self.swappable_indices
        avg_sl = self.avg_farmland_slope

        # Per-parcel features (K_PARCEL=10)
        per_parcel = np.column_stack([
            self._static_slopes_norm,
            (self.land_use[si] == FARMLAND).astype(np.float32),
            self.farmland_nbr_count[si].astype(np.float32) / np.maximum(self.total_nbr_count[si], 1.0),
            self._static_nbr_avg_slope,
            self._static_areas_norm,
            ((self.slopes[si] - avg_sl) / (abs(avg_sl) + 1e-8)).astype(np.float32),
            self._static_elevation_norm,
            self._static_aspect_sin,
            self._static_aspect_cos,
            self._static_shape_index_norm,
        ])

        # Global features (K_GLOBAL=9, one more than v7)
        cont = self.contiguity
        area_dev = self.area_deviation

        if self.enforce_pairs:
            phase_indicator = float(self.step_count % 2)
        else:
            area_change_signed = (self.total_farmland_area - self.initial_total_farmland_area) / \
                                 (self.initial_total_farmland_area + 1e-8)
            phase_indicator = area_change_signed

        global_f = np.array([
            (avg_sl - self.slope_min) / self.slope_range,
            cont / 10.0,
            phase_indicator,
            self.step_count / self.max_steps,
            self.n_farmland / self.n_parcels,
            self.n_forest / self.n_parcels,
            (avg_sl - self.initial_avg_slope) / (abs(self.initial_avg_slope) + 1e-8),
            (cont - self.initial_contiguity) / (abs(self.initial_contiguity) + 1e-8),
            area_dev,  # NEW: area deviation feature
        ], dtype=np.float32)

        return np.concatenate([per_parcel.ravel(), global_f])

    def action_masks(self):
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
        self.total_farmland_area = self._cache['total_farmland_area']

        self.step_count = 0
        self.completed_conversions = 0
        self.completed_pairs = 0
        self._converted[:] = False

        self.initial_avg_slope = self.avg_farmland_slope
        self.initial_contiguity = self.contiguity
        self.initial_n_farmland_count = self.n_farmland
        self.initial_total_farmland_area = self.total_farmland_area
        self.prev_avg_slope = self.initial_avg_slope
        self.prev_contiguity = self.initial_contiguity

        info = {
            'avg_slope': self.avg_farmland_slope,
            'contiguity': self.contiguity,
            'farmland_area': self.total_farmland_area,
            'area_deviation': 0.0,
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

        avg_sl = self.avg_farmland_slope
        cont = self.contiguity

        slope_r = (self.prev_avg_slope - avg_sl) / (abs(self.initial_avg_slope) + 1e-8)
        cont_r = (cont - self.prev_contiguity) / (abs(self.initial_contiguity) + 1e-8)

        # Area-based penalty (replaces count-based penalty)
        area_dev = self.area_deviation
        reward = (SLOPE_REWARD_WEIGHT * slope_r
                  + CONT_REWARD_WEIGHT * cont_r
                  - self.area_penalty_weight * area_dev * area_dev)

        # Area-scaled pair bonus (diminishes as area deviation grows)
        if self.n_farmland == self.initial_n_farmland_count:
            bonus = PAIR_BONUS * max(0.0, 1.0 - area_dev * 10)
            reward += bonus
            self.completed_pairs += 1

        self.prev_avg_slope = avg_sl
        self.prev_contiguity = cont

        terminated = self.step_count >= self.max_steps
        if not terminated:
            mask = self.action_masks()
            if not mask.any():
                terminated = True

        farmland_change = self.n_farmland - self.initial_n_farmland_count
        area_change_m2 = self.total_farmland_area - self.initial_total_farmland_area

        info = {
            'avg_slope': self.avg_farmland_slope,
            'contiguity': self.contiguity,
            'completed_conversions': self.completed_conversions,
            'completed_pairs': self.completed_pairs,
            'farmland_change': farmland_change,
            'farmland_area': self.total_farmland_area,
            'area_deviation': float(area_dev),
            'area_change_m2': float(area_change_m2),
        }

        return self._get_obs(), float(reward), terminated, False, info
