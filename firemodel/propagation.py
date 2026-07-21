import logging
from typing import List, Tuple

import numpy as np
import scipy.signal

from guidance_pso.viz.fire_display import Display

logger = logging.getLogger(__name__)

# -------------------- Constants --------------------
ELEVATION_SMOOTH_KERNEL = (15, 15)        # size of smoothing kernel for elevation
MOISTURE_SMOOTH_KERNEL  = (25, 25)        # size of smoothing kernel for moisture
SLOPE_WINDOW = (3, 3)                     # local neighborhood for slope effect
SLOPE_GAIN = 1.0                          # linear scaling factor for slope effect
SLOPE_CLIP = (0.5, 1.5)                   # clamp slope effect for stability

MOISTURE_MEAN = 0.30                      # target average moisture in the domain
MOISTURE_IGNITION_WEIGHT = 1.0            # how strongly moisture reduces ignition influence
MOISTURE_BURN_FACTOR     = 0.50           # how strongly moisture slows fuel consumption

BURN_RATE_RANGE = (0.5, 5.0)             # (min, max) burn rate per timestep per cell
CONTOUR_LEVELS = 12                       # number of contour levels for plotting
RNG = np.random.default_rng()             # reproducible RNG
# -----------------------------------------------------------


class Environment:
    """Environment for fire propagation model."""

    def __init__(
        self,
        grid_size: int,
        ignition_points: List[Tuple[int, int]],
        wind_direction: float,
        wind_strength: float = 1.0,
        vegetation_density: float = 0.95,
    ) -> None:
        self.grid_size_x = grid_size
        self.grid_size_y = grid_size
        self.ignition_points = ignition_points
        self.wind_strength = float(wind_strength)
        self.vegetation_density = float(vegetation_density)
        self.wind_direction = float(wind_direction)

        # Initialize maps
        self.fire_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._ignite_initial_points()
        self.fuel_map = self._generate_fuel_map()
        self.elevation_map = self._generate_elevation_map()
        self.moisture_map = self._generate_moisture_map()
        self.burn_rate_map = self._generate_burn_rate_map()

    # -------------------- maps generation --------------------

    def _ignite_initial_points(self) -> None:
        for x, y in self.ignition_points:
            if 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y:
                self.fire_map[x, y] = 1.0
            else:
                logger.warning("Ignition point (%d,%d) out of bounds; ignored.", x, y)

    def _generate_fuel_map(self) -> np.ndarray:
        fuel = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.float32)
        num_vegetations = int(self.grid_size_x * self.grid_size_y * self.vegetation_density)
        tx = RNG.integers(0, self.grid_size_x, num_vegetations)
        ty = RNG.integers(0, self.grid_size_y, num_vegetations)
        fuel_vals = RNG.integers(20, 256, num_vegetations, dtype=np.int32)
        fuel[tx, ty] = fuel_vals
        return fuel

    def _generate_elevation_map(self) -> np.ndarray:
        raw = RNG.random((self.grid_size_x, self.grid_size_y), dtype=np.float32)
        kx, ky = ELEVATION_SMOOTH_KERNEL
        kernel = np.ones((kx, ky), dtype=np.float32) / float(kx * ky)
        smooth = scipy.signal.convolve2d(raw, kernel, mode="same", boundary="symm")
        smooth = np.asarray(smooth, dtype=np.float32)
        mn, mx = float(smooth.min()), float(smooth.max())
        return (smooth - mn) / max(1e-12, (mx - mn))

    def _generate_moisture_map(self) -> np.ndarray:
        """Generate a smooth, normalized moisture map in [0, 1]."""
        raw = RNG.random((self.grid_size_x, self.grid_size_y), dtype=np.float32)
        kx, ky = MOISTURE_SMOOTH_KERNEL
        kernel = np.ones((kx, ky), dtype=np.float32) / float(kx * ky)
        smooth = scipy.signal.convolve2d(raw, kernel, mode="same", boundary="symm")
        mn, mx = float(smooth.min()), float(smooth.max())
        norm = (smooth - mn) / max(1e-12, (mx - mn))
        moist = np.clip(MOISTURE_MEAN + 0.4 * (norm - 0.5), 0.0, 1.0)
        return moist.astype(np.float32)

    def _generate_burn_rate_map(self) -> np.ndarray:
        """Compute per-cell burn rate from fuel, moisture, and slope."""
        fmin, fmax = BURN_RATE_RANGE

        fuel_norm = self.fuel_map / max(1e-6, self.fuel_map.max())
        moist = np.clip(self.moisture_map, 0.0, 1.0)

        # Combine effects multiplicatively
        rate_factor = fuel_norm  * (1.0 - 0.6 * moist)
        rate_map = fmin + (fmax - fmin) * np.clip(rate_factor, 0.0, 1.0)
        rate_map[self.fuel_map <= 0] = 0.0
        return rate_map.astype(np.float32)

    @staticmethod
    def _compute_slope_effect(elevation_map: np.ndarray) -> np.ndarray:
        """Compute local slope amplification factor from elevation."""
        kx, ky = SLOPE_WINDOW
        kernel = np.ones((kx, ky), dtype=np.float32) / float(kx * ky)
        local_mean = scipy.signal.convolve2d(
            elevation_map, kernel, mode="same", boundary="symm"
        ).astype(np.float32)
        raw = 1.0 + SLOPE_GAIN * (elevation_map - local_mean)
        slope_effect = np.clip(raw, SLOPE_CLIP[0], SLOPE_CLIP[1]).astype(np.float32)
        return slope_effect

    @classmethod
    def from_random(cls, grid_size: int, ignition_count: int = 1, wind_direction: float = 0.0) -> "Environment":
        ignition_points = [
            (int(RNG.integers(0, grid_size)), int(RNG.integers(0, grid_size)))
            for _ in range(ignition_count)
        ]
        return cls(grid_size, ignition_points, wind_direction)


# -----------------------------------------------------------------------------
class FirePropagation:
    """Deterministic cellular fire spread with wind, slope, fuel, and moisture"""

    neighbor_offsets: List[Tuple[int, int]] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    neighbor_angles: List[float] = [
        3 * np.pi / 4, np.pi / 2, np.pi / 4,
        np.pi, 0.0, 0.0,
        5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4
    ]

    def __init__(self, timesteps: int, visualize: bool = False) -> None:
        self.timesteps = int(timesteps)
        self.visualize = visualize

    def propagate_from_points(self, env: Environment):
        """Run deterministic propagation"""
        fire_state = np.copy(env.fire_map)
        fuel_map = np.copy(env.fuel_map)
        elevation = np.copy(env.elevation_map)
        moisture = np.copy(env.moisture_map)
        burn_rate_map = np.copy(env.burn_rate_map)

        fire_hist = np.zeros((self.timesteps, env.grid_size_x, env.grid_size_y), dtype=np.float32)
        fuel_hist = np.zeros_like(fire_hist)

        if self.visualize:
            viz = Display(env.grid_size_x, env.grid_size_y, env.wind_direction, elevation=elevation)
            fig, imgs = viz._initialize_visualization()
            
        slope_effect = Environment._compute_slope_effect(elevation)

        for t in range(self.timesteps):
            
            fire_state, fuel_map = self._spread_fire(
                fire_state, fuel_map, slope_effect, env.wind_direction, env.wind_strength,
                burn_rate_map, moisture
            )

            fire_hist[t] = fire_state
            fuel_hist[t] = fuel_map

            if self.visualize:
                viz.update(imgs, fire_state, fuel_map)

        return fire_hist, fuel_hist, elevation, moisture

    def _spread_fire(
        self,
        fire_state: np.ndarray,
        fuel_map: np.ndarray,
        slope_effect: np.ndarray,
        wind_dir: float,
        wind_strength: float,
        burn_rate_map: np.ndarray,
        moisture_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply one fire spread step."""
        new_fire = np.array(fire_state, copy=True)
        gx, gy = fire_state.shape

        already_burning = fire_state > 0

        for i in range(1, gx - 1):
            for j in range(1, gy - 1):
                if fuel_map[i, j] <= 0 or fire_state[i, j] > 0:
                    continue

                influence = 0.0
                for (dx, dy), neighbor_angle in zip(self.neighbor_offsets, self.neighbor_angles):
                    ni, nj = i + dx, j + dy
                    if fire_state[ni, nj] > 0:
                        angle_diff = ((neighbor_angle - wind_dir) % (2 * np.pi)) - np.pi
                        wind_factor = wind_strength * max(0.0, np.cos(angle_diff))
                        influence += wind_factor

                moist = float(moisture_map[i, j])
                ignition_effect = influence * slope_effect[i, j] * max(0.0, 1.0 - MOISTURE_IGNITION_WEIGHT * moist)

                if ignition_effect > 0.0:
                    new_fire[i, j] = 1.0
                    fuel_map[i, j] -= 1.0

        if np.any(already_burning):
            local_rate = burn_rate_map[already_burning]
            effective_burn = local_rate * (1.0 - MOISTURE_BURN_FACTOR * moisture_map[already_burning])
            fuel_map[already_burning] -= effective_burn

        fuel_map[fuel_map < 0] = 0
        new_fire[fuel_map == 0] = 0
        return new_fire.astype(np.float32), fuel_map.astype(np.float32)
    

# -------------------- simple demo --------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     env = Environment.from_random(grid_size=200, ignition_count=2, wind_direction=np.pi/4)
#     sim = FirePropagation(timesteps=150)
#     sim.propagate_from_points(env)
