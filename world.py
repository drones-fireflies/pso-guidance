"""
world.py  —  Physical world parameters and coordinate transforms
================================================================

The fire environment is modelled on an N×N grid where each cell
represents CELL_SIZE metres of real terrain.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class WorldParams:
    grid_size:      int   = 200
    cell_size:      float = 5.0      # metres per cell
    speed:          float = 10
    dt:             float = 1.0
    max_yaw_rate:   float = np.deg2rad(15.0)
    fov_width:      float = 50.0     # camera footprint width
    sep_min:        float = 100.0    # minimum inter-drone distance

    @property
    def grid_metres(self) -> float:
        return self.grid_size * self.cell_size

    @property
    def step_metres(self) -> float:
        return self.speed * self.dt

    @property
    def fov_half_cells(self) -> int:
        return max(1, int(round((self.fov_width / 2.0) / self.cell_size)))

    @property
    def step_cells(self) -> float:
        return self.step_metres / self.cell_size

    @property
    def sep_cells(self) -> float:
        return self.sep_min / self.cell_size

    # ── coordinate transforms ─────────────────────────────────────────────

    def m2cell(self, xy: np.ndarray) -> np.ndarray:
        """World metres (x, y) → grid (row, col)."""
        x, y = xy[..., 0], xy[..., 1]
        col = x / self.cell_size
        row = (self.grid_metres - y) / self.cell_size
        return np.stack([row, col], axis=-1)

    def cell2m(self, rc: np.ndarray) -> np.ndarray:
        """Grid (row, col) → world metres (x, y)."""
        row, col = rc[..., 0], rc[..., 1]
        x = col * self.cell_size
        y = self.grid_metres - row * self.cell_size
        return np.stack([x, y], axis=-1)

    def clip_metres(self, xy: np.ndarray) -> np.ndarray:
        """Clip world position to grid bounds."""
        margin = 0.0
        return np.clip(xy,
                       [margin, margin],
                       [self.grid_metres - margin, self.grid_metres - margin])

WORLD = WorldParams()
