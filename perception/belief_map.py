"""
Belief state shared across all drones.

Drones start with ZERO knowledge of where the fire is.
As each drone flies and observes within its square FoV, 
it corrects the belief map with the true fire values it sees.
"""

import numpy as np

from guidance_pso.perception.front_detection import cv_frontier, frontier_cells


class BeliefMaps:
    """

    Attributes
    ----------
    fire_belief : (H,W) float32 [0,1]   estimated fire map
    frontier    : (H,W) uint8  {0,255}  estimated frontier from fire_belief
    front_pts   : (K,2) int             frontier pixel coordinates [row, col]
    """

    def __init__(self, H: int, W: int):
        self.H = H
        self.W = W

        self.fire_belief = np.zeros((H, W), dtype=np.float32)
        self.frontier    = np.zeros((H, W), dtype=np.uint8)
        self.front_pts   = np.empty((0, 2), dtype=int)


    def observe(self, true_fire: np.ndarray, sensor_mask: np.ndarray):
        """Update belief with what drones actually see."""

        self.fire_belief[sensor_mask] = true_fire[sensor_mask]

        # Extract frontier from updated belief fire map
        fire_u8 = (self.fire_belief * 255).astype(np.uint8)
        self.frontier  = cv_frontier(fire_u8)
        self.front_pts = frontier_cells(fire_u8)