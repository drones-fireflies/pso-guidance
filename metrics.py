"""
metrics.py - Metrics for multi-UAV wildfire guidance
====================================================
"""

import numpy as np

from typing import Optional
from dataclasses import dataclass

@dataclass
class CoverageMetrics:
    step: int

    # coverage
    frontier_cells_total:    int
    frontier_cells_covered:  int
    coverage_rate:           float

    # overlap
    overlap_cells: int
    overlap_rate:  float

    # frontier IoU 
    frontier_iou: float

    # fire body IoU 
    fire_iou: float

    # decision latency
    decision_latency_ms: float

    def __str__(self) -> str:
        f_iou = (f"frontIoU={self.frontier_iou:5.3f}"
                 if self.frontier_iou >= 0 else "frontIoU=  n/a")
        r_iou = (f"fireIoU={self.fire_iou:5.3f}"
                 if self.fire_iou >= 0 else "fireIoU=  n/a")
        lat   = (f"lat={self.decision_latency_ms:6.2f}ms"
                 if self.decision_latency_ms >= 0 else "lat=   n/a")
        return (
            f"step={self.step:4d} | "
            f"ovlp={self.overlap_rate*100:4.1f}% | "
            f"{f_iou} | {r_iou} | {lat}"
        )


def _iou(A: np.ndarray, B: np.ndarray) -> float:
    """Pixel-level IoU of two bool masks. Returns -1.0 if union is empty."""
    inter = int((A & B).sum())
    union = int((A | B).sum())
    return -1.0 if union == 0 else float(inter) / float(union)


def _frontier_iou(
    true_frontier:   Optional[np.ndarray],
    belief_frontier: np.ndarray,
) -> float:
    if true_frontier is None:
        return -1.0
    return _iou(true_frontier > 0, belief_frontier > 0)


def _fire_iou(
    true_fire:   Optional[np.ndarray],
    fire_belief: np.ndarray,
) -> float:
    """
    IoU of the burning area: cells where fire > 0.5 in truth vs belief.
    Returns -1.0 when true_fire is not provided or both masks are empty.
    """
    if true_fire is None:
        return -1.0
    return _iou(true_fire > 0.5, fire_belief > 0.5)

def compute_metrics(
    step:                int,
    drone_positions:     np.ndarray,
    frontier:            np.ndarray,
    fire_belief:         np.ndarray,
    sensor_radius:       int,
    true_frontier:       Optional[np.ndarray] = None,
    true_fire:           Optional[np.ndarray] = None,
    decision_latency_ms: float                = -1.0,
) -> CoverageMetrics:
    H, W      = frontier.shape
    front_bin = frontier > 0
    front_pts = np.argwhere(front_bin)
    total     = int(front_bin.sum())

    # coverage & overlap
    coverage_map = np.zeros((H, W), dtype=np.int32)
    rr, cc = np.ogrid[:H, :W]
    for pos in drone_positions:
        mask = (rr - pos[0])**2 + (cc - pos[1])**2 <= sensor_radius**2
        coverage_map += mask.astype(np.int32)

    covered_mask = (coverage_map > 0) & front_bin
    overlap_mask = (coverage_map > 1) & front_bin
    covered  = int(covered_mask.sum())
    overlap  = int(overlap_mask.sum())
    cov_rate = covered / total   if total   > 0 else 0.0
    ovl_rate = overlap / covered if covered > 0 else 0.0

    return CoverageMetrics(
        step                         = step,
        frontier_cells_total         = total,
        frontier_cells_covered       = covered,
        coverage_rate                = cov_rate,
        overlap_cells                = overlap,
        overlap_rate                 = ovl_rate,
        frontier_iou                 = _frontier_iou(true_frontier, frontier),
        fire_iou                     = _fire_iou(true_fire, fire_belief),
        decision_latency_ms          = decision_latency_ms,
    )