"""
pso_agents.py  -  Velocity space PSO for frontier coverage and tracking
=======================================================================
"""

import time
import numpy as np

from guidance_pso.world import WORLD
from typing import List, Optional


_W_INERTIA = 0.60
_C1        = 1.50
_C2        = 1.50

_W_GAIN    = 1.0
_W_SCOUT   = 0.35
_W_ATTRACT = 0.4
_W_OVERLAP = 0.25
_W_BDRY    = 0.04

_LEASH_M     = 60.0
_SCOUT_CELLS = 6


class PSOAgent:
    """
    Per-drone decentralised PSO over velocity space (vx, vy).

    Parameters
    ----------
    n_particles   : population size
    n_iterations  : iterations per call
    sensor_radius : FoV half-side in grid cells
    world         : WorldParams
    """

    def __init__(
        self,
        n_particles:   int   = 40,
        n_iterations:  int   = 30,
        sensor_radius: int   = 5,
        world                = WORLD,
        lambda_gain:    float = _W_GAIN,
        lambda_scout:   float = _W_SCOUT,
        lambda_attract: float = _W_ATTRACT,
        lambda_overlap: float = _W_OVERLAP,
        lambda_boundary: float = _W_BDRY,
        leash_metres:  float = _LEASH_M,
        scout_cells:   int   = _SCOUT_CELLS,
        w: float = _W_INERTIA, c1: float = _C1, c2: float = _C2,
    ):
        self.W    = world
        self.P    = n_particles
        self.K    = n_iterations
        self.rho  = sensor_radius
        self.w    = w; self.c1 = c1; self.c2 = c2
        self.lg   = lambda_gain
        self.ls   = lambda_scout
        self.la   = lambda_attract
        self.lo   = lambda_overlap
        self.lb   = lambda_boundary
        self.leash       = leash_metres
        self.scout_cells = scout_cells

        self.times_ms: List[float] = []
        self._last_vx:    float = 0.0
        self._last_vy:    float = 0.0
        self._last_heading: Optional[float] = None   # smoothed heading (rad)
        self._last_sector_cen_m: Optional[np.ndarray] = None

        # Max heading change per step — physical yaw rate limit
        # 15 deg/s gives smooth, readable flight at 1 Hz control loop
        self._max_dheading = np.deg2rad(15.0)   # rad/step

    # ── candidate positions ───────────────────────────────────────────────────

    def _candidate_positions(
        self,
        pos_m:      np.ndarray,   # (2,) [x_east, y_north] metres
        vx_arr:     np.ndarray,   # (P,)
        vy_arr:     np.ndarray,   # (P,)
    ) -> np.ndarray:              # (P, 2) [row, col] grid
        """Compute candidate next grid positions from velocity particles."""
        dt = self.W.dt
        xs = np.clip(pos_m[0] + vx_arr * dt, 0.0, self.W.grid_metres)
        ys = np.clip(pos_m[1] + vy_arr * dt, 0.0, self.W.grid_metres)
        xy = np.stack([xs, ys], axis=1)   # (P, 2) metres
        rc = self.W.m2cell(xy)            # (P, 2) [row, col]
        rc[:, 0] = np.clip(rc[:, 0], 0, self.W.grid_size - 1)
        rc[:, 1] = np.clip(rc[:, 1], 0, self.W.grid_size - 1)
        return rc

    # ── fitness ───────────────────────────────────────────────────────────────

    def _fitness(
        self,
        cand_rc:        np.ndarray,   # (P, 2) [row, col]
        frontier:       np.ndarray,   # (H, W) uint8
        pre_frontier:   np.ndarray,   # (H, W) bool — scout buffer ahead of fire
        sector_cen_m:   Optional[np.ndarray],
        neighbour_pos_m: np.ndarray,
        neighbour_fovs:  np.ndarray,
    ) -> np.ndarray:
        rho  = self.rho
        slab = 2 * rho + 1
        H, W = frontier.shape
        P    = len(cand_rc)

        ri = cand_rc[:, 0].astype(int)
        ci = cand_rc[:, 1].astype(int)

        # 1. exclusive frontier gain
        excl = (frontier > 0) & (~neighbour_fovs)
        pad  = np.pad(excl.astype(np.float32), rho)
        norm = float(slab * slab) + 1e-9
        fit_gain = np.array([
            pad[r : r + slab, c : c + slab].sum() / norm
            for r, c in zip(ri, ci)
        ], dtype=np.float64)

        # 2. scout: pre-frontier buffer — cells just outside the current fire edge
        # Rewards drones for positioning to see where the fire will spread next.
        # Only counts cells not already covered by a neighbour's FoV.
        excl_scout = pre_frontier & (~neighbour_fovs)
        pad_scout  = np.pad(excl_scout.astype(np.float32), rho)
        fit_scout  = np.array([
            pad_scout[r : r + slab, c : c + slab].sum() / norm
            for r, c in zip(ri, ci)
        ], dtype=np.float64)

        # 3. sector centroid attraction
        if sector_cen_m is not None:
            cand_m = self.W.cell2m(cand_rc)
            diag   = self.W.grid_metres * np.sqrt(2) + 1e-9
            dist   = np.linalg.norm(cand_m - sector_cen_m, axis=1)
            fit_attract = 1.0 / (1.0 + dist / diag)
        else:
            fit_attract = np.zeros(P)

        # 4. separation
        sep_min  = self.W.sep_min
        fit_ovlp = np.zeros(P)
        if len(neighbour_pos_m):
            cand_m = self.W.cell2m(cand_rc)
            for nb in neighbour_pos_m:
                d = np.linalg.norm(cand_m - nb, axis=1)
                fit_ovlp += np.clip(sep_min - d, 0.0, None) ** 2

        # 5. boundary repulsion
        margin_m = self.rho * self.W.cell_size
        cand_m   = self.W.cell2m(cand_rc)
        fit_bdry = (
            np.maximum(0.0, margin_m - cand_m[:, 0]) +
            np.maximum(0.0, cand_m[:, 0] - (self.W.grid_metres - margin_m)) +
            np.maximum(0.0, margin_m - cand_m[:, 1]) +
            np.maximum(0.0, cand_m[:, 1] - (self.W.grid_metres - margin_m))
        )

        return (  self.lg * fit_gain
                + self.ls * fit_scout
                + self.la * fit_attract
                - self.lo * fit_ovlp
                - self.lb * fit_bdry )

    # ── neighbour FoV ─────────────────────────────────────────────────────────

    def _neighbour_fov_mask(self, neighbour_pos_m: np.ndarray) -> np.ndarray:
        """Bool (H,W) — union of all neighbour square FoVs."""
        H = W = self.W.grid_size
        mask = np.zeros((H, W), dtype=bool)
        rho  = self.rho
        for pos in neighbour_pos_m:
            rc = self.W.m2cell(pos)
            r  = int(np.clip(rc[0], 0, H - 1))
            c  = int(np.clip(rc[1], 0, W - 1))
            mask[max(0,r-rho):min(H,r+rho+1),
                 max(0,c-rho):min(W,c+rho+1)] = True
        return mask

    # ── public interface ──────────────────────────────────────────────────────

    def _pre_frontier_buffer(
        self,
        frontier:   np.ndarray,   # (H,W) uint8 — belief frontier
        fire_belief: np.ndarray,  # (H,W) float32 [0,1]
    ) -> np.ndarray:              # (H,W) bool
        """
        Pre-frontier buffer: cells within scout_cells of the frontier edge
        that are NOT currently burning.  These are the cells the fire will
        reach next — high future information value.

        Built by binary dilation of the frontier mask, then masking out
        cells where fire_belief > 0.3 (already observed as burning).
        """
        from scipy.ndimage import binary_dilation
        front_bin  = frontier > 0
        struct     = np.ones((2*self.scout_cells+1, 2*self.scout_cells+1), dtype=bool)
        dilated    = binary_dilation(front_bin, structure=struct)
        not_fire   = fire_belief < 0.3
        return dilated & not_fire & (~front_bin)   # outside frontier, not burning

    def compute_velocity(
        self,
        pos_m:           np.ndarray,
        frontier:        np.ndarray,
        front_pts:       np.ndarray,
        neighbour_pos_m: np.ndarray,
        sector_cen_m:    Optional[np.ndarray] = None,
        fire_belief:     Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Return (vx, vy) in m/s — EMA-normalised to airspeed.
        If outside leash, returns full-speed return velocity (no PSO).
        """
        t0 = time.perf_counter()
        V  = self.W.speed

        # ── adaptive leash ────────────────────────────────────────────────
        # Scale leash with frontier radius so drones can chase a growing fire.
        # If frontier radius > base leash, allow up to 1.5x frontier radius.
        if sector_cen_m is not None and len(front_pts) > 0:
            # frontier radius = mean dist from centroid to frontier pts in metres
            cen_rc  = front_pts.mean(axis=0)
            cen_m_f = self.W.cell2m(cen_rc)
            fp_m    = self.W.cell2m(front_pts)   # (K,2) metres
            front_radius_m = float(np.linalg.norm(fp_m - cen_m_f, axis=1).mean())
            effective_leash = max(self.leash, front_radius_m * 1.4)
        else:
            effective_leash = self.leash

        # Reset EMA if sector centroid jumped far (frontier grew quickly)
        if sector_cen_m is not None and self._last_sector_cen_m is not None:
            sector_jump = float(np.linalg.norm(sector_cen_m - self._last_sector_cen_m))
            if sector_jump > self.W.speed * self.W.dt * 3:
                # Sector moved more than 3 steps worth — reset velocity memory
                self._last_vx, self._last_vy = 0.0, 0.0
        self._last_sector_cen_m = sector_cen_m.copy() if sector_cen_m is not None else None

        if sector_cen_m is not None:
            dist_m = float(np.linalg.norm(pos_m - sector_cen_m))
            if dist_m > effective_leash:
                diff  = sector_cen_m - pos_m
                desired = float(np.arctan2(diff[1], diff[0]))
                if self._last_heading is None:
                    self._last_heading = desired
                else:
                    err = float(np.arctan2(np.sin(desired - self._last_heading),
                                           np.cos(desired - self._last_heading)))
                    self._last_heading += float(np.clip(err, -self._max_dheading,
                                                             self._max_dheading))
                vx_out = V * np.cos(self._last_heading)
                vy_out = V * np.sin(self._last_heading)
                self._last_vx, self._last_vy = vx_out, vy_out
                self.times_ms.append((time.perf_counter() - t0) * 1000)
                return float(vx_out), float(vy_out)

        # build pre-frontier buffer once (shared across all particles)
        H = self.W.grid_size
        if fire_belief is not None and len(front_pts) > 0:
            pre_front = self._pre_frontier_buffer(frontier, fire_belief)
        else:
            pre_front = np.zeros((H, H), dtype=bool)

        neighbour_fovs = self._neighbour_fov_mask(neighbour_pos_m)

        # warm-start
        n_warm = int(0.80 * self.P)
        noise  = np.random.normal(0, V * 0.25, (n_warm, 2))
        warm_vx = np.clip(self._last_vx + noise[:,0], -V, V)
        warm_vy = np.clip(self._last_vy + noise[:,1], -V, V)
        rand_ang = np.random.uniform(0, 2*np.pi, self.P - n_warm)
        pvx = np.concatenate([warm_vx, V*np.cos(rand_ang)])
        pvy = np.concatenate([warm_vy, V*np.sin(rand_ang)])
        # clamp to disc
        sp = np.hypot(pvx, pvy); ov = sp > V
        pvx[ov] = pvx[ov]/sp[ov]*V; pvy[ov] = pvy[ov]/sp[ov]*V

        vvx = np.zeros(self.P); vvy = np.zeros(self.P)

        cand_rc = self._candidate_positions(pos_m, pvx, pvy)
        pb_fit  = self._fitness(cand_rc, frontier, pre_front, sector_cen_m,
                                neighbour_pos_m, neighbour_fovs)
        pb_vx, pb_vy = pvx.copy(), pvy.copy()
        best = int(np.argmax(pb_fit))
        gb_vx, gb_vy, gb_fit = pvx[best], pvy[best], pb_fit[best]

        for _ in range(self.K):
            r1, r2 = np.random.rand(self.P), np.random.rand(self.P)
            vvx = np.clip(self.w*vvx + self.c1*r1*(pb_vx-pvx) + self.c2*r2*(gb_vx-pvx), -V, V)
            vvy = np.clip(self.w*vvy + self.c1*r1*(pb_vy-pvy) + self.c2*r2*(gb_vy-pvy), -V, V)
            pvx = np.clip(pvx+vvx, -V, V); pvy = np.clip(pvy+vvy, -V, V)
            sp = np.hypot(pvx, pvy); ov = sp > V
            pvx[ov] = pvx[ov]/sp[ov]*V; pvy[ov] = pvy[ov]/sp[ov]*V

            cand_rc = self._candidate_positions(pos_m, pvx, pvy)
            fit     = self._fitness(cand_rc, frontier, pre_front, sector_cen_m,
                                    neighbour_pos_m, neighbour_fovs)
            imp = fit > pb_fit
            pb_vx[imp] = pvx[imp]; pb_vy[imp] = pvy[imp]; pb_fit[imp] = fit[imp]
            best = int(np.argmax(pb_fit))
            if pb_fit[best] > gb_fit:
                gb_vx, gb_vy, gb_fit = pb_vx[best], pb_vy[best], pb_fit[best]

        # ── heading-rate-limited output ───────────────────────────────────
        # Convert best (vx,vy) to a desired heading, then limit how fast
        # the heading can change per step. This is the only correct way to
        # prevent brutal direction reversals — velocity vector EMA breaks
        # near 180° because averaging two opposite unit vectors gives zero.
        desired_heading = float(np.arctan2(gb_vy, gb_vx))

        if self._last_heading is None:
            smoothed_heading = desired_heading
        else:
            # Shortest angular error, rate-limited
            err = float(np.arctan2(
                np.sin(desired_heading - self._last_heading),
                np.cos(desired_heading - self._last_heading),
            ))
            dh  = float(np.clip(err, -self._max_dheading, self._max_dheading))
            smoothed_heading = self._last_heading + dh

        self._last_heading = smoothed_heading
        vx_out = V * np.cos(smoothed_heading)
        vy_out = V * np.sin(smoothed_heading)
        self._last_vx, self._last_vy = vx_out, vy_out

        self.times_ms.append((time.perf_counter() - t0) * 1000)
        return float(vx_out), float(vy_out)

    def timing_summary(self):
        t = np.array(self.times_ms)
        if not len(t):
            return {"mean_ms": None, "n_pso_calls": 0}
        return {"mean_ms": float(t.mean()), "std_ms": float(t.std()),
                "min_ms": float(t.min()), "max_ms": float(t.max()),
                "n_pso_calls": len(t)}
