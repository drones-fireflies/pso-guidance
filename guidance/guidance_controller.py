import numpy as np

from guidance_pso.guidance.pso_agents import PSOAgent
from guidance_pso.perception.belief_map import BeliefMaps
from typing import List, Optional
from guidance_pso.world import WORLD, WorldParams
from guidance_pso.metrics import CoverageMetrics, compute_metrics
from guidance_pso.guidance.velocity_adapter import VelocityCommand, make_velocity_command, integrate_position


def _fov_mask(drones: list, H: int, W: int, fov_half: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for d in drones:
        r, c = d.grid_rc_int
        mask[max(0,r-fov_half):min(H,r+fov_half+1),
             max(0,c-fov_half):min(W,c+fov_half+1)] = True
    return mask


class DroneState:
    def __init__(self, x: float, y: float, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def grid_rc(self) -> np.ndarray:
        return WORLD.m2cell(self.pos)

    @property
    def grid_rc_int(self) -> np.ndarray:
        rc = self.grid_rc
        return np.array([
            int(np.clip(rc[0], 0, WORLD.grid_size - 1)),
            int(np.clip(rc[1], 0, WORLD.grid_size - 1)),
        ])

    def pose_vec(self) -> np.ndarray:
        """([row, col, θ], for display compatibility."""
        rc = self.grid_rc
        return np.array([rc[0], rc[1], self.heading], dtype=np.float32)


class WildfireGuidanceController:

    def __init__(
        self,
        fire_map:        np.ndarray,
        n_drones:        int                   = 6,
        spawn_positions: Optional[np.ndarray] = None, 
        world:           WorldParams           = WORLD,
        pso_particles:   int                  = 40,
        pso_iterations:  int                  = 30,
        fov_s:           int                  = None,
    ):
        self.world = world
        W  = world
        Ng = W.grid_size
        self.H = Ng; self.W_g = Ng
        self.true_fire     = fire_map.copy().astype(np.float32)
        self.N             = n_drones
        self.fov_half      = (fov_s // 2) if fov_s else W.fov_half_cells

        self.belief = BeliefMaps(Ng, Ng)

        # drones initial positions
        if spawn_positions is not None:
            sp = np.array(spawn_positions, dtype=float)
        else:
            sp = self._random_spawn_metres()

        self._drones: List[DroneState] = [
            DroneState(float(sp[i, 0]), float(sp[i, 1]),
                       float(np.random.uniform(0, 2*np.pi)))
            for i in range(n_drones)
        ]

        # Decentralized PSO (one per drone)
        self.pso_agents: List[PSOAgent] = [
            PSOAgent(
                n_particles   = pso_particles,
                n_iterations  = pso_iterations,
                sensor_radius = self.fov_half,
                world         = W,
            )
            for _ in range(n_drones)
        ]

        self.velocity_commands: List[VelocityCommand] = [
            VelocityCommand(0.0, 0.0, 0.0, 0.0) for _ in range(n_drones)
        ]

        self.step_count = 0
        self.history: List[CoverageMetrics] = []

        self._do_observations()

    # ── spawn ─────────────────────────────────────────────────────────────────

    def _random_spawn_metres(self) -> np.ndarray:
        W  = self.world
        nb = np.argwhere(self.true_fire < 0.5)
        if len(nb) >= self.N:
            rc = nb[np.random.choice(len(nb), self.N, replace=False)].astype(float)
            return W.cell2m(rc)
        return np.random.uniform([0,0], [W.grid_metres]*2, (self.N, 2))

    def _do_observations(self):
        mask = _fov_mask(self._drones, self.H, self.W_g, self.fov_half)
        self.belief.observe(self.true_fire, mask)

    def _assign_sectors(self, front_pts: np.ndarray) -> list:
        """N angular sectors → list of (2,) sector centroid in metres."""
        if len(front_pts) == 0:
            return [None] * self.N
        centroid = front_pts.mean(axis=0)
        delta    = front_pts - centroid
        angles   = np.arctan2(delta[:, 1], delta[:, 0])
        centroids = []
        for i in range(self.N):
            lo = -np.pi + (2*np.pi/self.N)*i
            hi = lo + (2*np.pi/self.N)
            if hi <= np.pi:
                mask = (angles >= lo) & (angles < hi)
            else:
                mask = (angles >= lo) | (angles < hi - 2*np.pi)
            pts = front_pts[mask] if mask.any() else front_pts
            centroids.append(self.world.cell2m(pts.mean(axis=0)))
        return centroids

    def _pso_step(self):
        frontier  = self.belief.frontier
        front_pts = self.belief.front_pts
        sectors   = self._assign_sectors(front_pts)

        for i, (drone, agent) in enumerate(zip(self._drones, self.pso_agents)):
            neighbour_pos_m = np.array([
                self._drones[j].pos for j in range(self.N) if j != i
            ])
            vx, vy = agent.compute_velocity(
                pos_m           = drone.pos,
                frontier        = frontier,
                front_pts       = front_pts,
                neighbour_pos_m = neighbour_pos_m,
                sector_cen_m    = sectors[i],
                fire_belief     = self.belief.fire_belief,
            )
            cmd     = make_velocity_command(vx, vy, drone.heading, self.world.dt, self.world.speed)
            new_pos = integrate_position(drone.pos, cmd, self.world.dt)
            drone.x, drone.y = float(new_pos[0]), float(new_pos[1])
            drone.heading     = cmd.heading
            self.velocity_commands[i] = cmd

    def step(self) -> CoverageMetrics:
                    
        self._pso_step()
        recent = [a.times_ms[-1] for a in self.pso_agents if a.times_ms]
        latency_ms = float(np.mean(recent)) if recent else -1.0

        self._do_observations()

        from guidance_pso.perception.front_detection import cv_frontier as _cv_f
        true_frontier = _cv_f((self.true_fire*255).astype(np.uint8))

        drone_cells = np.array([d.grid_rc_int for d in self._drones])
        rho = int(np.ceil(np.sqrt(2) * self.fov_half))

        self.step_count += 1
        m = compute_metrics(
            step                = self.step_count,
            drone_positions     = drone_cells,
            frontier            = self.belief.frontier,
            fire_belief         = self.belief.fire_belief,
            sensor_radius       = rho,
            true_frontier       = true_frontier,
            true_fire           = self.true_fire,
            decision_latency_ms = latency_ms,
        )
        self.history.append(m)
        return m

    def run(self, max_steps: int = 300, verbose: bool = True):
        for _ in range(max_steps):
            m = self.step()
            if verbose: print(m)
        return self.history

    def summary(self) -> dict:
        pso_hist = [m for m in self.history]
        if not pso_hist: return {"warning": "no PSO steps recorded"}
        def _b(v): return {"mean": float(np.mean(v)), "max": float(np.max(v)), "final": float(v[-1])} if v else {}
        fi  = [m.frontier_iou      for m in pso_hist if m.frontier_iou >= 0]
        ri  = [m.fire_iou          for m in pso_hist if m.fire_iou     >= 0]
        lat = [m.decision_latency_ms for m in pso_hist if m.decision_latency_ms >= 0]
        return {
            "world": {"grid_m": self.world.grid_metres, "cell_m": self.world.cell_size,
                      "speed_ms": self.world.speed, "dt_s": self.world.dt},
            "frontier_iou": _b(fi), "fire_iou": _b(ri),
            "latency_ms": {"mean": float(np.mean(lat)), "std": float(np.std(lat))} if lat else {},
        }

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def poses(self) -> np.ndarray:
        return np.array([d.pose_vec() for d in self._drones], dtype=np.float32)

    @property
    def drones(self) -> np.ndarray:
        return np.array([d.grid_rc_int for d in self._drones])

    @property
    def positions_metres(self) -> np.ndarray:
        return np.array([d.pos for d in self._drones])

    @property
    def frontier(self): return self.belief.frontier

    @property
    def front_pts(self): return self.belief.front_pts
