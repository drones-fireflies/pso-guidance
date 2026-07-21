"""Wildfire multi-UAV guidance simulation demo"""

import time
import json
import logging
import numpy as np

from guidance_pso.metrics import CoverageMetrics
from guidance_pso.guidance.guidance_controller import WildfireGuidanceController
from guidance_pso.firemodel.propagation import Environment, FirePropagation
from guidance_pso.perception.front_detection import cv_frontier, frontier_cells
from guidance_pso.viz.display_utils import RealtimeAnimator, plot_metrics

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    GRID_SIZE      = 200
    N_DRONES       = 5
    N_STEPS        = 600
    FIRE_PERIOD    = 10      # propagate fire every N drone steps
    PSO_PARTICLES  = 20
    PSO_ITERATIONS = 15
    FOV_S          = None
    SAVE_ANIM      = None    # e.g. "simulation.gif"
    SAVE_PLOT      = None    # e.g. "metrics.pdf"

    # -------------------------------------------------------------------------
    # 1. Environment setup
    # -------------------------------------------------------------------------
    env = Environment(
        grid_size=GRID_SIZE,
        ignition_points=[(GRID_SIZE // 2, GRID_SIZE // 2)],
        wind_direction=2.55,
    )

    # logging.info("Environment (grid: %dx%d)", env.grid_size_x, env.grid_size_y)
    # logging.info("Ignition point: %s", env.ignition_points)
    # logging.info("Wind direction: %.2f rad", env.wind_direction)

    # -------------------------------------------------------------------------
    # 2. Initial fire state
    # -------------------------------------------------------------------------
    fire_sim = FirePropagation(timesteps=20, visualize=False)
    fire_states, fuel_states, _, _ = fire_sim.propagate_from_points(env)
    current_fire = fire_states[-1].copy()
    current_fuel = fuel_states[-1].copy()
    slope_effect = env._compute_slope_effect(env.elevation_map)
    H, W = current_fire.shape

    # -------------------------------------------------------------------------
    # 3. Guidance controller + animator
    # -------------------------------------------------------------------------
    ctrl = WildfireGuidanceController(
        fire_map=current_fire,
        n_drones=N_DRONES,
        fov_s=FOV_S,
        pso_particles=PSO_PARTICLES,
        pso_iterations=PSO_ITERATIONS,
    )
    anim = RealtimeAnimator(n_drones=N_DRONES, fov_half=ctrl.fov_half, grid_shape=(H, W))

    def push(label):
        anim.push_frame(
            true_fire=current_fire,
            true_frontier=cv_frontier((current_fire * 255).astype(np.uint8)),
            belief_fire=ctrl.belief.fire_belief,
            belief_frontier=ctrl.belief.frontier,
            drone_pos=ctrl.drones,
            poses=ctrl.poses,
            label=label,
        )

    push("t=0  ignition")

    # -------------------------------------------------------------------------
    # 4. Mission loop
    # -------------------------------------------------------------------------
    history: list[CoverageMetrics] = []
    start = time.time()

    for t in range(N_STEPS):
        m = ctrl.step()
        history.append(m)
        cmd = ctrl.velocity_commands[0]
        push(f"t={t+1:4d}s  frontIoU={m.frontier_iou:.3f}  fireIoU={m.fire_iou:.3f}  "
             f"vx={cmd.vx:+.1f} vy={cmd.vy:+.1f} m/s")

        if (t + 1) % FIRE_PERIOD == 0:
            current_fire, current_fuel = fire_sim._spread_fire(
                current_fire, current_fuel, slope_effect,
                env.wind_direction, env.wind_strength,
                env.burn_rate_map, env.moisture_map,
            )
            ctrl.true_fire = current_fire.copy()
            ctrl.belief.front_pts = frontier_cells((current_fire * 255).astype(np.uint8))

    print(f"\nSimulation completed in {time.time() - start:.2f} seconds")

    # -------------------------------------------------------------------------
    # 5. Results
    # -------------------------------------------------------------------------
    anim.finalize()
    if SAVE_ANIM:
        anim.save(SAVE_ANIM, fps=10, dpi=120)

    print(json.dumps(ctrl.summary(), indent=2))
    plot_metrics(ctrl, history, save_path=SAVE_PLOT)