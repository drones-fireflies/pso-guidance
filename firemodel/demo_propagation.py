"""Wildfire propagation simulation demo"""

import time
import logging

from guidance_pso.viz.fire_display import Display
from guidance_pso.firemodel.propagation import Environment, FirePropagation


if __name__ == "__main__":
    """Wildfire propagation demo."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # -------------------------------------------------------------------------
    # 1. Environment setup
    # -------------------------------------------------------------------------
    env = Environment(
        grid_size=100,
        ignition_points=[(50, 50)],
        wind_direction=2.55,
    )

    # Show some environment info
    logging.info("Environment created (grid: %dx%d)", env.grid_size_x, env.grid_size_y)
    logging.info("Ignition points: %s", env.ignition_points)
    logging.info("Wind direction: %.2f rad", env.wind_direction)

    # Display a small sample of the fuel/elevation maps
    # print("\nFuel map (top-left 5x5):\n", env.fuel_map[:5, :5])
    # print("\nElevation map (top-left 5x5):\n", env.elevation_map[:5, :5])

    # -------------------------------------------------------------------------
    # 2. Fire propagation
    # -------------------------------------------------------------------------
    sim = FirePropagation(timesteps=100,visualize=False)

    start = time.time()
    fire_maps, _, elevation_map, _ = sim.propagate_from_points(env)
    duration = time.time() - start

    print(f"\nSimulation completed in {duration:.2f} seconds")

    visualizer = Display(
        height = env.grid_size_x,
        width = env.grid_size_y,
        wind_direction = env.wind_direction,
        elevation = elevation_map
    )
    visualizer.show(fire_maps)
