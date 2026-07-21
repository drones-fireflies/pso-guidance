"""
velocity_adapter.py  -  Velocity commands adaptation
====================================================
"""

import numpy as np
from dataclasses import dataclass
from guidance_pso.world import WORLD


@dataclass
class VelocityCommand:
    vx:       float
    vy:       float
    yaw_rate: float
    heading:  float


def make_velocity_command(
    vx: float,
    vy: float,
    prev_heading: float,
    dt: float = None,
    max_speed: float = None,
) -> VelocityCommand:
    """
    Build a VelocityCommand from desired (vx, vy).

    Enforces max_speed, derives heading from velocity direction,
    computes yaw_rate from heading change.

    """

    if dt is None:
        dt = WORLD.dt
    if max_speed is None:
        max_speed = WORLD.airspeed

    # Enforce max speed
    speed = float(np.hypot(vx, vy))
    if speed > max_speed:
        vx = vx * max_speed / speed
        vy = vy * max_speed / speed
        speed = max_speed

    # Heading from velocity direction
    if speed > 0.1:
        heading = float(np.arctan2(vy, vx))
    else:
        heading = prev_heading

    # Yaw rate from heading change
    dtheta   = float(np.arctan2(np.sin(heading - prev_heading),
                                np.cos(heading - prev_heading)))
    yaw_rate = dtheta / dt

    return VelocityCommand(
        vx       = float(vx),
        vy       = float(vy),
        yaw_rate = float(yaw_rate),
        heading  = float(heading),
    )

def integrate_position(
    position: np.ndarray,
    cmd:        VelocityCommand,
    dt:         float = None,
) -> np.ndarray:
    
    if dt is None:
        dt = WORLD.dt

    new_x = position[0] + cmd.vx * dt
    new_y = position[1] + cmd.vy * dt
    return WORLD.clip_metres(np.array([new_x, new_y]))