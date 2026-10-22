"""Booster's explicit joint PD with torque speed clipping, as the robot's joint drivers apply it."""

from __future__ import annotations

import numpy as np

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import BoosterPdGains


def booster_pd_torque(
    target: np.ndarray, pos: np.ndarray, vel: np.ndarray, gains: BoosterPdGains
) -> np.ndarray:
    """Joint torque kp·(target - pos) - kd·vel, clipped to Booster's symmetric torque speed
    curve: `effort_limit` up to |vel| = `knee_point_velocity`, linear to 0 at `velocity_limit`.
    All arrays (num_joints,); rad, rad/s, N·m."""
    torque = gains.stiffness * (target - pos) - gains.damping * vel
    span = np.maximum(gains.velocity_limit - gains.knee_point_velocity, 1e-6)
    limit = np.clip(gains.effort_limit * (gains.velocity_limit - np.abs(vel)) / span, 0.0, None)
    limit = np.minimum(limit, gains.effort_limit)
    return np.clip(torque, -limit, limit)
