"""Chassis facts and measured spatial velocity for the native Omron controller."""

import numpy as np


def intrinsics(world, pose_frame="world"):
    execution = {"mode": "realtime", "command_timeout_s": None}
    if not world.realtime:
        execution = {"mode": "stepped", "interval_s": 1.0 / world.env.control_freq}
    return {
        "source": "introspected", "method": "introspection", "calibrated_at": None,
        "mobility": "mobile",
        "control": {
            "frame": "body",
            "components": [
                {"axis": axis, "units": "normalized", "bounds": {"min": -1, "max": 1}}
                for axis in ("vx", "vy", "wz")
            ],
            "execution": execution,
        },
        "feedback": {
            "pose_frame": pose_frame, "velocity_frame": "body",
            "velocity_components": ["vx", "vy", "wz"], "clock": "simulation",
        },
    }


def reading(model, data, body):
    import mujoco

    # Object velocity is at the body origin, including its offset from the yaw
    # joint. Raw generalized qvel is neither body-frame velocity nor this point.
    spatial = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, spatial, 1)
    quat = data.xquat[body]
    return {
        "pose": {
            "translation_m": data.xpos[body].tolist(),
            "rotation_xyzw": [*quat[1:].tolist(), float(quat[0])],
        },
        "velocities": spatial[[3, 4, 2]].tolist(),
        "faults": [], "sample_time_s": float(data.time),
    }
