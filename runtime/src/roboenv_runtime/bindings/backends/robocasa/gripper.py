"""Typed gripper facts; native command dimensions differ from measured joints."""


def intrinsics(robot, sim, joints):
    return {
        "source": "introspected", "method": "introspection", "calibrated_at": None,
        "tool_type": "parallel_gripper", "aperture_m": None,
        "control": {
            "mode": "direction", "open_target": -1, "close_target": 1,
            "increment_per_interval": float(robot.gripper["right"].speed),
        },
        "feedback": {
            "positions": [{"name": sim.model.joint_id2name(int(j)), "units": "m"} for j in joints],
            "velocities": True, "sensors": [],
            "clock": "simulation", "aperture_model": None,
        },
        "stop": {"behavior": "hold", "affected_devices": []},
    }
