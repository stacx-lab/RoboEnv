"""Fixed manipulator facts extracted from the compiled model, in physical units."""


def intrinsics(sim, indexes, chain, home, name):
    model = sim.model._model
    joints = [
        {
            "name": sim.model.joint_id2name(int(index)),
            "type": "prismatic" if model.jnt_type[index] == 2 else
                    "revolute" if model.jnt_limited[index] else "continuous",
            "limits": {"min": float(model.jnt_range[index, 0]), "max": float(model.jnt_range[index, 1])}
                      if model.jnt_limited[index] else None,
            "max_velocity": None,
            "max_effort": None,
        }
        for index in indexes
    ]
    return {
        "source": "introspected", "method": "introspection", "calibrated_at": None,
        "kinematics": {
            "model": {"format": "serial_chain", "joints": joints, "chain": chain},
            "ik": {"backend": "numerical", "method": "damped_least_squares",
                   "position_tolerance_m": 0.0001, "rotation_tolerance_rad": 0.001,
                   "max_iterations": 200, "damping": 0.01},
        },
        "home": {joint["name"]: float(value) for joint, value in zip(joints, home, strict=True)},
        "control": {"speed": None},
        "feedback": {"velocities": True, "efforts": False, "pose_source": "driver",
                     "flange_to_reported": [0, 0, 0, 0, 0, 0, 1]},
        "stop": {"behavior": "hold", "affected_devices": [name]},
    }
