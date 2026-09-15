"""Six G1 device contracts from extracted asset geometry; no simulator imports."""

import copy


NAMES = ("base", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
META = {"source": "introspected", "method": "introspection", "calibrated_at": None}
IDENTITY = [0, 0, 0, 0, 0, 0, 1]


def profiles(model, *, camera_profile):
    result = {
        "base": {
            "type": "base",
            "intrinsics": {
                **META,
                "mobility": "mobile",
                "control": model["control"],
                "feedback": {
                    "pose_frame": "world",
                    "velocity_frame": "base",
                    "velocity_components": ["vx", "vy", "vz", "wx", "wy", "wz"],
                    "clock": "simulation",
                },
                "stop": {"behavior": "stand", "affected_devices": ["base"]},
            },
            "extrinsics": {**META, "frames": []},
        }
    }
    for name in NAMES[1:]:
        hand = name.endswith("hand")
        part = model["parts"][name]
        parent = (
            "base" if name == "waist" else name.replace("hand", "arm") if hand else "waist"
        )
        intrinsics = {**META, "stop": {"behavior": "hold", "affected_devices": [name]}}
        if hand:
            intrinsics.update(
                tool_type="dexterous_hand",
                control={"mode": "joint_position", "joints": part["joints"], "speed": None},
                feedback={
                    "positions": [
                        {"name": joint["name"], "units": "rad"} for joint in part["joints"]
                    ],
                    "velocities": True,
                    "sensors": [],
                    "clock": "simulation",
                    "aperture_model": None,
                },
            )
        if not hand:
            intrinsics.update(
                kinematics={
                    "model": {
                        "format": "serial_chain",
                        "joints": part["joints"],
                        "chain": part["chain"],
                    },
                    "ik": {
                        "backend": "numerical",
                        "method": "damped_least_squares",
                        "position_tolerance_m": 0.001,
                        "rotation_tolerance_rad": 0.01,
                        "max_iterations": 200,
                        "damping": 0.01,
                    },
                },
                home=part["home"],
                control={"speed": None},
                feedback={
                    "velocities": True,
                    "efforts": False,
                    "pose_source": "driver",
                    "flange_to_reported": IDENTITY,
                },
            )
        mount = part["mount"] if hand else IDENTITY
        result[name] = {
            "type": "end_effector" if hand else "manipulator",
            "intrinsics": intrinsics,
            "extrinsics": {
                **META,
                "mount": {
                    "parent": parent,
                    "parent_frame": parent if parent == "base" else "flange",
                    "translation_m": mount[:3],
                    "rotation_xyzw": mount[3:],
                },
            },
        }
    result["world"] = {"type": "world", "intrinsics": {"bounds": None, "surfaces": []}}
    result.update({name: camera_profile(spec) for name, spec in model["cameras"].items()})
    return copy.deepcopy(result)
