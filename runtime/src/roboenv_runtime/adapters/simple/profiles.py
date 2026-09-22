"""SIMPLE control contracts from extracted asset geometry; no simulator imports."""

import copy


# SIMPLE advances its control loop at a fixed 50 Hz.
CONTROL_INTERVAL_S = 0.02
NAMES = ("base", "left_arm", "right_arm", "left_hand", "right_hand")
META = {"source": "introspected", "method": "introspection", "calibrated_at": None}
IDENTITY = [0, 0, 0, 0, 0, 0, 1]


def profiles(model):
    result = {
        "base": {
            "intrinsics": {
                **META,
                "frames": {"torso": None},
                "control": model["control"],
                "feedback": {
                    "pose_frame": "world",
                    "velocity_frame": "base",
                    "velocity_components": ["vx", "vy", "vz", "wx", "wy", "wz"],
                    "clock": "simulation",
                },
                "stop": {"behavior": "stand", "affected_devices": ["base"]},
            },
            "extrinsics": dict(META),
            "type": "velocity",
        }
    }
    for name in NAMES[1:]:
        hand = name.endswith("hand")
        part = model["parts"][name]
        parent = name.replace("hand", "arm") if hand else "base"
        intrinsics = {
            **META,
            "frames": part["frames"] if hand else {"flange": None},
            **({"geometry": part["geometry"]} if hand else {}),
            "model": {"format": "joints", "joints": part["joints"]}
            if hand
            else {
                "format": "serial_chain",
                "joints": part["joints"],
                "chain": part["chain"],
            },
            "kinematics": None
            if hand
            else {
                "frame": "flange",
                "fk": "numerical",
                "ik": {
                    "backend": "numerical",
                    "method": "damped_least_squares",
                    "position_tolerance_m": 0.001,
                    "rotation_tolerance_rad": 0.01,
                    "max_iterations": 200,
                    "damping": 0.01,
                },
                "feedback": "driver",
                "frame_to_reported": IDENTITY,
            },
            "home": part["home"],
            "control": {
                "mode": "position",
                "speed": None,
            },
            "feedback": {"velocities": True, "efforts": False, "clock": "simulation"},
            "stop": {"behavior": "hold", "affected_devices": [name]},
        }
        mount = part["mount"] if hand else IDENTITY
        result[name] = {
            "type": "dexterous_hand" if hand else "joint",
            "intrinsics": intrinsics,
            "extrinsics": {
                **META,
                "mount": {
                    "parent": parent,
                    "parent_frame": "flange" if hand else "torso",
                    "translation_m": mount[:3],
                    "rotation_xyzw": mount[3:],
                },
            },
        }
    result["world"] = {"type": "world", "intrinsics": {"bounds": None, "surfaces": []}}
    result.update({name: camera_profile(spec) for name, spec in model["cameras"].items()})
    return copy.deepcopy(result)


def camera_profile(spec):
    return {
        "type": "rgb",
        "intrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "color": {
                "width": spec["width"],
                "height": spec["height"],
                "fps": None,
                "pixel_format": "rgb8",
                "image_format": "png",
                "projection": spec["projection"],
                "distortion": {"model": "none"},
            },
        },
        "extrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "mount": spec["mount"],
        },
    }
