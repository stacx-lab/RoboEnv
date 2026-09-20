"""Portable Go2–D1 device facts from the prepared robot description."""

import math
import numpy as np
from ..g1_cameras import profile as camera_profile
from .gait import BOUNDS, DEFAULT
from .kinematics import matrix_quat, pose

META = {"source": "introspected", "method": "introspection", "calibrated_at": None}
IDENTITY = [0, 0, 0, 0, 0, 0, 1]


def profiles(model, home, realtime=False, timeout=0.5):
    execution = (
        {"mode": "realtime", "command_timeout_s": timeout}
        if realtime
        else {"mode": "stepped", "interval_s": 0.02}
    )

    def component(axis):
        lo, hi = BOUNDS[axis]
        item = {
            "axis": axis,
            "units": "m/s"
            if axis in ("vx", "vy")
            else "m"
            if axis == "height"
            else "rad/s"
            if axis == "wz"
            else "rad",
            "bounds": {"min": lo, "max": hi},
        }
        if axis in ("height", "roll", "pitch"):
            item["default"] = float(DEFAULT[list(BOUNDS).index(axis)])
        return item

    result = {
        "base": {
            "type": "velocity",
            "intrinsics": {
                **META,
                "frames": {},
                "control": {
                    "mode": "motion",
                    "body_frame": "base",
                    "modes": {
                        "locomotion": {
                            "velocity": {
                                "frame": "body",
                                "components": [component(k) for k in ("vx", "vy", "wz")],
                            },
                            "posture": {
                                "reference": "ground",
                                "components": [component(k) for k in ("height", "roll", "pitch")],
                            },
                        }
                    },
                    "execution": execution,
                },
                "feedback": {
                    "pose_frame": "world",
                    "velocity_frame": "base",
                    "velocity_components": ["vx", "vy", "vz", "wx", "wy", "wz"],
                    "clock": "simulation",
                },
                "stop": {"behavior": "stand", "affected_devices": ["base"]},
            },
            "extrinsics": dict(META),
        }
    }
    frames = model.forward({})
    chain = []
    joints = []
    for i, name in enumerate(model.arm_names):
        j = model.joints[name]
        lo, hi = j["limits"]
        joints.append(
            {
                "name": name,
                "type": "revolute",
                "limits": {"min": lo, "max": hi},
                "max_velocity": 1.5,
                "max_effort": 30.0,
            }
        )
        t = j["origin"]
        chain.append(
            {
                "origin": [*t[:3, 3].tolist(), *matrix_quat(t[:3, :3]).tolist()],
                "joint": i,
                "axis": j["axis"].tolist(),
                "anchor": [0, 0, 0],
                "reference": 0,
            }
        )
    result["arm"] = {
        "type": "joint",
        "intrinsics": {
            **META,
            "frames": {"flange": None},
            "model": {"format": "serial_chain", "joints": joints, "chain": chain},
            "home": {n: home[n] for n in model.arm_names},
            "kinematics": {
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
            "control": {
                "mode": "position",
                "speed": None,
                "execution": {"mode": "realtime", "command_timeout_s": None}
                if realtime
                else execution,
            },
            "feedback": {"velocities": True, "efforts": False, "clock": "simulation"},
            "stop": {"behavior": "hold", "affected_devices": ["arm"]},
        },
        "extrinsics": {
            **META,
            "mount": {"parent": "base", "parent_frame": "base", **pose(frames["d1_base_link"])},
        },
    }
    result["gripper"] = {
        "type": "gripper",
        "intrinsics": {
            **META,
            "aperture_m": None,
            "control": {
                "mode": "position",
                "units": "normalized",
                "resolution": None,
                "bounds": {"min": 0, "max": 1},
                "open_target": 1,
                "close_target": 0,
                "speed": None,
            },
            "feedback": {
                "positions": [{"name": n, "units": "m"} for n in ["d1_Joint7_1", "d1_Joint7_2"]],
                "velocities": True,
                "sensors": [],
                "clock": "simulation",
            },
            "stop": {"behavior": "hold", "affected_devices": ["gripper"]},
            "frames": {},
        },
        "extrinsics": {
            **META,
            "mount": {
                "parent": "arm",
                "parent_frame": "flange",
                "translation_m": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
            },
        },
    }
    for name, native, parent in [
        ("camera_wrist", "wrist_d405_color_optical_frame", "arm"),
        ("camera_head", "head_d435_color_optical_frame", "base"),
    ]:
        frame = (
            np.linalg.inv(frames["d1_Link6"]) @ frames[native]
            if parent == "arm"
            else frames[native]
        )
        # Registered RGB-D simulation uses the RGB FOV for both streams.
        fov = 87 if parent == "arm" else 69
        spec = {
            "width": 480,
            "height": 270,
            "focal_length": 24.0,
            "aperture": 48 * math.tan(math.radians(fov / 2)),
            "clipping_range": [0.015, 20.0],
            "parent": parent,
            "parent_frame": "flange" if parent == "arm" else "base",
            **pose(frame),
        }
        result[name] = {"type": "rgbd", **camera_profile(spec)}
    result["world"] = {
        "type": "world",
        "intrinsics": {"bounds": None, "surfaces": []},
        "extrinsics": {},
    }
    return result
