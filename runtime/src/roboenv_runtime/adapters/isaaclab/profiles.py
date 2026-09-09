"""Go2 device facts shared by catalog generation and native binding validation.

No simulator imports or asset acquisition. Names select logical device references;
all geometry and capabilities describe the pinned rig, not a robot-wide controller.
"""

from .rig import (
    APERTURE, CAMERA_POSITION, CAMERA_ROTATION, CAMERA_SIZE, CONTROL_DT,
    FEET, FOCAL_LENGTH, JOINTS,
)
from roboenv_runtime.bindings.backends.shared.depth import DEPTH_SCALE

MEASURED = {"source": "introspected", "method": "introspection", "calibrated_at": None}
DEVICE_NAMES = {"body": "body", "camera_front": "camera_front", "imu": "imu",
                **{foot: f"contact_{foot.lower()}" for foot in FEET}, "world": "world"}


def channel():
    return {"axes": ["x", "y", "z"], "range": None, "resolution": None,
            "bias": {"x": 0, "y": 0, "z": 0}}


def device_profiles(model, names=None):
    body = (names or {}).get("body", "body")
    profiles = {}
    joints = [
        {"name": name, "type": "revolute", "limits": {"min": lo, "max": hi},
         "max_velocity": None, "max_effort": None}
        for name, (lo, hi) in zip(JOINTS, model["limits"], strict=True)
    ]
    profiles["body"] = {
        "type": "legged",
        "intrinsics": {
            **MEASURED,
            "joints": joints,
            "home": dict(zip(JOINTS, model["home"], strict=True)),
            "legs": [
                {"name": foot.removesuffix("_foot"), "joints": list(JOINTS[i * 3:i * 3 + 3]),
                 "foot_frame": foot} for i, foot in enumerate(FEET)
            ],
            "control": {"mode": "joint_position",
                        "execution": {"mode": "stepped", "interval_s": CONTROL_DT}},
            "feedback": {"velocities": True, "efforts": False, "pose_frame": None,
                         "velocity_frame": None, "velocity_components": [], "clock": "simulation"},
            "stop": {"behavior": "hold", "affected_devices": [body]},
        },
        "extrinsics": {**MEASURED, "frames": list(FEET)},
    }
    focal = CAMERA_SIZE * FOCAL_LENGTH / APERTURE
    profiles["camera_front"] = {
        "type": "rgbd",
        "intrinsics": {
            **MEASURED,
            "color": {
                "width": CAMERA_SIZE, "height": CAMERA_SIZE, "fps": None,
                "pixel_format": "rgb8", "image_format": "png",
                "projection": {"model": "pinhole", "fx": focal, "fy": focal,
                               "cx": CAMERA_SIZE / 2, "cy": CAMERA_SIZE / 2, "skew": 0},
                "distortion": {"model": "none"},
            },
            "depth": {
                "registration": "color", "width": CAMERA_SIZE, "height": CAMERA_SIZE, "fps": None,
                "pixel_format": "uint16", "image_format": "pgm16_be", "quantity": "z",
                "scale_m_per_unit": DEPTH_SCALE, "invalid_value": 0,
                "valid_range_m": {"min": DEPTH_SCALE, "max": 65535 * DEPTH_SCALE}, "reliable_range_m": "???",
            },
        },
        "extrinsics": {
            **MEASURED,
            "mount": {"parent": body, "parent_frame": body,
                      "translation_m": list(CAMERA_POSITION),
                      "rotation_xyzw": list(CAMERA_ROTATION)},
        },
    }
    profiles["imu"] = {
        "type": "imu",
        "intrinsics": {
            **MEASURED,
            "measurement": {
                "angular_velocity": channel(),
                "linear_acceleration": {"quantity": "specific_force", **channel()},
                "orientation": None, "magnetic_field": None,
            },
            "sample_rate_hz": None,
        },
        "extrinsics": {
            **MEASURED,
            "mount": {"parent": body, "parent_frame": body,
                      "translation_m": model["imu_position"], "rotation_xyzw": [0, 0, 0, 1]},
        },
    }
    for foot in FEET:
        profiles[foot] = {
            "type": "force_torque",
            "intrinsics": {
                **MEASURED,
                "measurement": {
                    "quantity": "contact_normal", "target": {"device": body, "frame": foot},
                    "force": channel(),
                    "torque": None,
                },
                "sample_rate_hz": None,
            },
            "extrinsics": {
                **MEASURED,
                "mount": {"parent": body, "parent_frame": foot,
                          "translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
            },
        }
    profiles["world"] = {
        "type": "world",
        "intrinsics": {
            "source": "default",
            "bounds": {"min": [None] * 3, "max": [None] * 3},
            "surfaces": [],
        },
    }
    return profiles
