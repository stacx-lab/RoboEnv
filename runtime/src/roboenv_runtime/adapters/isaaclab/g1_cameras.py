"""Configured RGB-D observation rig for the G1 practice scene (optical xyzw)."""

from roboenv_runtime.bindings.backends.shared.depth import DEPTH_SCALE


def specs():
    common = {
        "width": 512,
        "height": 512,
        "focal_length": 24.0,
        "aperture": 40.2767822965,
        "clipping_range": [0.05, 20.0],
    }
    return {
        "camera_torso": {
            **common,
            "aperture": 57.204172444521,  # 100 degrees covers both hands and the wheel.
            "parent": "waist",
            "parent_frame": "flange",
            "prim_path": "{ENV_REGEX_NS}/Robot/torso_link/CameraTorso",
            "translation_m": [0.12, 0, 0.32],
            "rotation_xyzw": [
                0.62721137512625,
                -0.62721137512625,
                0.32650557562197696,
                -0.32650557562197696,
            ],
        },
        "camera_scene": {
            **common,
            "parent": "world",
            "parent_frame": "world",
            "prim_path": "{ENV_REGEX_NS}/CameraScene",
            "translation_m": [2.0, -1.8, 1.8],
            "rotation_xyzw": [
                0.7708244688916118,
                0.2836074522700417,
                -0.19696984075991342,
                -0.5353497296215576,
            ],
        },
    }


def profile(spec):
    width, height = spec["width"], spec["height"]
    focal = width * spec["focal_length"] / spec["aperture"]
    meta = {"source": "default", "method": "manual", "calibrated_at": None}
    return {
        "type": "rgbd",
        "intrinsics": {
            **meta,
            "color": {
                "width": width,
                "height": height,
                "fps": None,
                "pixel_format": "rgb8",
                "image_format": "png",
                "projection": {
                    "model": "pinhole",
                    "fx": focal,
                    "fy": focal,
                    "cx": width / 2,
                    "cy": height / 2,
                    "skew": 0,
                },
                "distortion": {"model": "none"},
            },
            "depth": {
                "registration": "color",
                "width": width,
                "height": height,
                "fps": None,
                "pixel_format": "uint16",
                "image_format": "pgm16_be",
                "quantity": "z",
                "scale_m_per_unit": DEPTH_SCALE,
                "invalid_value": 0,
                "valid_range_m": {"min": DEPTH_SCALE, "max": 65535 * DEPTH_SCALE},
                "reliable_range_m": {
                    "min": spec["clipping_range"][0],
                    "max": spec["clipping_range"][1],
                },
            },
        },
        "extrinsics": {
            **meta,
            "mount": {
                key: spec[key]
                for key in ("parent", "parent_frame", "translation_m", "rotation_xyzw")
            },
        },
    }


def configure(scene, cameras):
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    for name, spec in cameras.items():
        setattr(
            scene,
            name,
            CameraCfg(
                prim_path=spec["prim_path"],
                width=spec["width"],
                height=spec["height"],
                data_types=["rgb", "distance_to_image_plane"],
                spawn=PinholeCameraCfg(
                    focal_length=spec["focal_length"],
                    horizontal_aperture=spec["aperture"],
                    vertical_aperture=spec["aperture"],
                    clipping_range=tuple(spec["clipping_range"]),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(spec["translation_m"]),
                    rot=tuple(spec["rotation_xyzw"]),
                    convention="ros",
                ),
            ),
        )
