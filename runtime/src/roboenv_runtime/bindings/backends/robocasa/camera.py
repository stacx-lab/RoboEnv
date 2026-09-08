"""Camera calibration from the live MuJoCo model and RoboCasa depth encoder."""

import numpy as np

from roboenv_runtime.adapters.robocasa.depth import DEPTH_SCALE


def intrinsics(width, height, vertical_fov, rgbd):
    focal = float(height / (2 * np.tan(np.deg2rad(vertical_fov) / 2)))

    def ray(u, v):
        return np.array([(u - width / 2) / focal, (v - height / 2) / focal, 1.0])

    def angle(a, b):
        return float(np.rad2deg(np.arccos(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1
        ))))

    result = {
        "source": "introspected", "method": "introspection", "calibrated_at": None,
        "color": {
            "width": width, "height": height, "fps": None,
            "pixel_format": "rgb8", "image_format": "png",
            "projection": {
                "model": "pinhole", "fx": focal, "fy": focal,
                "cx": width / 2, "cy": height / 2, "skew": 0,
            },
            "distortion": {"model": "none"},
            "fov": {
                "horizontal_deg": angle(ray(-0.5, (height - 1) / 2),
                                        ray(width - 0.5, (height - 1) / 2)),
                "vertical_deg": angle(ray((width - 1) / 2, -0.5),
                                      ray((width - 1) / 2, height - 0.5)),
            },
        },
    }
    if rgbd:
        result["depth"] = {
            "registration": "color", "width": width, "height": height, "fps": None,
            "pixel_format": "uint16", "image_format": "pgm16_be", "quantity": "z",
            "scale_m_per_unit": DEPTH_SCALE, "invalid_value": 0,
            "valid_range_m": {"min": DEPTH_SCALE, "max": 65535 * DEPTH_SCALE},
            "reliable_range_m": "???",
        }
    return result
