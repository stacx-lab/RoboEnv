"""Camera calibration from the live MuJoCo model and RoboCasa depth encoder."""

import numpy as np

from roboenv_runtime.adapters.robocasa.depth import DEPTH_SCALE


def intrinsics(width, height, vertical_fov, rgbd):
    focal = float(height / (2 * np.tan(np.deg2rad(vertical_fov) / 2)))

    def ray(u, v):
        return np.array([(u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal, 1.0])

    def angle(a, b):
        return float(
            np.rad2deg(
                np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            )
        )

    result = {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
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
                "cx": (width - 1) / 2,
                "cy": (height - 1) / 2,
                "skew": 0,
            },
            "distortion": {"model": "none"},
            "fov": {
                "horizontal_deg": angle(
                    ray(-0.5, (height - 1) / 2), ray(width - 0.5, (height - 1) / 2)
                ),
                "vertical_deg": angle(
                    ray((width - 1) / 2, -0.5), ray((width - 1) / 2, height - 0.5)
                ),
            },
        },
    }
    if rgbd:
        result["depth"] = {
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
            "reliable_range_m": "???",
        }
    return result


def bind(spec, root, context):
    import uuid
    from pathlib import Path

    from roboenv_runtime.adapters.robocasa.depth import write_depth

    from .common import action, finish

    definition = spec.definition(root)
    if definition.type not in {"rgb", "rgbd"} or set(spec.config) != {"camera"}:
        raise ValueError("camera binding requires a camera type and native camera name")
    world = context.native._env
    sim = world.sim
    model = sim.model._model
    camera = spec.config["camera"]
    if not world._render or camera not in world._camera_names:
        raise ValueError("Camera must be enabled in the task's simulator configuration")
    camera_id = sim.model.camera_name2id(camera)
    height, width = world._camera_size
    facts = intrinsics(width, height, model.cam_fovy[camera_id], definition.type == "rgbd")

    def read():
        from roboenv_runtime.runtime.worker import write_png

        directory = Path(context.native.run_dir) / "captures"
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{uuid.uuid4().hex}.png"
        extra = {}
        # Use the same clock as joint feedback; receipt time is added by transport.
        acquired = float(sim.data.time)
        if definition.type == "rgbd":
            rgb, depth = world.render_rgbd(camera)
            if np.asarray(depth).shape != np.asarray(rgb).shape[:2]:
                raise ValueError("Depth and RGB image dimensions must match")
            depth_file = file.with_suffix(".pgm")
            write_depth(depth_file, depth)
            extra = {
                "depth_file": str(depth_file),
                "depth_width": width,
                "depth_height": height,
                "depth_sample_time_s": acquired,
            }
        else:
            rgb = world.render(camera)
        h, w, _ = write_png(file, rgb)
        return {"file": str(file), "height": h, "width": w, "sample_time_s": acquired, **extra}

    return finish(spec, definition, context, facts, {"read": action("read", read)})
