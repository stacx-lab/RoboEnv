"""Native camera observations without physics stepping."""

from pathlib import Path


def camera_state(camera, directory, stamp):
    import secrets

    from roboenv_runtime.bindings.backends.shared.depth import write_depth
    from roboenv_runtime.runtime.worker import write_png

    camera.update(0.0, force_recompute=True)
    outputs = camera.data.output
    rgb = outputs["rgb"].torch[0, :, :, :3].cpu().numpy()
    depth = outputs["distance_to_image_plane"].torch[0].cpu().numpy().squeeze(-1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{secrets.token_hex(8)}.png"
    if rgb.shape != (camera.cfg.height, camera.cfg.width, 3) or depth.shape != rgb.shape[:2]:
        raise ValueError("Camera dimensions differ from the profile")
    write_png(file, rgb)
    write_depth(file.with_suffix(".pgm"), depth)
    return {
        "file": str(file),
        "depth_file": str(file.with_suffix(".pgm")),
        "timestamp": stamp,
    }
