"""Native camera observations without physics stepping."""

from pathlib import Path


def camera_state(camera, directory, stamp, *, rgb_size=None):
    import secrets

    from roboenv_runtime.bindings.backends.shared.depth import write_depth
    from roboenv_runtime.runtime.worker import write_png

    camera.update(0.0, force_recompute=True)
    outputs = camera.data.output
    rgb_data = outputs["rgb"]
    rgb = getattr(rgb_data, "torch", rgb_data)[0, :, :, :3].cpu().numpy()
    if rgb.shape != (camera.cfg.height, camera.cfg.width, 3):
        raise ValueError("Camera dimensions differ from the render configuration")
    if rgb_size is not None:
        if "distance_to_image_plane" in outputs:
            raise ValueError("Supersampling is supported only for RGB cameras")
        width, height = rgb_size
        if (type(width) is not int or type(height) is not int
                or not 0 < width <= camera.cfg.width or not 0 < height <= camera.cfg.height
                or camera.cfg.width % width or camera.cfg.height % height):
            raise ValueError("RGB output size must divide the rendered image dimensions")
        # Integer-factor area integration; no interpolation invents marker edges.
        blocks = rgb.reshape(height, camera.cfg.height // height,
                             width, camera.cfg.width // width, 3)
        rgb = (blocks.mean(axis=(1, 3)) + 0.5).astype(rgb.dtype)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{secrets.token_hex(8)}.png"
    write_png(file, rgb)
    result = {"file": str(file), "timestamp": stamp}
    if "distance_to_image_plane" in outputs:
        depth_data = outputs["distance_to_image_plane"]
        depth = getattr(depth_data, "torch", depth_data)[0].cpu().numpy().squeeze(-1)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("Depth dimensions differ from the profile")
        write_depth(file.with_suffix(".pgm"), depth)
        result["depth_file"] = str(file.with_suffix(".pgm"))
    return result
