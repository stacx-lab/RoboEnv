"""Native camera observations without physics stepping."""

from pathlib import Path


def camera_state(camera, directory, stamp):
    import uuid

    from roboenv_runtime.bindings.backends.shared.depth import write_depth
    from roboenv_runtime.runtime.worker import write_png

    camera.update(0.0, force_recompute=True)
    outputs = camera.data.output
    rgb = outputs["rgb"].torch[0, :, :, :3].cpu().numpy()
    depth = outputs["distance_to_image_plane"].torch[0].cpu().numpy().squeeze(-1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{uuid.uuid4().hex}.png"
    h, w, _ = write_png(file, rgb)
    write_depth(file.with_suffix(".pgm"), depth)
    return {
        "file": str(file),
        "depth_file": str(file.with_suffix(".pgm")),
        "height": h,
        "width": w,
        "sample_time_s": stamp,
        "depth_width": w,
        "depth_height": h,
        "depth_sample_time_s": stamp,
    }
