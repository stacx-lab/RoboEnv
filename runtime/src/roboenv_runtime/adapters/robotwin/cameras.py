"""Pixel-center depth beside RoboTwin's supersampled ray-traced RGB."""

from pathlib import Path

import numpy as np


def pixel_center_depth(world, name, camera):
    # The RT Position buffer contains the last Halton sample, not the pixel center.
    # A raster camera uses the same optical projection without changing native RGB.
    import sapien

    if name not in world._depth_cameras:
        depth = sapien.render.RenderCameraComponent(
            camera.get_width(), camera.get_height(),
            shader_dir=str(Path(sapien.__file__).parent / "vulkan_shader/default"),
        )
        entity = sapien.Entity()
        entity.name = "roboenv_depth_" + name
        entity.add_component(depth)
        world.task.scene.add_entity(entity)
        world._depth_cameras[name] = depth
    depth = world._depth_cameras[name]
    k = camera.get_intrinsic_matrix()
    depth.set_perspective_parameters(
        camera.get_near(), camera.get_far(),
        float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), float(k[0, 1]),
    )
    depth.entity.set_pose(camera.get_global_pose())
    world.task.scene.update_render()
    depth.take_picture()
    position = np.asarray(depth.get_picture("Position"))
    z = -position[..., 2].copy()
    z[(position[..., 3] >= 1) | ~np.isfinite(z) | (z <= 0)] = 0
    return z
