"""Validate Isaac optical calibration against the portable robot profile."""

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix


def validate_isaac(world):
    renderer = world.env.isaac
    if renderer is None:
        return
    from omni.isaac.core.prims import XFormPrim

    torso = XFormPrim(renderer.robot.prim_path + "/torso_link")
    position, quaternion = torso.get_world_pose()
    parent = matrix([*position, *quaternion[[1, 2, 3, 0]]])
    for name, spec in world.model["cameras"].items():
        camera = renderer.cameras[spec["native"]]
        if tuple(camera.get_resolution()) != (spec["width"], spec["height"]):
            raise ValueError(f"SIMPLE Isaac camera dimensions differ: {name}")
        projection = spec["projection"]
        expected = [
            [projection["fx"], 0, projection["cx"]],
            [0, projection["fy"], projection["cy"]],
            [0, 0, 1],
        ]
        native = np.array(camera.get_intrinsics_matrix(), copy=True)
        # Native matrices use pixel-edge coordinates; catalog pixels have integer centers.
        native[:2, 2] -= 0.5
        if not np.allclose(native, expected, atol=1e-4, rtol=0):
            raise ValueError(f"SIMPLE Isaac camera intrinsics differ: {name}")
        position, quaternion = camera.get_world_pose(camera_axes="ros")
        optical = matrix([*position, *quaternion[[1, 2, 3, 0]]])
        mount = spec["mount"]
        expected = matrix(mount["translation_m"] + mount["rotation_xyzw"])
        if not np.allclose(np.linalg.inv(parent) @ optical, expected, atol=1e-4, rtol=0):
            raise ValueError(f"SIMPLE Isaac camera mount differs: {name}")
