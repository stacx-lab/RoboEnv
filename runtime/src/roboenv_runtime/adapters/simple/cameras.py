"""Validate Isaac optical calibration against the portable robot profile."""

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix


def validate_isaac(world):
    renderer = world.env.isaac
    if renderer is None:
        return
    # USD authored transforms do not track the live articulation. MuJoCo owns
    # the physics state mirrored into Isaac, including the measured torso frame.
    parent = matrix(world._pose("torso_link"))
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
        actual = np.linalg.inv(parent) @ optical
        if not np.allclose(actual, expected, atol=1e-4, rtol=0):
            translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            rotation = actual[:3, :3] @ expected[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            raise ValueError(
                f"SIMPLE Isaac camera mount differs: {name} "
                f"(translation {translation:g} m, rotation {angle:g} rad)"
            )
