"""Hand-local geometric references from one native MuJoCo sample."""

import numpy as np

from ..urdf_geometry import matrix, matrix_pose


def frame_poses(world, name):
    part = world.model["parts"][name]
    geometry = part["geometry"]
    parent = np.linalg.inv(matrix(world._pose(part["base"])))
    frames = {
        finger["tip_frame"]: matrix_pose(
            parent @ matrix(world._pose(part["frame_bodies"][finger["tip_frame"]]))
            @ matrix(finger["tip_offset"])
        )
        for finger in geometry["fingers"].values()
    }
    center = geometry["center"]
    frames[center["frame"]] = [
        sum(weight * frames[geometry["fingers"][finger]["tip_frame"]][axis]
            for finger, weight in center["weights"].items())
        for axis in range(3)
    ] + center["rotation_xyzw"]
    return frames
