"""Owner-only T2 overview, head optics and close-up of both hands."""

from itertools import product

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix_pose

from ...common.spectator import SpectatorSource, fit_view, look_at


def sources(world):
    config = world.robot.camera

    def overview():
        pos = world.native.root_pose()[0]
        return look_at(pos + [2.5, -3.0, 1.6], pos + [0, 0, -0.2])

    def head():
        transform = world.native.head_camera_pose().copy()
        # Float32 native quaternions can produce slightly non-orthogonal matrices.
        # Correct only this spectator pose, leaving physics and device RGB unchanged.
        u, _, vh = np.linalg.svd(transform[:3, :3])
        transform[:3, :3] = u @ vh
        return matrix_pose(transform)

    def hands():
        palms, _ = world.native.link_poses(["left_palm", "right_palm"])
        target = np.mean(palms, axis=0)
        w, x, y, z = world.native.root_pose()[1]
        # Extract yaw without tilting the horizon when the robot leans or falls.
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.cross(forward, [0, 0, 1])
        eye = target + 0.62 * forward + 0.65 * right + [0, 0, 0.42]
        # Include fingers and nearby objects; widen only when the hands separate.
        bounds = (palms[:, None, :] + np.array(list(product((-0.16, 0.16), repeat=3))))
        return fit_view(eye, target, bounds.reshape(-1, 3))

    fov_y = np.rad2deg(2 * np.arctan(config.height / (2 * config.intrinsics()[1, 1])))
    return {
        "third_person": SpectatorSource(overview),
        "head": SpectatorSource(head, round(720 * config.width / config.height), 720, float(fov_y)),
        "hand": SpectatorSource(hands),
    }
