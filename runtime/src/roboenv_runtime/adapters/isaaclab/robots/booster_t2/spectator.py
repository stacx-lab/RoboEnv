"""Owner-only T2 overview, head optics and close-up of both hands."""

from itertools import product

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

from ...common.spectator import SpectatorSource, fit_view, look_at
from .household_models import PORTAL_X, ROOM_BOUNDS


def household_eye(root, eye):
    """Keep an overview inside the occupied room instead of behind its walls."""
    low = np.array([*ROOM_BOUNDS[0], 0.1]) + [0.3, 0.3, 0]
    high = np.array([*ROOM_BOUNDS[1], 3.2]) - [0.3, 0.3, 0]
    if root[0] < PORTAL_X:
        high[0] = PORTAL_X - 0.30
    else:
        low[0] = PORTAL_X + 0.30
    return np.clip(eye, low, high)


def hssd_eye(root, eye):
    """Keep spectator optics inside the occupied HSSD room and below its ceiling."""
    x, y = root[:2]
    if y < -5.7:
        low, high = (-3.5, -6.8, .4), (3.0, -5.8, 2.65)
    elif x > 3.25 and y < -1.7:
        low, high = (3.5, -4.65, .4), (7.4, -1.95, 2.65)
    elif y < -3.0:
        low, high = (2.3, -5.65, .4), (3.05, -3.05, 2.65)
    elif x < 2.2 and .3 < y < 3.3:
        # The dining-room north wall is inside the house's overall bounds.
        low, high = (-2.1, -2.8, .4), (7.4, 2.85, 2.65)
    else:
        low, high = (-2.1, -2.8, .4), (7.4, 8.1, 2.65)
    return np.clip(eye, low, high)


def sources(world):
    config = world.robot.camera

    def overview():
        pos = world.native.root_pose()[0]
        eye = pos + [2.5, -3.0, 1.6]
        if world.definition["version"] in (3, 4):
            eye = household_eye(pos, eye)
        elif world.definition["version"] == 5:
            w, x, y, z = world.native.root_pose()[1]
            yaw = np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
            right = np.cross(forward, [0, 0, 1])
            eye = hssd_eye(pos, pos - 1.8*forward + 1.8*right + [0, 0, 1.45])
            return look_at(eye, pos + .65*forward + [0, 0, -.05])
        return look_at(eye, pos + [0, 0, -0.2])

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
        if world.definition["version"] in (3, 4):
            eye = household_eye(world.native.root_pose()[0], eye)
        elif world.definition["version"] == 5:
            eye = hssd_eye(world.native.root_pose()[0], eye)
        # Include fingers and nearby objects; widen only when the hands separate.
        bounds = (palms[:, None, :] + np.array(list(product((-0.16, 0.16), repeat=3))))
        pose = fit_view(eye, target, bounds.reshape(-1, 3))
        if world.definition["version"] == 5:
            # Framing can move the eye backwards across a previously applied wall limit.
            eye = hssd_eye(world.native.root_pose()[0], matrix(pose)[:3, 3])
            return look_at(eye, target)
        return pose

    fov_y = np.rad2deg(2 * np.arctan(config.height / (2 * config.intrinsics()[1, 1])))
    return {
        "third_person": SpectatorSource(overview),
        "head": SpectatorSource(head, round(720 * config.width / config.height), 720, float(fov_y)),
        "hand": SpectatorSource(hands),
    }
