"""Small xyzw pose operations at the device boundary (Isaac Lab 3 uses xyzw)."""

import numpy as np


def rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def relative_pose(parent, child):
    a = np.asarray(parent[3:]) * [-1, -1, -1, 1]
    b = np.asarray(child[3:])
    q = np.r_[
        a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])
    ]
    p = rotation(parent[3:]).T @ (np.asarray(child[:3]) - parent[:3])
    return np.r_[p, q].tolist()
