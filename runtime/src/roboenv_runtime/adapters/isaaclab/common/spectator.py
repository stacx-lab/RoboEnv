"""Owner-only view definitions and camera geometry, without simulator imports."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix_pose


@dataclass(frozen=True)
class SpectatorSource:
    pose: Callable[[], list[float]]
    width: int = 1280
    height: int = 720
    fov_y: float = 50.0


def look_at(eye, target):
    """World pose with OpenCV optical axes: right, down, forward; world Z is up."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all():
        raise ValueError("Camera eye and target must be finite XYZ vectors")
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Camera eye and target must differ")
    forward /= norm
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Camera direction must not be parallel to world up")
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    transform[:3, 3] = eye
    return matrix_pose(transform)


def fit_view(eye, target, points, *, fov_y=50.0, aspect=16 / 9, margin=0.1):
    """Move back along the same ray only when bounds exceed the safe image area."""
    from roboenv_runtime.adapters.urdf_geometry import matrix

    transform = matrix(look_at(eye, target))
    relative = (np.asarray(points, dtype=float) - transform[:3, 3]) @ transform[:3, :3]
    tangent = np.tan(np.deg2rad(fov_y / 2)) * (1 - 2 * margin)
    extra = max(
        0.0,
        float(
            np.max(
                np.maximum(
                    np.abs(relative[:, 0]) / (tangent * aspect), np.abs(relative[:, 1]) / tangent
                )
                - relative[:, 2]
            )
        ),
        float(np.max(0.05 - relative[:, 2])),
    )
    return look_at(transform[:3, 3] - extra * transform[:3, 2], target)
