"""Pose and side conventions shared by T2 low-level control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")
"""Arm sides, in the order every per-side list uses."""


@dataclass(frozen=True)
class Pose:
    """A rigid pose. The frame is stated wherever a `Pose` is used."""

    pos: np.ndarray
    """Position. Shape (3,), m."""
    quat: np.ndarray
    """Orientation wxyz, unit, w >= 0. Shape (4,)."""

    @classmethod
    def of(cls, pos, quat) -> Pose:
        """A pose from any position and quaternion; the quaternion is normalized and its sign set
        to w >= 0."""
        q = np.asarray(quat, dtype=float)
        q = q / np.linalg.norm(q)
        return cls(np.asarray(pos, dtype=float), -q if q[0] < 0 else q)
