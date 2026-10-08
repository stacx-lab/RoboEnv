"""Third-person composition; tasks declare fixed workspaces, otherwise follow XY."""

import itertools

import numpy as np

from ...common.spectator import SpectatorSource, fit_view, look_at


def workspace(eye, target, positions):
    """Frame a starting robot region plus the task's interaction regions."""
    points = list(itertools.product((-0.65, 0.85), (-0.65, 0.65), (0.0, 1.15)))
    for position in positions:
        points.extend(
            (np.array(position) + offset).tolist()
            for offset in itertools.product((-0.45, 0.45), (-0.65, 0.65), (0, 1.15))
        )
    return fit_view(eye, target, points)


def sources(environment):
    native = environment.native
    fixed = native.scene.spectator_pose()
    if fixed is None:

        def pose():
            shift = native.root_pose()[:3].copy()
            shift[2] = 0.0
            return look_at(np.array([1.6, -2.2, 1.4]) + shift, np.array([0.3, 0.0, 0.4]) + shift)
    else:

        def pose():
            return fixed.copy()

    return {"third_person": SpectatorSource(pose)}
