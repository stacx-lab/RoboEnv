"""Body-frame conversion for the pinned native mobile-base velocity controller."""

import numpy as np


def native_command(controller, target):
    # The native controller rotates input by -theta into its initial joint
    # frame; body-frame translation requires +theta. Pre-rotate by 2*theta.
    # Apply on every interval so a retained command follows the turning body.
    _, current = controller.get_base_pose()
    initial = controller.init_ori
    theta = np.arctan2(current[1, 0], current[0, 0]) - np.arctan2(initial[1, 0], initial[0, 0])
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    value = np.array(target, dtype=float, copy=True)
    value[:2] = np.array([[c, -s], [s, c]]) @ value[:2]
    return value
