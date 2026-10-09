"""Conservative impossibility check for the released Panda reaching geometry.

Passing this necessary condition does not certify IK, collision-free paths, or
a feasible schedule. Failing it proves that at least one target cannot be met,
even after removing obstacles and relaxing most joint limits.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def wrist_distance_lower_bound(position, rotation, shoulder, wrist_vectors, *,
                               position_tolerance, angle_tolerance, sampling_error):
    """Minimize a necessary wrist distance over all accepted tool orientations.

    For a fixed wrist-to-tip vector, a rotation within angle_tolerance can turn
    that vector toward the shoulder-to-target direction by at most that angle.
    The law of cosines then gives the minimum distance. Subtract the full position
    tolerance and the arc-length bound between joint-7 samples. This enlarges the
    accepted pose set and can therefore only weaken an impossibility certificate.
    """
    displacement = np.asarray(position) - shoulder
    distance = np.linalg.norm(displacement)
    rotated = wrist_vectors @ Rotation.from_matrix(rotation).as_matrix().T
    lengths = np.linalg.norm(rotated, axis=1)
    if distance == 0:
        return float(lengths.min() - position_tolerance - sampling_error)
    cosine = np.sum(rotated * displacement, axis=1) / (distance * lengths)
    angle = np.maximum(np.arccos(np.clip(cosine, -1., 1.)) - angle_tolerance, 0.)
    required = np.sqrt(np.maximum(
        distance**2 + lengths**2 - 2 * distance * lengths * np.cos(angle), 0.))
    return float(required.min() - position_tolerance - sampling_error)


def unreachable_targets(world, robots, targets, *, position_tolerance, angle_tolerance):
    """Return targets excluded for every arm by the source Panda chain lengths."""
    def body(name):
        return world._get_body_transform(world._model.body(name)).astype(float)

    name = robots[0]
    shoulder = body(name + "/link1")[:3, 3]
    elbow = body(name + "/link4")[:3, 3]
    wrist = body(name + "/link6")[:3, 3]
    # The shoulder/elbow distance is constant because the joint-3 offset is
    # perpendicular to the 316 mm segment. The elbow/wrist link is also rigid.
    reach = np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow)
    joint7 = body(name + "/link7")
    fixed = np.linalg.inv(joint7) @ world.GetRobotTcpTransform(name)
    radius = np.linalg.norm(joint7[:3, 3] - wrist)
    # Allow the whole joint-7 circle, including angles outside its real limits.
    angles = np.linspace(-np.pi, np.pi, 4097)
    circle = radius * np.stack([np.cos(angles), -np.sin(angles), np.zeros_like(angles)], axis=1)
    vectors = (circle + fixed[:3, 3]) @ fixed[:3, :3]
    sampling_error = radius * (angles[1] - angles[0]) / 2
    # Source accessors return float32 transforms. This margin exceeds their
    # observed reconstruction error and is applied on both sides of the bound.
    numeric_margin = 1e-5
    shoulders = [body(robot + "/link1")[:3, 3] for robot in robots]
    excluded = []
    for index, target in enumerate(targets):
        lower = min(wrist_distance_lower_bound(
            target["position"], target["rotation"], origin, vectors,
            position_tolerance=position_tolerance, angle_tolerance=angle_tolerance,
            sampling_error=sampling_error + numeric_margin) for origin in shoulders)
        if lower > reach + numeric_margin:
            excluded.append({"target": index, "gap_lower_m": float(lower - reach - numeric_margin)})
    return excluded
