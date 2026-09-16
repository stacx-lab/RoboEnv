"""Read measured transforms and export static geometry; no kinematic solvers."""

import numpy as np


def vector(value, size, *, lower=None, upper=None):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite numbers")
    if lower is not None and (result < lower).any():
        raise ValueError("Target below device limits")
    if upper is not None and (result > upper).any():
        raise ValueError("Target above device limits")
    return result


def pose(model, data, body, base=0):
    import mujoco

    rotation = data.xmat[base].reshape(3, 3).T
    position = rotation @ (data.xpos[body] - data.xpos[base])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, (rotation @ data.xmat[body].reshape(3, 3)).ravel())
    return [*position.tolist(), *quaternion[1:].tolist(), float(quaternion[0])]


def chain(model, joints, body, base):
    """Export this device's fixed origins, axes, anchors and joint references."""
    bodies = []
    while body != base:
        if body == 0:
            raise ValueError("JointDevice base must be an ancestor of its flange")
        bodies.append(body)
        body = int(model.body_parentid[body])
    indexes = {int(j): i for i, j in enumerate(joints)}
    links = []
    for body in reversed(bodies):
        origin = model.body_pos[body].tolist() + model.body_quat[body][[1, 2, 3, 0]].tolist()
        attached = range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body])
        for j in attached:
            if j not in indexes or model.jnt_type[j] not in (2, 3):
                raise ValueError("Device chain contains an undeclared or non-scalar joint")
            links.append(
                {
                    "origin": origin,
                    "joint": indexes[j],
                    "axis": model.jnt_axis[j].tolist(),
                    "anchor": model.jnt_pos[j].tolist(),
                    "reference": float(model.qpos0[model.jnt_qposadr[j]]),
                }
            )
            origin = [0, 0, 0, 0, 0, 0, 1]
        if not model.body_jntnum[body]:
            links.append({"origin": origin})
    return {"source": "MuJoCo compiled model", "chain": links}
