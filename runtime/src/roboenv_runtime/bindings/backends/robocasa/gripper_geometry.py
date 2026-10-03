"""Panda mounting-base geometry, fixed TCP and aperture conversion."""

from roboenv.devices.model import local_frame

import numpy as np


def geometry(gripper, sim, joints):
    import mujoco

    model, data = sim.model._model, sim.data._data
    site = sim.model.site_name2id(gripper.important_sites["grip_site"])
    base = int(model.body_parentid[model.site_bodyid[site]])
    base_rotation = data.xmat[base].reshape(3, 3)
    tcp_rotation = base_rotation.T @ data.site_xmat[site].reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, tcp_rotation.ravel())
    tcp = [*np.round(base_rotation.T @ (data.site_xpos[site] - data.xpos[base]), 12).tolist(),
           *np.round(quat[[1, 2, 3, 0]], 12).tolist()]
    pads = [
        sim.model.geom_name2id(gripper.important_geoms[key][0])
        for key in ("left_fingerpad", "right_fingerpad")
    ]
    if len(joints) != 2 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in pads):
        raise ValueError("Panda contact extraction requires two box pads and two sliding joints")
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE for j in joints):
        raise ValueError("Panda aperture extraction requires sliding joints")

    approach = data.site_xmat[site].reshape(3, 3)[:, 2]
    separation = data.geom_xpos[pads[1]] - data.geom_xpos[pads[0]]
    normal = separation - approach * np.dot(separation, approach)
    normal /= np.linalg.norm(normal)
    points = [
        data.geom_xpos[geom] + direction * (
            np.abs(data.geom_xmat[geom].reshape(3, 3).T @ direction) @ model.geom_size[geom]
        )
        for geom, direction in zip(pads, (normal, -normal), strict=True)
    ]
    gap = float(np.dot(points[1] - points[0], normal))
    coefficients = [
        round(
            float(
                sign
                * np.dot(data.xmat[model.jnt_bodyid[j]].reshape(3, 3) @ model.jnt_axis[j], normal)
            ),
            12,
        )
        for j, sign in zip(joints, (-1, 1), strict=True)
    ]
    offset = gap - np.dot(coefficients, data.qpos[model.jnt_qposadr[joints]])
    maximum = offset + sum(
        c * model.jnt_range[j, 1 if c > 0 else 0] for c, j in zip(coefficients, joints, strict=True)
    )
    facts = {
        "frames": {"tcp": local_frame(tcp)},
        "geometry": {
            "description": (
                "Two opposing fingers slide along a fixed closing axis. Each inner pad has a "
                "16 by 16 mm rectangular surface. Aperture is the inner-pad gap; "
                "it does not describe whole-finger clearance."
            ),
            "approach_axis": np.round(tcp_rotation[:, 2], 12).tolist(),
            "closing_axis": np.round(base_rotation.T @ normal, 12).tolist(),
        },
        "aperture_m": {"min": 0.0, "max": round(float(maximum), 12)},
    }
    aperture = {
        "coefficients": coefficients,
        "offset_m": round(float(offset), 12),
    }
    return facts, aperture
