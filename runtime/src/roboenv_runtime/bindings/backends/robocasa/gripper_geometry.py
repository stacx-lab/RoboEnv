"""Panda pad surfaces expressed in the simulator's existing grip-site TCP."""

import numpy as np


def geometry(gripper, sim, joints):
    import mujoco

    model, data = sim.model._model, sim.data._data
    site = sim.model.site_name2id(gripper.important_sites["grip_site"])
    pads = [
        sim.model.geom_name2id(gripper.important_geoms[key][0])
        for key in ("left_fingerpad", "right_fingerpad")
    ]
    if len(joints) != 2 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in pads):
        raise ValueError("Panda contact extraction requires two box pads and two sliding joints")
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE for j in joints):
        raise ValueError("Panda aperture extraction requires sliding joints")

    def surfaces():
        rotation = data.site_xmat[site].reshape(3, 3)
        approach = rotation[:, 2]
        separation = data.geom_xpos[pads[1]] - data.geom_xpos[pads[0]]
        normal = separation - approach * np.dot(separation, approach)
        normal /= np.linalg.norm(normal)
        points, frames, sizes = [], {}, {}
        for name, geom, direction in zip(
            ("left_pad", "right_pad"), pads, (normal, -normal), strict=True
        ):
            axes = data.geom_xmat[geom].reshape(3, 3)
            size = model.geom_size[geom]
            tangent = np.cross(approach, direction)
            patch = np.column_stack((tangent, approach, direction))
            point = data.geom_xpos[geom] + direction * (np.abs(axes.T @ direction) @ size)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rotation.T @ patch).ravel())
            frames[name] = [
                *(rotation.T @ (point - data.site_xpos[site])).tolist(),
                *quat[1:].tolist(),
                float(quat[0]),
            ]
            sizes[name] = [
                round(float(2 * (np.abs(axes.T @ axis) @ size)), 12) for axis in (tangent, approach)
            ]
            points.append(point)
        return frames, sizes, normal, float(np.dot(points[1] - points[0], normal))

    frames, sizes, normal, gap = surfaces()
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
        "frames": dict.fromkeys(frames),
        "geometry": {
            "description": (
                "TCP is grip_site, 97 mm from the flange. Parallel jaws; contact frames locate "
                "the inner pad surfaces. Contact +Z points toward the object; local XY bounds "
                "describe each pad, not collision clearance."
            ),
            "approach_axis": [0, 0, 1],
            "closing_axis": np.round(data.site_xmat[site].reshape(3, 3).T @ normal, 12).tolist(),
            "contacts": {name: {"frame": name, "size_m": sizes[name]} for name in frames},
        },
        "aperture_m": {"min": 0.0, "max": round(float(maximum), 12)},
    }
    aperture = {
        "model": "affine",
        "coefficients": coefficients,
        "offset_m": round(float(offset), 12),
    }
    return facts, aperture, lambda: surfaces()[0]
