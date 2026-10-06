"""Source-derived YAM geometry, shared by generation and live inventory checks."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from roboenv.devices.model import local_frame
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.adapters.robocasa.geometry import chain, pose
from roboenv_runtime.adapters.urdf_geometry import (
    gripper_intrinsics,
    joint_intrinsics,
    matrix_pose,
    mount,
)
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics as camera_intrinsics

CAMERA_SIZE = (480, 640)


def canonical(value):
    if isinstance(value, float):
        return round(value, 12) or 0.0
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def task_free_model(path):
    """Compile only the fixed camera mount and robot subtrees, never task objects."""
    path = Path(path)
    tree = ET.parse(path).getroot()
    world = tree.find("worldbody")
    for element in list(world):
        if element.tag != "body" or element.get("name") not in {
            "left_arm",
            "right_arm",
            "gate_collision",
        }:
            world.remove(element)
    for tag in ("keyframe", "contact", "sensor", "tendon"):
        for element in tree.findall(tag):
            tree.remove(element)
    for tag in ("actuator", "equality"):
        group = tree.find(tag)
        if group is not None:
            for element in list(group):
                selector = element.get("joint", element.get("joint1", ""))
                if not selector.startswith(("left_", "right_")):
                    group.remove(element)
    # Use absolute asset paths for this maintenance-only compile.
    compiler = tree.find("compiler")
    for key in ("meshdir", "texturedir"):
        compiler.set(key, str((path.parent / compiler.get(key, "")).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))


def finger_pads(model, side):
    ids = []
    for suffix in ("lf_down", "rf_down"):
        body = model.body(side + "_" + suffix).id
        choices = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and np.allclose(model.geom_size[g], [0.006, 0.002, 0.04])
        ]
        if len(choices) != 1:
            raise ValueError("YAM contact pad geometry differs from the supported rig")
        ids.append(choices[0])
    return ids


def aperture(model, data, side):
    a, b = finger_pads(model, side)
    # Parallel pad faces: project their centre separation onto the pad normal.
    axis = data.geom_xmat[a].reshape(3, 3)[:, 1]
    return max(
        0.0,
        float(
            abs(np.dot(data.geom_xpos[b] - data.geom_xpos[a], axis))
            - model.geom_size[a, 1]
            - model.geom_size[b, 1]
        ),
    )


def describe(model):
    from abc_sim.config import _base_config

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = _base_config()
    facts = {
        "world": {
            "type": "world",
            "intrinsics": {
                "source": "default",
                "method": "unknown",
                "calibrated_at": None,
                "bounds": None,
                "surfaces": [],
            },
            "extrinsics": {},
        }
    }
    for side in ("left", "right"):
        name = side + "_arm"
        base, flange = model.body(name).id, model.body(side + "_link_6").id
        joints = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        geometry = {
            "format": "serial_chain",
            "chain": chain(model, joints, flange, base)["chain"],
            "joints": [
                {
                    "name": model.joint(j).name,
                    "type": "revolute",
                    "limits": {
                        "min": float(model.jnt_range[j, 0]),
                        "max": float(model.jnt_range[j, 1]),
                    },
                    "max_velocity": None,
                    "max_effort": None,
                }
                for j in joints
            ],
        }
        facts[name] = {
            "type": "joint",
            "intrinsics": joint_intrinsics(geometry, config.robots[side].init_q[:6], name),
            "extrinsics": mount("world", pose(model, data, base)),
        }
        grip = side + "_gripper"
        intrinsics = gripper_intrinsics(grip)
        site = model.site(side + "_grasp_site").id
        tcp = [*model.site_pos[site].tolist(), *model.site_quat[site][[1, 2, 3, 0]].tolist()]
        intrinsics["frames"] = {"tcp": local_frame(tcp)}
        a, b = finger_pads(model, side)
        rotation = data.xmat[flange].reshape(3, 3).T
        axis = rotation @ data.geom_xmat[a].reshape(3, 3)[:, 1]
        intrinsics["geometry"] = {
            "description": "Opposing sliding fingers with box pads and native capsule/sphere contacts. "
            "Aperture measures separation between the main parallel pad faces.",
            "approach_axis": [0.0, 0.0, 1.0],
            "closing_axis": axis.tolist(),
        }
        limits = []
        for target in (0.0, 0.0475):
            sample = mujoco.MjData(model)
            sample.qpos[model.joint(side + "_left_finger").qposadr[0]] = target
            sample.qpos[model.joint(side + "_right_finger").qposadr[0]] = -target
            mujoco.mj_forward(model, sample)
            limits.append(aperture(model, sample, side))
        if limits[1] <= limits[0]:
            raise ValueError("YAM normalized open direction differs from its source protocol")
        intrinsics["aperture_m"] = {"min": limits[0], "max": limits[1]}
        facts[grip] = {
            "type": "gripper",
            "intrinsics": intrinsics,
            "extrinsics": mount(name, [0, 0, 0, 0, 0, 0, 1], "flange"),
        }
    for camera in ("top", "left", "right"):
        cid = model.camera(camera).id
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = data.cam_xpos[cid]
        parent, frame = "world", "world"
        if camera != "top":
            parent, frame = camera + "_arm", "flange"
            base = np.eye(4)
            body = model.body(camera + "_link_6").id
            base[:3, :3], base[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
            transform = np.linalg.inv(base) @ transform
        facts["camera_" + camera] = {
            "type": "rgb",
            "intrinsics": camera_intrinsics(640, 480, model.cam_fovy[cid], False),
            "extrinsics": mount(parent, matrix_pose(transform), frame),
        }
    return canonical(facts)


def rig_id(facts):
    return "abc-yam-" + digest(facts)[:10]
