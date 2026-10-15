"""Complete source-derived ALOHA AgileX rig, without scene or GPU creation."""


import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from roboenv.devices.model import local_frame
from roboenv_runtime.adapters.gripper_geometry import ParallelGripper
from roboenv_runtime.adapters.urdf_geometry import (
    fixed_transform,
    joint_intrinsics,
    matrix,
    matrix_pose,
    mount,
    serial_chain,
)
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics as camera_intrinsics

EMBODIMENTS = {"aloha_agilex": ["aloha-agilex"]}
OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def native_tcp(urdf, config, index, flange):
    """Flange-to-TCP transform matching RoboTwin Robot._trans_endpose."""
    joint = next(j for j in ET.parse(urdf).getroot().findall("joint")
                 if j.get("name") == config["ee_joints"][index])
    transform = fixed_transform(urdf, flange, joint.find("child").get("link"))
    if config.get("ee_frame", "joint") == "joint":
        # SAPIEN's URDF loader uses this basis for the joint's child anchor.
        # Even a +X URDF axis has a rotated anchor; it is not the link frame.
        node = joint.find("axis")
        axis = np.fromstring(node.get("xyz", "1 0 0") if node is not None else "1 0 0", sep=" ")
        axis /= np.linalg.norm(axis)
        tangent = np.cross(axis, [0, 0, 1] if abs(axis[0]) > 0.9 else [1, 0, 0])
        tangent /= np.linalg.norm(tangent)
        transform[:3, :3] = transform[:3, :3] @ np.column_stack(
            (axis, tangent, np.cross(axis, tangent))
        )
    elif config["ee_frame"] != "link":
        raise ValueError("Unsupported RoboTwin end-effector frame")
    transform[:3, :3] = (transform[:3, :3]
                        @ np.asarray(config.get("global_trans_matrix", np.eye(3)))
                        @ np.asarray(config.get("delta_matrix", np.eye(3))))
    transform[:3, 3] += transform[:3, :3] @ [config["gripper_bias"], 0, 0]
    return transform


def describe(root, embodiment, *, assets=None):
    root = Path(root)
    if embodiment not in EMBODIMENTS:
        raise ValueError("RoboTwin supports only the ALOHA AgileX rig")
    selected = EMBODIMENTS[embodiment]
    assets = Path(assets) if assets else root / "assets/embodiments"
    directory = assets / selected[0]
    config_path = directory / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    urdf = directory / config["urdf_path"]
    tree = ET.parse(urdf).getroot()
    child_links = {joint.find("child").attrib["link"] for joint in tree.findall("joint")}
    roots = [
        link.attrib["name"]
        for link in tree.findall("link")
        if link.attrib["name"] not in child_links
    ]
    if len(roots) != 1:
        raise ValueError("Robot description must have one root link")
    camera_config = yaml.safe_load((root / "env_cfg/task_config/_camera_config.yml").read_text())[
        "D435"
    ]
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
    frames = {}
    for index, side in enumerate(("left", "right")):
        arm, gripper = (side + "_arm", side + "_gripper")
        model, base, flange = serial_chain(urdf, config["arm_joints_name"][index])
        # SAPIEN imports joint position ranges, not URDF velocity/effort ratings.
        for joint in model["joints"]:
            joint["max_velocity"] = joint["max_effort"] = None
        root_pose = config["robot_pose"][0]
        pose = [*root_pose[:3], *root_pose[4:7], root_pose[3]]
        arm_mount = matrix(pose) @ fixed_transform(urdf, roots[0], base)
        facts[arm] = {
            "intrinsics": joint_intrinsics(model, config["homestate"][index], arm),
            "extrinsics": mount("world", matrix_pose(arm_mount)),
            "type": "joint",
        }
        grip_config = config["gripper_name"][index]
        grip_names = [grip_config["base"], *[entry[0] for entry in grip_config["mimic"]]]
        geometry = ParallelGripper(urdf, grip_names, flange, "aloha")
        facts[gripper] = geometry.device(gripper, arm)
        facts[gripper]["intrinsics"]["frames"]["tcp"] = local_frame(matrix_pose(
            np.linalg.inv(geometry.mount) @ native_tcp(urdf, config, index, flange)
        ))
        links = {link.attrib["name"] for link in tree.findall("link")}
        camera_link = side + "_camera" if side + "_camera" in links else "camera"
        camera_mount = fixed_transform(urdf, flange, camera_link) @ OPTICAL
        facts[side + "_camera"] = {
            "type": "rgbd",
            "intrinsics": camera_intrinsics(
                camera_config["w"], camera_config["h"], camera_config["fovy"], True
            ),
            "extrinsics": mount(arm, matrix_pose(camera_mount), "flange"),
        }
        frames[side] = {
            "base": base,
            "flange": flange,
            "gripper": geometry,
        }
    camera = next(item for item in config["static_camera_list"] if item["name"] == "head_camera")
    forward = np.asarray(camera["forward"], dtype=float)
    left = np.asarray(camera["left"], dtype=float)
    forward /= np.linalg.norm(forward)
    left /= np.linalg.norm(left)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.column_stack((forward, left, np.cross(forward, left)))
    camera_pose[:3, 3] = camera["position"]
    facts["head_camera"] = {
        "type": "rgbd",
        "intrinsics": camera_intrinsics(
            camera_config["w"], camera_config["h"], camera_config["fovy"], True
        ),
        "extrinsics": mount("world", matrix_pose(camera_pose @ OPTICAL)),
    }
    provenance = {
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    return (facts, frames, provenance)
