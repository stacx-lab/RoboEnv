"""Complete source-derived Piper and ALOHA rigs, without scene or GPU creation."""


import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

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

EMBODIMENTS = {"piper": ["piper", "piper", 0.6], "aloha_agilex": ["aloha-agilex"]}
OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def describe(root, embodiment, *, assets=None):
    root = Path(root)
    if embodiment not in EMBODIMENTS:
        raise ValueError("Typed RoboTwin supports Piper and ALOHA-AgileX rigs")
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
        if len(selected) == 3:
            pose[0] += (-1 if index == 0 else 1) * selected[2] / 2
        arm_mount = matrix(pose) @ fixed_transform(urdf, roots[0], base)
        facts[arm] = {
            "intrinsics": joint_intrinsics(model, config["homestate"][index], arm),
            "extrinsics": mount("world", matrix_pose(arm_mount)),
            "type": "joint",
        }
        grip_config = config["gripper_name"][index]
        grip_names = [grip_config["base"], *[entry[0] for entry in grip_config["mimic"]]]
        geometry = ParallelGripper(
            urdf, grip_names, flange, "piper" if embodiment == "piper" else "aloha"
        )
        facts[gripper] = geometry.device(gripper, arm)
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
