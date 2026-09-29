"""Task-free robot and optical geometry from the pinned RoboDojo assets."""

import ast
import hashlib
import re
import runpy
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


def camera_intrinsics(template):
    """Describe the native tiled renderer's horizontal-fit, square-pixel projection."""
    width, height = template["resolution"]
    fx = width * template["focal_length"] / template["horizontal_aperture"]
    fy = fx

    def ray(u, v):
        return np.array([(u - width / 2) / fx, (v - height / 2) / fy, 1.0])

    def angle(a, b):
        return float(
            np.rad2deg(
                np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            )
        )

    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "color": {
            "width": width,
            "height": height,
            "fps": None,
            "pixel_format": "rgb8",
            "image_format": "png",
            "projection": {
                "model": "pinhole",
                "fx": fx,
                "fy": fy,
                "cx": width / 2,
                "cy": height / 2,
                "skew": 0,
            },
            "distortion": {"model": "none"},
            "fov": {
                "horizontal_deg": angle(
                    ray(-0.5, (height - 1) / 2), ray(width - 0.5, (height - 1) / 2)
                ),
                "vertical_deg": angle(
                    ray((width - 1) / 2, -0.5), ray((width - 1) / 2, height - 0.5)
                ),
            },
        },
    }


def euler_pose(position, degrees):
    result = np.eye(4)
    for axis, angle in enumerate(np.deg2rad(degrees)):
        quaternion = [0.0, 0.0, 0.0, float(np.cos(angle / 2))]
        quaternion[axis] = float(np.sin(angle / 2))
        result = result @ matrix([0.0, 0.0, 0.0, *quaternion])
    result[:3, 3] = position
    return result


def describe(root, embodiment, *, assets=None):
    root = Path(root)
    if embodiment not in {"arx_x5", "yam"}:
        raise ValueError("RoboDojo supports ARX X5 and YAM source rigs")
    model_name = "x5" if embodiment == "arx_x5" else "yam"
    assets = Path(assets) if assets else root / "Assets"
    directory = assets / "Robots" / model_name
    config_path = directory / "robot_config.yml"
    config = yaml.safe_load(config_path.read_text())
    urdf = directory / config["urdf_path"]
    model, base, flange = serial_chain(urdf, config["arm_joints_name"], base=config["base_link"])
    actuator_path = root / f"env/robot_manager/robot_config/{model_name}.py"
    actuators = []
    for node in ast.walk(ast.parse(actuator_path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and (node.func.id == "ImplicitActuatorCfg")
        ):
            actuators.append({item.arg: ast.literal_eval(item.value) for item in node.keywords})
    for joint in model["joints"]:
        matches = [
            actuator
            for actuator in actuators
            if any(re.fullmatch(pattern, joint["name"]) for pattern in actuator["joint_names_expr"])
        ]
        if len(matches) != 1:
            raise ValueError("Each native joint requires one source actuator configuration")
        for field, native in (
            ("max_velocity", "velocity_limit_sim"),
            ("max_effort", "effort_limit_sim"),
        ):
            joint[field] = matches[0].get(native, joint[field])
    robots = yaml.safe_load((root / f"env_cfg/robot/dual_{model_name}.yml").read_text())["robots"]
    templates = runpy.run_path(str(root / "env_cfg/camera/template.py"))
    camera_config = yaml.safe_load((root / "env_cfg/camera/camera_config.yml").read_text())
    optical = np.diag([1.0, -1.0, -1.0, 1.0])
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

    def camera_fact(config, transform, parent, frame=None):
        if config["mesh"] != "pinhole":
            raise ValueError("A mesh camera requires its own source-derived optical transform")
        key = {"d435": "D435", "Gemini_345Lg": "GEMINI_345LG"}[config["type"]]
        template = templates[key]
        return {
            "type": "rgb",
            "intrinsics": camera_intrinsics(template),
            "extrinsics": mount(parent, matrix_pose(transform @ optical), frame),
        }

    for side, robot in zip(("left", "right"), robots, strict=True):
        arm, gripper = (side + "_arm", side + "_gripper")
        quat = robot["default_root_rot"]
        pose = [*robot["default_root_pos"], *quat[1:], quat[0]]
        facts[arm] = {
            "intrinsics": joint_intrinsics(model, [0.0] * len(model["joints"]), arm),
            "extrinsics": mount("world", matrix_pose(matrix(pose))),
            "type": "joint",
        }
        geometry = ParallelGripper(urdf, config["gripper_joints_name"], flange, model_name)
        facts[gripper] = geometry.device(gripper, arm)
        camera = next(value for value in config["camera"] if value["name"] == "cam_wrist")
        transform = fixed_transform(urdf, flange, camera["link"])
        transform = transform @ euler_pose(camera["pos"], camera["ori"])
        facts[side + "_camera"] = camera_fact(camera, transform, arm, "flange")
        frames[side] = {"base": base, "flange": flange, "gripper": geometry}
    head = camera_config["cam_head"]["camera"]
    facts["head_camera"] = camera_fact(head, euler_pose(head["pos"], head["ori"]), "world")
    provenance = {
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "actuators_sha256": hashlib.sha256(actuator_path.read_bytes()).hexdigest(),
    }
    for value in facts.values():
        if value["type"] == "joint":
            value["intrinsics"]["frames"] = {value["intrinsics"]["kinematics"]["frame"]: None}
    return (facts, frames, provenance)
