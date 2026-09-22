"""Compile R1Pro device geometry from a task-free native robot and its URDF."""

import hashlib
from pathlib import Path

import numpy as np

from roboenv_runtime.adapters.gripper_geometry import ParallelGripper

from roboenv_runtime.adapters.urdf_geometry import (
    fixed_transform,
    joint_intrinsics,
    matrix_pose,
    mount,
    serial_chain,
)
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def describe(robot, interval, *, realtime=False):
    from pxr import UsdGeom

    if robot.model != "r1pro" or not robot.is_holonomic_base:
        raise ValueError("BEHAVIOR device integration requires the native R1Pro model")
    urdf = Path(robot.urdf_path)
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
    facts["base"] = {
        "intrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "control": {
                "mode": "motion",
                "body_frame": "base",
                "modes": {
                    "drive": {
                        "velocity": {
                            "frame": "body",
                            "components": [
                                {
                                    "axis": axis,
                                    "units": units,
                                    "bounds": {"min": -bound, "max": bound},
                                }
                                for axis, units, bound in (
                                    ("vx", "m/s", 0.5),
                                    ("vy", "m/s", 0.5),
                                    ("wz", "rad/s", 1.0),
                                )
                            ],
                        },
                        "posture": None,
                    }
                },
            },
            "stop": {"behavior": "halt", "affected_devices": ["base"]},
            "feedback": {
                "pose_frame": "world",
                "velocity_frame": "body",
                "velocity_components": ["vx", "vy", "wz"],
                "clock": "simulation",
            },
        },
        "extrinsics": {"source": "introspected"},
        "type": "velocity",
    }
    groups = {}
    frames = {"base": {"flange": robot.base_footprint_link_name}}
    joint_indices = robot.controller_joint_idx
    for group in robot.controller_order:
        indices = array(joint_indices[group]).astype(int).tolist()
        names = [robot.dof_names_ordered[index] for index in indices]
        if group == "base":
            if len(indices) != 3:
                raise ValueError("Expected a three-axis native holonomic base controller")
            groups["base"] = {"controller": group, "indices": indices, "names": names}
            continue
        name = {
            "arm_left": "left_arm",
            "arm_right": "right_arm",
            "gripper_left": "left_gripper",
            "gripper_right": "right_gripper",
        }.get(group, group)
        groups[name] = {"controller": group, "indices": indices, "names": names}
        if group.startswith("gripper_"):
            continue
        chain, base, flange = serial_chain(urdf, names)
        home = array(robot.reset_joint_pos)[indices].tolist()
        facts[name] = {
            "intrinsics": joint_intrinsics(chain, home, name),
            "type": "joint",
        }
        frames[name] = {"base": base, "flange": flange}
    for name, group in groups.items():
        if not name.endswith("_gripper"):
            continue
        arm = name.removesuffix("_gripper") + "_arm"
        geometry = ParallelGripper(urdf, group["names"], frames[arm]["flange"], "r1pro")
        group["gripper"] = geometry
        facts[name] = geometry.device(name, arm)
    for name, frame in frames.items():
        if name == "base":
            continue
        matches = []
        for parent, parent_frame in frames.items():
            if parent == name:
                continue
            try:
                transform = fixed_transform(urdf, parent_frame["flange"], frame["base"])
            except ValueError:
                continue
            matches.append((parent, transform))
        if len(matches) != 1:
            raise ValueError(f"R1Pro {name} has no unique fixed parent chain")
        parent, transform = matches[0]
        facts[name]["extrinsics"] = mount(
            parent, matrix_pose(transform), "base" if parent == "base" else "flange"
        )
    cameras = {}
    cache = UsdGeom.XformCache()
    for sensor_name, sensor in sorted(robot.sensors.items()):
        if not {"rgb", "depth_linear"}.issubset(sensor.modalities):
            continue
        link_matches = [
            (name, link)
            for name, link in robot.links.items()
            if str(sensor.prim_path).startswith(str(link.prim_path) + "/")
        ]
        if len(link_matches) != 1:
            raise ValueError("Camera must be attached to exactly one robot link")
        link_name, link = link_matches[0]
        relative, _ = cache.ComputeRelativeTransform(sensor.prim, link.prim)
        local = np.asarray(relative, dtype=float).T
        matches = []
        for parent, frame in frames.items():
            try:
                transform = fixed_transform(urdf, frame["flange"], link_name)
            except ValueError:
                continue
            matches.append((parent, transform @ local @ np.diag([1.0, -1.0, -1.0, 1.0])))
        if len(matches) != 1:
            raise ValueError(
                f"Camera crosses an undeclared moving chain: {sensor_name}, link={link_name}, parents={[parent for parent, _ in matches]}, frames={frames}, urdf={urdf.name}"
            )
        parent, transform = matches[0]
        name = "camera_" + str(len(cameras))
        width, height = (sensor.image_width, sensor.image_height)
        k = array(sensor.intrinsic_matrix)
        calibration = intrinsics(
            width, height, np.rad2deg(2 * np.arctan(height / (2 * k[1, 1]))), True
        )
        calibration["color"]["projection"].update(
            fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2])
        )
        calibration["depth"]["reliable_range_m"] = "???"
        facts[name] = {
            "type": "rgbd",
            "intrinsics": calibration,
            "extrinsics": mount(
                parent, matrix_pose(transform), "base" if parent == "base" else "flange"
            ),
        }
        cameras[name] = sensor_name
    if not cameras:
        raise ValueError("R1Pro has no RGB and metric-depth robot cameras")
    for value in facts.values():
        if value["type"] == "joint":
            value["intrinsics"]["frames"] = {value["intrinsics"]["kinematics"]["frame"]: None}
    return (
        facts,
        groups,
        frames,
        cameras,
        {"urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()},
    )
