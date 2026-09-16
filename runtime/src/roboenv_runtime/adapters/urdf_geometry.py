"""Extract portable serial chains from pinned robot descriptions without an engine."""

import math
import xml.etree.ElementTree as ET

import numpy as np

IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) == 0:
        raise ValueError("Pose requires finite xyz and a nonzero xyzw quaternion")
    x, y, z, w = np.asarray(pose[3:], dtype=float) / np.linalg.norm(pose[3:])
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return result


def matrix_pose(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-07, rtol=0))
        or (not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-06, rtol=0))
        or (not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-06, rtol=0))
    ):
        raise ValueError("Expected a finite rigid transform with a proper rotation")
    r = np.asarray(value, dtype=float)[:3, :3]
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
            [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
            [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
        ]
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return np.asarray(value)[:3, 3].tolist() + q.tolist()


def fixed_transform(path, base, child):
    nodes = ET.parse(path).getroot().findall("joint")
    by_child = {joint.find("child").attrib["link"]: joint for joint in nodes}
    result, visited = (np.eye(4), set())
    while child != base:
        if child in visited or child not in by_child:
            raise ValueError("No fixed URDF path between these frames")
        visited.add(child)
        joint = by_child[child]
        if joint.attrib["type"] != "fixed":
            raise ValueError("Mount path crosses a moving joint")
        result = matrix(origin(joint)) @ result
        child = joint.find("parent").attrib["link"]
    return result


def origin(joint):
    element = joint.find("origin")
    if element is None:
        return list(IDENTITY)
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    if len(xyz) != 3 or len(rpy) != 3 or (not all(math.isfinite(v) for v in xyz + rpy)):
        raise ValueError("URDF origin requires finite xyz and rpy triples")
    roll, pitch, yaw = [v / 2 for v in rpy]
    cr, cp, cy = (math.cos(roll), math.cos(pitch), math.cos(yaw))
    sr, sp, sy = (math.sin(roll), math.sin(pitch), math.sin(yaw))
    return xyz + [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def serial_chain(path, names, *, base=None, flange=None):
    root = ET.parse(path).getroot()
    nodes = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    if not names or len(set(names)) != len(names) or any(name not in nodes for name in names):
        raise ValueError("JointDevice requires distinct declared URDF joints")
    base = base or nodes[names[0]].find("parent").attrib["link"]
    flange = flange or nodes[names[-1]].find("child").attrib["link"]
    by_child = {joint.find("child").attrib["link"]: joint for joint in nodes.values()}
    route, seen, current = ([], set(), flange)
    while current != base:
        if current in seen or current not in by_child:
            raise ValueError("URDF flange is not descended from the selected base")
        seen.add(current)
        joint = by_child[current]
        route.append(joint)
        current = joint.find("parent").attrib["link"]
    route.reverse()
    moving = [joint.attrib["name"] for joint in route if joint.attrib["type"] != "fixed"]
    if moving != list(names):
        raise ValueError("URDF chain contains unselected joints or a different joint order")
    joints, chain = ([], [])
    for joint in route:
        link = {"origin": origin(joint)}
        kind = joint.attrib["type"]
        if kind != "fixed":
            if (
                kind not in {"revolute", "continuous", "prismatic"}
                or joint.find("mimic") is not None
            ):
                raise ValueError("JointDevice requires independent scalar URDF joints")
            limit = joint.find("limit")
            if limit is None and kind != "continuous":
                raise ValueError("Bounded URDF joint is missing limits")
            bounds = (
                None
                if kind == "continuous"
                else {"min": float(limit.attrib["lower"]), "max": float(limit.attrib["upper"])}
            )
            if bounds and (
                not all(math.isfinite(v) for v in bounds.values())
                or not bounds["min"] < bounds["max"]
            ):
                raise ValueError("URDF joint limits must have positive width")
            axis_node = joint.find("axis")
            axis = np.asarray(
                [
                    float(v)
                    for v in (
                        axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0"
                    ).split()
                ]
            )
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                raise ValueError("URDF joint axis must be a finite nonzero vector")
            link.update(
                joint=len(joints),
                axis=(axis / np.linalg.norm(axis)).tolist(),
                anchor=[0.0, 0.0, 0.0],
                reference=0.0,
            )
            joints.append(
                {
                    "name": joint.attrib["name"],
                    "type": kind,
                    "limits": bounds,
                    "max_velocity": float(limit.attrib["velocity"])
                    if limit is not None and "velocity" in limit.attrib
                    else None,
                    "max_effort": float(limit.attrib["effort"])
                    if limit is not None and "effort" in limit.attrib
                    else None,
                }
            )
        chain.append(link)
    return ({"format": "serial_chain", "joints": joints, "chain": chain}, base, flange)


def joint_intrinsics(model, home, name):
    if len(home) != len(model["joints"]):
        raise ValueError("Home configuration differs from the robot joint order")
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "home": {
            joint["name"]: float(value) for joint, value in zip(model["joints"], home, strict=True)
        },
        "control": {
            "speed": None,
            "mode": "position",
            "execution": {"mode": "realtime", "command_timeout_s": None},
        },
        "feedback": {"velocities": True, "efforts": False, "clock": "simulation"},
        "stop": {"behavior": "hold", "affected_devices": [name]},
        "model": model,
        "kinematics": {
            "frame": "flange",
            "fk": "numerical",
            "ik": {
                "backend": "numerical",
                "method": "damped_least_squares",
                "position_tolerance_m": 0.0001,
                "rotation_tolerance_rad": 0.001,
                "max_iterations": 200,
                "damping": 0.01,
            },
            "feedback": "driver",
            "frame_to_reported": list(IDENTITY),
        },
    }


def mount(parent, pose, frame=None):
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "mount": {
            "parent": parent,
            "parent_frame": frame or parent,
            "translation_m": list(pose[:3]),
            "rotation_xyzw": list(pose[3:]),
        },
    }


def gripper_intrinsics(names, name, *, units="m", bounds=(0, 1), open_target=1, close_target=0):
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "aperture_m": None,
        "control": {
            "mode": "position",
            "units": "normalized",
            "resolution": None,
            "bounds": {"min": bounds[0], "max": bounds[1]},
            "open_target": open_target,
            "close_target": close_target,
            "speed": None,
        },
        "feedback": {
            "positions": [{"name": item, "units": units} for item in names],
            "velocities": True,
            "sensors": [],
            "clock": "simulation",
            "aperture_model": None,
        },
        "stop": {"behavior": "hold", "affected_devices": [name]},
    }
