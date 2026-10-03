"""Fixed TCP and aperture conversion from source-calibrated finger surfaces."""

from roboenv.devices.model import local_frame

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .urdf_geometry import fixed_transform, gripper_intrinsics, matrix, matrix_pose, mount, origin


class ParallelGripper:
    def __init__(self, urdf, names, flange, model):
        calibration = json.loads(Path(__file__).with_name("gripper_contacts.json").read_text())[
            model
        ]
        nodes = {joint.get("name"): joint for joint in ET.parse(urdf).getroot().findall("joint")}
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("Parallel gripper requires two distinct measured fingers")
        joints = [nodes[name] for name in names]
        parents = [joint.find("parent").get("link") for joint in joints]
        if parents[0] != parents[1]:
            raise ValueError("Parallel fingers must share a mounting body")
        self.mount = fixed_transform(urdf, flange, parents[0])
        if any(joint.get("type") != "prismatic" for joint in joints):
            raise ValueError("Parallel gripper geometry requires sliding joints")
        transforms = [
            fixed_transform(urdf, flange, joint.find("parent").get("link")) @ matrix(origin(joint))
            for joint in joints
        ]
        axes = np.array(
            [
                transform[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                for transform, joint in zip(transforms, joints, strict=True)
            ]
        )
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        if not np.allclose(axes[0], -axes[1], atol=1e-5, rtol=0):
            raise ValueError("Parallel fingers must travel along opposite axes")
        approach = np.asarray(calibration["approach_axis"], dtype=float)
        points, normals = [], []
        for transform, contact in zip(transforms, calibration["contacts"], strict=True):
            normal = transform[:3, :3] @ contact["normal"]
            normals.append(normal / np.linalg.norm(normal))
            points.append(transform[:3, :3] @ contact["point_m"] + transform[:3, 3])
        closing = axes[0] * np.sign(np.dot(axes[0], normals[0]))
        if np.dot(closing, normals[1]) > -0.98:
            raise ValueError("Contact normals must face into the opposing jaws")
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = (points[0] + points[1]) / 2
        self.coefficients = np.array([-np.dot(axes[0], closing), np.dot(axes[1], closing)])
        self.offset = float(np.dot(points[1] - points[0], closing))
        bounds = np.array(
            [
                [float(joint.find("limit").get(key)) for key in ("lower", "upper")]
                for joint in joints
            ]
        )
        extent = self.coefficients[:, None] * bounds
        self.intrinsics = {
            "frames": {"tcp": local_frame(matrix_pose(np.linalg.inv(self.mount) @ self.tcp))},
            "aperture_m": {
                "min": max(0.0, float(self.offset + extent.min(axis=1).sum())),
                "max": max(0.0, float(self.offset + extent.max(axis=1).sum())),
            },
            "geometry": {
                "description": (
                    "Two opposing fingers slide along a fixed closing axis. The finger contact "
                    "surfaces are mesh-defined, with no calibrated rectangular pad dimensions. "
                    "Aperture is the separation of calibrated surface reference points along "
                    "the closing axis, not whole-finger clearance."
                ),
                "approach_axis": (self.mount[:3, :3].T @ approach).tolist(),
                "closing_axis": (self.mount[:3, :3].T @ closing).tolist(),
            },
        }

    def device(self, name, parent):
        intrinsics = gripper_intrinsics(name)
        intrinsics.update(self.intrinsics)
        return {
            "type": "gripper",
            "intrinsics": intrinsics,
            "extrinsics": mount(parent, matrix_pose(self.mount), "flange"),
        }

    def read(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (2,) or not np.isfinite(positions).all():
            raise ValueError("Aperture requires two finite measured finger positions")
        return {"aperture_m": max(0.0, float(self.offset + np.dot(self.coefficients, positions)))}
