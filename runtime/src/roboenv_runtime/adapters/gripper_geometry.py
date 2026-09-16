"""Source-calibrated contact frames for two independently measured sliding fingers."""

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
        self.names = list(names)
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
        self.axes = axes
        self.pads = {}
        approach = np.asarray(calibration["approach_axis"], dtype=float)
        for index, (transform, contact) in enumerate(
            zip(transforms, calibration["contacts"], strict=True)
        ):
            normal = transform[:3, :3] @ contact["normal"]
            normal /= np.linalg.norm(normal)
            tangent = approach - normal * np.dot(approach, normal)
            tangent /= np.linalg.norm(tangent)
            pad = np.eye(4)
            pad[:3, :3] = np.column_stack((np.cross(tangent, normal), tangent, normal))
            pad[:3, 3] = transform[:3, :3] @ contact["point_m"] + transform[:3, 3]
            self.pads[f"pad_{index + 1}"] = pad
        first, second = self.pads.values()
        closing = axes[0] * np.sign(np.dot(axes[0], first[:3, 2]))
        if np.dot(closing, second[:3, 2]) > -0.98:
            raise ValueError("Contact normals must face into the opposing jaws")
        # Keep flange axes; move the device origin to the midpoint of the two
        # source contact centers at q=0. Asymmetric motion remains visible in pads.
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = (first[:3, 3] + second[:3, 3]) / 2
        self.coefficients = np.array([-np.dot(axes[0], closing), np.dot(axes[1], closing)])
        self.offset = float(np.dot(second[:3, 3] - first[:3, 3], closing))
        bounds = np.array(
            [
                [float(joint.find("limit").get(key)) for key in ("lower", "upper")]
                for joint in joints
            ]
        )
        extent = self.coefficients[:, None] * bounds
        self.aperture_model = {
            "model": "affine",
            "coefficients": self.coefficients.tolist(),
            "offset_m": self.offset,
        }
        self.intrinsics = {
            "frames": dict.fromkeys(self.pads),
            "aperture_m": {
                "min": max(0.0, float(self.offset + extent.min(axis=1).sum())),
                "max": max(0.0, float(self.offset + extent.max(axis=1).sum())),
            },
            "geometry": {
                "description": (
                    "TCP is the midpoint of the source contact centers at zero finger displacement, "
                    "with axes inherited from the arm flange. Parallel sliding fingers; pad_1 and "
                    "pad_2 follow feedback position channels 1 and 2. Contact +Z points into the jaws. "
                    "Aperture is the projected separation of these contact centers, clamped at zero; "
                    "it is not collision clearance. Mesh patches have no assumed rectangular size."
                ),
                "approach_axis": approach.tolist(),
                "closing_axis": closing.tolist(),
                "contacts": {
                    name: {"frame": name, "size_m": contact["size_m"]}
                    for name, contact in zip(self.pads, calibration["contacts"], strict=True)
                },
            },
        }

    def device(self, name, parent):
        intrinsics = gripper_intrinsics(self.names, name)
        intrinsics.update(self.intrinsics)
        intrinsics["feedback"]["aperture_model"] = self.aperture_model
        return {
            "type": "gripper",
            "intrinsics": intrinsics,
            "extrinsics": mount(parent, matrix_pose(self.tcp), "flange"),
        }

    def read(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (2,) or not np.isfinite(positions).all():
            raise ValueError("Contact frames require two finite measured finger positions")
        frames = {}
        for index, (name, pad) in enumerate(self.pads.items()):
            value = pad.copy()
            value[:3, 3] += self.axes[index] * positions[index] - self.tcp[:3, 3]
            frames[name] = matrix_pose(value)
        return {
            "frame_poses": frames,
            "aperture_m": max(0.0, float(self.offset + np.dot(self.coefficients, positions))),
        }
