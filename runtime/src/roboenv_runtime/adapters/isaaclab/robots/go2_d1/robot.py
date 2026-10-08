"""Go2–D1 URDF geometry and mass model."""

import xml.etree.ElementTree as ET

import numpy as np

from .controllers import rotation, rpy_matrix, solve_pose_ik, transform


class RobotModel:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.joints = {}
        for elem in self.root.findall("joint"):
            origin = elem.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, :3] = rpy_matrix(*map(float, origin.get("rpy", "0 0 0").split()))
                t[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
            axis = elem.find("axis")
            limit = elem.find("limit")
            self.joints[elem.get("name")] = {
                "parent": elem.find("parent").get("link"),
                "child": elem.find("child").get("link"),
                "type": elem.get("type"),
                "origin": t,
                "axis": np.array(list(map(float, axis.get("xyz").split())))
                if axis is not None
                else np.array([1.0, 0, 0]),
                "limits": None
                if limit is None
                else [float(limit.get("lower")), float(limit.get("upper"))],
            }
        self.arm_names = [f"d1_Joint{i}" for i in range(1, 7)]

    def forward(self, positions):
        frames = {"base": np.eye(4)}
        pending = list(self.joints.items())
        while pending:
            before = len(pending)
            for name, j in pending[:]:
                if j["parent"] not in frames:
                    continue
                t = j["origin"].copy()
                q = positions.get(name, 0.0)
                if j["type"] in ("revolute", "continuous"):
                    t[:3, :3] = t[:3, :3] @ rotation(j["axis"], q)
                elif j["type"] == "prismatic":
                    t[:3, 3] += t[:3, :3] @ (j["axis"] * q)
                frames[j["child"]] = frames[j["parent"]] @ t
                pending.remove((name, j))
            if len(pending) == before:
                raise ValueError("URDF is not a connected tree rooted at base")
        return frames

    def center_of_mass(self, positions):
        """Assembly COM in the base frame, including fixed payload/camera masses."""
        frames = self.forward(positions)
        weighted = np.zeros(3)
        total = 0.0
        for link in self.root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            transform = frames[link.get("name")]
            weighted += mass * (transform[:3, :3] @ xyz + transform[:3, 3])
            total += mass
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Robot assembly must have a positive finite mass")
        return weighted / total

    def arm_fk(self, q):
        frames = self.forward(dict(zip(self.arm_names, q, strict=True)))
        return np.linalg.inv(frames["d1_base_link"]) @ frames["d1_Link6"]

    def arm_ik(self, target, seed, iterations=150):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("IK pose must be finite [x,y,z,qx,qy,qz,qw]")
        if abs(np.linalg.norm(target[3:]) - 1) > 1e-3:
            raise ValueError("IK quaternion must be unit length")
        bounds = np.array([self.joints[n]["limits"] for n in self.arm_names])
        q = np.asarray(seed, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("IK seed must contain six finite joint angles")
        desired = transform({"translation_m": target[:3], "rotation_xyzw": target[3:]})
        return solve_pose_ik(
            desired, q, bounds, self.arm_fk, iterations=iterations,
            failure="Arm IK did not converge within joint limits; no command applied",
        ).tolist()
