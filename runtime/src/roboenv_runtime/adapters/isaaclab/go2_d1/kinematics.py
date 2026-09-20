"""Task-free URDF forward/inverse kinematics in metres and radians."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def rpy_matrix(roll, pitch, yaw):
    return rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)


def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat(r):
    # Largest diagonal branch is stable even at rotations of pi.
    candidates = np.array(
        [
            1 + r[0, 0] - r[1, 1] - r[2, 2],
            1 - r[0, 0] + r[1, 1] - r[2, 2],
            1 - r[0, 0] - r[1, 1] + r[2, 2],
            1 + np.trace(r),
        ]
    )
    i = int(np.argmax(candidates))
    q = np.zeros(4)
    q[i] = math.sqrt(max(candidates[i], 0)) / 2
    d = 4 * q[i]
    if i == 3:
        q[:3] = [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]
        q[:3] /= d
    else:
        j, k = (i + 1) % 3, (i + 2) % 3
        q[j], q[k], q[3] = (r[j, i] + r[i, j]) / d, (r[k, i] + r[i, k]) / d, (r[k, j] - r[j, k]) / d
    return q / np.linalg.norm(q)


def pose(t):
    return {"translation_m": t[:3, 3].tolist(), "rotation_xyzw": matrix_quat(t[:3, :3]).tolist()}


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
        q = np.clip(q, bounds[:, 0], bounds[:, 1])
        desired_r = quat_matrix(target[3:])

        def error(q):
            t = self.arm_fk(q)
            # Rotation-vector residual remains informative at pi (cross-product does not).
            r = desired_r @ t[:3, :3].T
            quat = matrix_quat(r)
            if quat[3] < 0:
                quat = -quat
            norm = np.linalg.norm(quat[:3])
            rot = quat[:3] * (2 * math.atan2(norm, quat[3]) / max(norm, 1e-12))
            return np.r_[target[:3] - t[:3, 3], rot]

        for _ in range(iterations):
            e = error(q)
            if np.linalg.norm(e[:3]) < 0.001 and np.linalg.norm(e[3:]) < 0.01:
                return q.tolist()
            jac = np.column_stack([(error(q + np.eye(6)[i] * 1e-5) - e) / 1e-5 for i in range(6)])
            step = -jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-4, e)
            q = np.clip(q + np.clip(step, -0.15, 0.15), bounds[:, 0], bounds[:, 1])
        raise ValueError("Arm IK did not converge within joint limits; no command applied")
