"""The robot config (YAML in `configs/`): model, camera, actuation, standing keyframe and
simulation settings."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from roboenv_runtime.adapters.isaaclab.robots.booster_t2 import assets

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
"""Directory of the robot and policy configs."""


def first_match(table: dict[str, Any], name: str) -> Any:
    """Value of the first regex key of `table` that fully matches `name`, else None."""
    for pattern, value in table.items():
        if re.fullmatch(pattern, name):
            return value
    return None


@dataclass(frozen=True)
class BoosterPdGains:
    """Per-joint PD gains and motor limits. Every field has shape (num_joints,)."""

    stiffness: np.ndarray
    """kp, N·m/rad."""
    damping: np.ndarray
    """kd, N·m·s/rad."""
    effort_limit: np.ndarray
    """Peak torque, N·m."""
    velocity_limit: np.ndarray
    """Speed at which the available torque reaches zero, rad/s."""
    knee_point_velocity: np.ndarray
    """Speed up to which the full `effort_limit` is available, rad/s."""


@dataclass(frozen=True)
class CameraConfig:
    """A camera on the robot."""

    name: str
    body: str
    """Link it is fixed to."""
    pos: tuple[float, float, float]
    """Optical frame origin in the link frame, m."""
    rpy: tuple[float, float, float]
    """Optical frame orientation in the link frame, URDF roll-pitch-yaw, rad. The optical frame
    is x right, y down, z forward (OpenCV)."""
    width: int
    """Image width, px."""
    height: int
    """Image height, px."""
    horizontal_fov: float
    """Horizontal field of view, rad."""
    near: float
    """Nearest rendered distance, m."""
    far: float
    """Farthest reported depth, m."""
    rate: float
    """Capture rate, Hz."""

    def intrinsics(self) -> np.ndarray:
        """Pinhole intrinsics K, pixel centers at integer coordinates. Shape (3, 3)."""
        f = self.width / (2.0 * math.tan(self.horizontal_fov / 2))
        return np.array(
            [[f, 0.0, (self.width - 1) / 2], [0.0, f, (self.height - 1) / 2], [0, 0, 1]]
        )


@dataclass(frozen=True)
class CollisionBox:
    """A box collision shape in a link frame, unrotated."""

    size: tuple[float, float, float]
    """Full extents along the link axes, m."""
    pos: tuple[float, float, float]
    """Center in the link frame, m."""


@dataclass(frozen=True)
class SimulationConfig:
    """How the robot is simulated in Isaac Lab."""

    timestep: float
    """Physics timestep, s."""
    friction: float
    """Static and dynamic friction of the robot and the ground."""
    self_collision: bool
    """Whether the robot's links collide with each other."""
    solver_iterations: tuple[int, int]
    """PhysX position and velocity solver iterations."""
    collision_boxes: dict[str, CollisionBox]
    """Link name -> the box that replaces that link's URDF collision."""


@dataclass(frozen=True)
class FingerConfig:
    """The Revo2 finger joints and their simulated drives."""

    joints: tuple[str, ...]
    """Every movable finger joint, URDF order; the distal ones follow `mimic`."""
    mimic: dict[str, tuple[str, float, float]]
    """Distal joint -> (joint it follows, multiplier, offset rad), from the URDF."""
    stiffness: float
    """Position drive stiffness, N·m/rad."""
    damping: float
    """Position drive damping, N·m·s/rad."""
    armature: float
    """Armature of each finger joint, kg·m²."""

    def positions(self, commanded: dict[str, float]) -> np.ndarray:
        """Positions of all `joints` from those of the joints the others follow; unlisted ones at
        0. Shape (len(joints),), rad."""
        q = {name: commanded.get(name, 0.0) for name in self.joints if name not in self.mimic}
        for name, (leader, multiplier, offset) in self.mimic.items():
            q[name] = multiplier * q[leader] + offset
        return np.array([q[name] for name in self.joints])


@dataclass(frozen=True)
class UrdfJoint:
    """A movable joint of the URDF."""

    name: str
    """Joint name."""
    lower: float
    """Position limit, rad."""
    upper: float
    """Position limit, rad."""
    velocity: float
    """Speed limit, rad/s."""
    effort: float
    """Torque limit, N·m."""


def _urdf_joints(urdf: Path) -> tuple[dict[str, UrdfJoint], dict[str, tuple[str, float, float]]]:
    joints, mimic = {}, {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        name, limit = joint.get("name"), joint.find("limit").attrib
        joints[name] = UrdfJoint(
            name,
            *(float(limit[k]) for k in ("lower", "upper", "velocity", "effort")),
        )
        m = joint.find("mimic")
        if m is not None:
            mimic[name] = (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
    return joints, mimic


@dataclass(frozen=True)
class Motor:
    """Effective specs of the motor driving one joint."""

    effort_limit: float
    """Peak torque, N·m."""
    velocity_limit: float
    """Speed at which the available torque reaches zero, rad/s."""
    knee_point_velocity: float
    """Speed up to which the full `effort_limit` is available, rad/s."""
    armature: float
    """Reflected rotor inertia, kg·m²."""


class RobotConfig:
    """A robot config: model, camera, actuation, standing keyframe and simulation settings.

    Args:
        path: YAML file; a bare file name is looked up in `CONFIG_DIR`.
    """

    def __init__(self, path: str | Path = "t2_revo2.yaml", *, asset_directory=None) -> None:
        path = Path(path)
        self.path = CONFIG_DIR / path if len(path.parts) == 1 else path
        self._cfg: dict[str, Any] = yaml.safe_load(self.path.read_text())
        self.urdf: Path = Path(asset_directory or assets.directory()) / "assets" / self._cfg["urdf"]
        """The robot's URDF."""
        self.root_body: str = self._cfg["root_body"]
        """Name of the floating root link."""
        self.joints, mimic = _urdf_joints(self.urdf)
        """Movable URDF joints by name, with their limits."""
        self.joint_groups: dict[str, tuple[str, ...]] = {
            name: tuple(joints) for name, joints in self._cfg["joint_groups"].items()
        }
        """Group name -> its joints, in group order: the head, the arms and the hands."""
        f = self._cfg["fingers"]
        finger_joints = tuple(n for n in self.joints if any(t in n for t in f["tokens"]))
        self.fingers = FingerConfig(
            joints=finger_joints,
            mimic={n: m for n, m in mimic.items() if n in finger_joints},
            stiffness=f["stiffness"],
            damping=f["damping"],
            armature=f["armature"],
        )
        """The finger joints and their simulated drives."""
        c = self._cfg["camera"]
        self.camera = CameraConfig(
            name=c["name"],
            body=c["body"],
            pos=tuple(c["pos"]),
            rpy=tuple(c["rpy"]),
            width=c["width"],
            height=c["height"],
            horizontal_fov=math.radians(c["horizontal_fov_deg"]),
            near=c["near"],
            far=c["far"],
            rate=c["rate_hz"],
        )
        """The head camera."""
        sim = self._cfg["simulation"]
        self.simulation = SimulationConfig(
            timestep=float(sim["timestep"]),
            friction=float(sim["friction"]),
            self_collision=bool(sim["self_collision"]),
            solver_iterations=tuple(sim["solver_iterations"]),
            collision_boxes={
                link: CollisionBox(size=tuple(box["size"]), pos=tuple(box["pos"]))
                for link, box in sim["collision_boxes"].items()
            },
        )
        """The Isaac Lab model and solver settings."""

    @property
    def timestep(self) -> float:
        """Physics timestep, s."""
        return self.simulation.timestep

    def motor(self, joint: str) -> Motor:
        """Effective motor of `joint`, linkage scaling applied.

        Raises:
            ValueError: `joint` matches no actuator entry, or more than one.
        """
        hits = [
            a for a in self._cfg["actuators"] if any(re.fullmatch(p, joint) for p in a["joints"])
        ]
        if len(hits) != 1:
            raise ValueError(f"{joint} matches {len(hits)} actuator entries")
        entry = hits[0]
        spec = self._cfg["motors"][entry["motor"]]
        scale = {"effort": 1.0, "velocity": 1.0, "armature": 1.0, **entry.get("scale", {})}
        rpm = 2.0 * math.pi / 60.0 * scale["velocity"]
        return Motor(
            effort_limit=spec["effort_limit"] * scale["effort"],
            velocity_limit=spec["velocity_limit_rpm"] * rpm,
            knee_point_velocity=spec["knee_point_rpm"] * rpm,
            armature=spec["armature"] * scale["armature"],
        )

    def pd_gains(self, joints: Sequence[str]) -> BoosterPdGains:
        """PD gains and motor limits of `joints`, in that order."""
        omega = 2.0 * math.pi * self._cfg["pd"]["natural_frequency_hz"]
        zeta = self._cfg["pd"]["damping_ratio"]
        motors = [self.motor(j) for j in joints]
        armature = np.array([m.armature for m in motors])
        return BoosterPdGains(
            stiffness=armature * omega**2,
            damping=2.0 * zeta * armature * omega,
            effort_limit=np.array([m.effort_limit for m in motors]),
            velocity_limit=np.array([m.velocity_limit for m in motors]),
            knee_point_velocity=np.array([m.knee_point_velocity for m in motors]),
        )

    def limits(self, joints: Sequence[str]) -> np.ndarray:
        """URDF position limits of `joints`, in that order: columns lower, upper. Shape (len, 2),
        rad."""
        return np.array([[self.joints[j].lower, self.joints[j].upper] for j in joints])

    def keyframe_root_pos(self) -> np.ndarray:
        """Standing start position of the root link. Shape (3,), m."""
        return np.array(self._cfg["keyframe"]["root_pos"], dtype=float)

    def keyframe_joint_pos(self, joints: Sequence[str]) -> np.ndarray:
        """Standing start position of `joints`, in that order; 0 where unlisted. Shape (len,),
        rad."""
        table = self._cfg["keyframe"]["joint_pos"]
        return np.array([first_match(table, j) or 0.0 for j in joints], dtype=float)
