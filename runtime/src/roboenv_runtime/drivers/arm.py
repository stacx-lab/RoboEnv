"""Proposed common Cartesian arm driver, deliberately not a servo or planner.

One request issues at most one motion command through an injected ArmPort. The
world still owns serialization, deadlines, simulation steps, budgets and verdicts.
This opt-in interface is under design review; existing bindings are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from roboenv.device_model import Action, BoundDevice, DeviceError, ExecutionContext


def vector(values, length, label):
    try:
        items = tuple(values)
    except TypeError as error:
        raise DeviceError("invalid_args", f"{label} must be a numeric vector") from error
    if len(items) != length or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in items
    ):
        raise DeviceError("invalid_args", f"{label} needs {length} finite numbers")
    return tuple(float(value) for value in items)


def quaternion(values):
    result = vector(values, 4, "quaternion_xyzw")
    norm = math.sqrt(sum(value * value for value in result))
    if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-3):
        raise DeviceError("invalid_args", "quaternion_xyzw must have unit length")
    return tuple(value / norm for value in result)


@dataclass(frozen=True)
class Pose:
    position: tuple[float, ...]
    quaternion_xyzw: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "position", vector(self.position, 3, "position_m"))
        object.__setattr__(self, "quaternion_xyzw", quaternion(self.quaternion_xyzw))

    @classmethod
    def from_wxyz(cls, position, orientation):
        w, x, y, z = vector(orientation, 4, "native quaternion_wxyz")
        return cls(tuple(position), (x, y, z, w))

    def as_dict(self):
        return {
            "position_m": list(self.position),
            "quaternion_xyzw": list(self.quaternion_xyzw),
        }


@dataclass(frozen=True)
class ArmSemantics:
    frame: str
    control_point: str
    orientation_frame: str
    motion_policy: str

    def as_dict(self):
        return {
            "frame": self.frame,
            "control_point": self.control_point,
            "orientation_frame": self.orientation_frame,
            "motion_policy": self.motion_policy,
            "length_unit": "m",
            "quaternion_order": "xyzw",
        }


class ArmPort(Protocol):
    """An existing arm entity; ports must never create, step or close a world.

    Reads and commands must use the request's execution context. Implementations
    either call counted world controllers or explicitly record native operations.
    Merely conforming to this Python protocol is not a sandbox or GPU certificate.
    """

    semantics: ArmSemantics

    def unavailable(self, *, motion: bool) -> str | None: ...

    def read_pose(self, execution: ExecutionContext) -> Pose: ...

    def move_pose(self, pose: Pose, execution: ExecutionContext): ...


class ArmDriver:
    """Validation, orientation preservation, displacement and measured outcomes."""

    def __init__(self, port: ArmPort, *, position_tolerance_m=0.01, orientation_tolerance_rad=0.05):
        self.port = port
        for value in (position_tolerance_m, orientation_tolerance_rad):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < math.inf:
                raise ValueError("arm observation tolerances must be finite and positive")
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)

    def _frame(self, args):
        if args.get("frame") != self.port.semantics.frame:
            raise DeviceError("invalid_args", "target frame differs from this arm's declared frame")

    def read(self, args, execution):
        if args:
            raise DeviceError("invalid_args", "arm read accepts no arguments")
        return {**self.port.read_pose(execution).as_dict(), **self.port.semantics.as_dict()}

    def write(self, args, execution):
        if set(args) != {"frame", "target"}:
            raise DeviceError("invalid_args", "write requires exactly frame and target")
        self._frame(args)
        target = args["target"]
        if not isinstance(target, (list, tuple)) or len(target) not in (3, 7):
            raise DeviceError("invalid_args", "target must be xyz or xyz + quaternion xyzw")
        position = vector(target[:3], 3, "target position")
        # Validate every supplied input before even making a backend read.
        orientation = quaternion(target[3:]) if len(target) == 7 else None
        if orientation is None:
            orientation = self.port.read_pose(execution).quaternion_xyzw
        return self._move(Pose(position, orientation), execution)

    def displace(self, args, execution):
        if set(args) != {"frame", "delta"}:
            raise DeviceError("invalid_args", "displace requires exactly frame and delta")
        self._frame(args)
        delta = vector(args["delta"], 3, "delta_m")
        before = self.port.read_pose(execution)
        target = Pose(
            tuple(value + change for value, change in zip(before.position, delta, strict=True)),
            before.quaternion_xyzw,
        )
        return self._move(target, execution)

    def _move(self, target, execution):
        report = self.port.move_pose(target, execution)
        if isinstance(report, dict) and report.get("ok") is False:
            raise DeviceError("device_failed", "backend rejected the motion command")
        observed = self.port.read_pose(execution)
        distance = math.dist(target.position, observed.position)
        dot = abs(
            sum(
                a * b for a, b in zip(target.quaternion_xyzw, observed.quaternion_xyzw, strict=True)
            )
        )
        angle = 2 * math.acos(min(1.0, dot))
        return {
            "target": target.as_dict(),
            "observed": {**observed.as_dict(), **self.port.semantics.as_dict()},
            "position_error_m": distance,
            "orientation_error_rad": angle,
            "target_reached": distance <= self.position_tolerance_m
            and angle <= self.orientation_tolerance_rad,
            "backend_report": report,
        }

    def bind(self, definition, *, binding):
        numeric = {"type": "number"}
        triple = {"type": "array", "items": numeric, "minItems": 3, "maxItems": 3}
        empty = {"type": "object", "properties": {}, "additionalProperties": False}

        def command(field, value):
            return {
                "type": "object",
                "properties": {"frame": {"const": self.port.semantics.frame}, field: value},
                "required": ["frame", field],
                "additionalProperties": False,
            }

        target = {"oneOf": [triple, {**triple, "minItems": 7, "maxItems": 7}]}
        actions = {}
        for name, handler, schema in (
            ("read", self.read, empty),
            ("state", self.read, empty),
            ("write", self.write, command("target", target)),
            ("displace", self.displace, command("delta", triple)),
        ):
            reason = self.port.unavailable(motion=name not in {"read", "state"})
            actions[name] = Action(reason=reason) if reason else Action(handler, schema)
        for name in ("stop", "ik", "fk", "move_joints", "pick", "place"):
            actions[name] = Action(reason="Not implemented by this Cartesian driver proposal")
        return BoundDevice(
            definition,
            actions,
            implementation="shared-cartesian-arm-proposal",
            validation="opt-in design prototype; not GPU-certified",
            binding={**binding, "semantics": self.port.semantics.as_dict()},
        )
