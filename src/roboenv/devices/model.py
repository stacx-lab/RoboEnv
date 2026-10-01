"""Simulator-independent device definitions and per-action execution context.

No clients, catalogs, simulators, filesystem drivers, or agent frameworks live
here. Bindings supply implementations; the environment supplies serialization.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from roboenv.devices.contract import DEVICE_CONTRACT, READ_ACTIONS
from roboenv.devices.execution import map_result
from roboenv.validation import json_value


class DeviceError(RuntimeError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.details = {"effects": "none"} if details is None else details


@dataclass(frozen=True)
class DeviceDefinition:
    name: str
    type: str
    intrinsics: dict = field(default_factory=dict)
    extrinsics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name
        ):
            raise ValueError("device name must be one nonempty logical name")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("device type must be a nonempty string")
        for name in ("intrinsics", "extrinsics"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"device {name} must be an object")
            json_value(value)
            object.__setattr__(self, name, copy.deepcopy(value))


class ExecutionContext:
    """One request's effect guard and trace. Bindings must route effects through it."""

    def __init__(self, *, read_only):
        self.read_only = read_only
        self.calls = []

    def perform(self, op, args, operation, *, mutating=False):
        if self.read_only and mutating:
            raise DeviceError("permission_denied", "Read action attempted a mutating operation")
        entry = {"op": op, "args": copy.deepcopy(args)}
        self.calls.append(entry)
        try:
            result = operation()

            def success(value):
                entry["ok"] = True
                return value

            def failure(error):
                entry.update(ok=False, error=str(error))
                raise error

            return map_result(result, success, failure)
        except Exception as exc:
            entry.update(ok=False, error=str(exc))
            raise


@dataclass(frozen=True)
class Action:
    handler: Callable[[dict, ExecutionContext], Any] | None = None
    input_schema: dict = field(default_factory=lambda: {"type": "object"})
    reason: str | None = None
    output_schema: dict = field(default_factory=dict)
    validate: Callable[[dict], None] | None = None

    def __post_init__(self):
        if self.handler is not None and not callable(self.handler):
            raise ValueError("action handler must be callable")
        if self.handler is None and not self.reason:
            raise ValueError("unsupported actions require a reason")


class Device(Protocol):
    def describe(self) -> dict: ...

    def execute(self, action: str, args: dict, context: ExecutionContext) -> Any: ...

    def close(self) -> None: ...


class BoundDevice:
    """A reusable definition paired with a binding's supported actions."""

    def __init__(
        self,
        definition: DeviceDefinition,
        actions: Mapping[str, Action],
        *,
        implementation: str,
        validation="implementation-present; not hardware-probed",
        binding=None,
        on_close=None,
    ):
        self.definition = copy.deepcopy(definition)
        self.actions = dict(actions)
        if (not self.actions and definition.type != "world") or not all(
            isinstance(a, Action) for a in self.actions.values()
        ):
            raise ValueError("device needs an explicit action table")
        if not all(isinstance(name, str) and name for name in self.actions):
            raise ValueError("action names must be nonempty strings")
        self.implementation, self.validation = implementation, validation
        self.binding, self._on_close = copy.deepcopy(binding), on_close
        self._closed = False

    def validate(self, action, args):
        check = self.actions[action].validate
        if check is not None:
            check(args)

    def describe(self):
        definition = self.definition
        result = {
            "schema_version": DEVICE_CONTRACT,
            "name": definition.name,
            "type": definition.type,
            "intrinsics": copy.deepcopy(definition.intrinsics),
            "extrinsics": copy.deepcopy(definition.extrinsics),
            "actions": {
                name: {
                    "supported": action.handler is not None,
                    "read_only": name in READ_ACTIONS,
                    "input_schema": copy.deepcopy(action.input_schema),
                    **(
                        {"output_schema": copy.deepcopy(action.output_schema)}
                        if action.output_schema
                        else {}
                    ),
                    **({"reason": action.reason} if action.handler is None else {}),
                }
                for name, action in self.actions.items()
            },
            "execution": {
                "mode": "synchronous",
                "interruptible": False,
            },
            "implementation": self.implementation,
            "validation": self.validation,
        }
        if self.binding is not None:
            result["binding"] = copy.deepcopy(self.binding)
        return result

    def execute(self, action, args, context):
        if self._closed:
            raise DeviceError("device_closed", "Device binding has been closed")
        spec = self.actions.get(action)
        if spec is None or spec.handler is None:
            raise DeviceError("unsupported_action", spec.reason if spec else "Action not declared")
        result = spec.handler(args, context)
        if action == "read" and self.definition.type in {"rgb", "rgbd"}:
            # Bind each image to live calibration, including parent-relative mounts.
            # Drivers compare this with their copied profile before returning an image.
            calibration = {
                "intrinsics": self.definition.intrinsics,
                "extrinsics": self.definition.extrinsics,
            }
            return map_result(
                result, lambda value: {**value, "calibration": copy.deepcopy(calibration)}
            )
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if self._on_close is not None:
                self._on_close()
