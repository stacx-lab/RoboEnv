"""Backend-independent devices/1 service. World ownership is injected, not imported."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterable

from .contract import DEVICE_CONTRACT, READ_ACTIONS
from .device_model import Device, DeviceError, ExecutionContext
from .validation import action_validator, json_value


class DeviceService:
    def __init__(self, devices: Iterable[Device], *, manifest=None, execution_lock=None):
        self._devices = {}
        self._descriptors = {}
        self._validators = {}
        for device in devices:
            descriptor = copy.deepcopy(device.describe())
            name = descriptor["name"]
            if name in self._devices:
                raise ValueError(f"duplicate logical device: {name}")
            if descriptor.get("schema_version") != DEVICE_CONTRACT:
                raise ValueError("unsupported device descriptor contract")
            for action, spec in descriptor["actions"].items():
                if spec.get("read_only") is not (action in READ_ACTIONS):
                    raise ValueError("read-only action semantics must match devices/1")
                self._validators[name, action] = action_validator(spec.get("input_schema", {}))
            self._devices[name], self._descriptors[name] = device, descriptor
        if not self._devices:
            raise ValueError("device server needs a nonempty registry")
        self.manifest = copy.deepcopy(manifest or {})
        self._lock = execution_lock if execution_lock is not None else threading.RLock()
        self.sequence = 0
        self._closed = False

    def describe(self, name):
        if not isinstance(name, str) or name not in self._descriptors:
            raise DeviceError("unknown_device", f"unknown device: {name!r}")
        return copy.deepcopy(self._descriptors[name])

    def inventory(self):
        return {
            "schema_version": DEVICE_CONTRACT,
            "devices": [self.describe(name) for name in sorted(self._devices)],
        }

    def environment(self):
        return {
            **copy.deepcopy(self.manifest),
            "device_contract": DEVICE_CONTRACT,
            "devices": sorted(self._devices),
            "execution": {
                "mode": "synchronous",
                "world_serialized": True,
                "stop_supported": any(
                    d["actions"].get("stop", {}).get("supported", False)
                    for d in self._descriptors.values()
                ),
                "artifact_transport": "shared-filesystem",
            },
        }

    def call(self, args, *, submitted=False):
        with self._lock:
            if self._closed:
                raise DeviceError("device_closed", "Device registry has been closed")
            if (
                not isinstance(args, dict)
                or set(args) != {"device", "action", "args"}
                or not isinstance(args["args"], dict)
            ):
                raise DeviceError(
                    "invalid_args", "Expected exactly device, action, and object args"
                )
            name, action = args["device"], args["action"]
            try:
                json_value(args["args"])
            except ValueError as exc:
                raise DeviceError("invalid_args", str(exc)) from exc
            descriptor = self.describe(name)
            spec = descriptor["actions"].get(action) if isinstance(action, str) else None
            if spec is None or spec.get("supported") is not True:
                raise DeviceError(
                    "unsupported_action", (spec or {}).get("reason", "Action not declared")
                )
            read_only = action in READ_ACTIONS
            if submitted and not read_only:
                raise DeviceError(
                    "already_submitted", "Mutating device calls are frozen after submit"
                )
            try:
                self._validators[name, action](args["args"])
            except ValueError as exc:
                raise DeviceError("invalid_args", str(exc)) from exc
            context = ExecutionContext(read_only=read_only)
            self.sequence += 1
            started = time.time_ns()
            try:
                data = self._devices[name].execute(action, copy.deepcopy(args["args"]), context)
                if any(entry.get("ok") is not True for entry in context.calls):
                    raise DeviceError("device_failed", "Binding suppressed an underlying failure")
                if isinstance(data, dict) and data.get("ok") is False:
                    raise DeviceError(
                        "device_failed", str(data.get("error", "Device rejected action"))
                    )
            except Exception as exc:
                # A native binding is trusted implementation code. No trace does
                # NOT prove it had no effects; preserve fail-closed verification.
                raise DeviceError(
                    getattr(exc, "code", "device_failed"),
                    str(exc),
                    details={
                        "calls": context.calls,
                        "effects": "read_only" if read_only else "unknown",
                    },
                ) from exc
            return {
                "schema_version": DEVICE_CONTRACT,
                "device": name,
                "action": action,
                "sequence": self.sequence,
                "started_unix_ns": started,
                "finished_unix_ns": time.time_ns(),
                "read_only": read_only,
                "data": data,
                "calls": context.calls,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failures = []
            for device in reversed(list(self._devices.values())):
                try:
                    device.close()
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError("Device cleanup failed: " + "; ".join(failures))
