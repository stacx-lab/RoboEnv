"""Backend-independent devices/1 service. World ownership is injected, not imported."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future

from roboenv.devices.contract import DEVICE_CONTRACT, READ_ACTIONS
from roboenv.devices.execution import map_result, scheduling
from roboenv.devices.model import Device, DeviceError, ExecutionContext
from roboenv.validation import action_validator, json_value


class DeviceService:
    def __init__(
        self,
        devices: Iterable[Device],
        *,
        manifest=None,
        execution_lock=None,
        segment_scheduler=None,
        resolve=None,
    ):
        self._devices = {}
        self._descriptors = {}
        self._validators = {}
        self._output_validators = {}
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
                self._output_validators[name, action] = action_validator(
                    spec.get("output_schema", {})
                )
            if segment_scheduler is not None and descriptor["actions"].get("stop", {}).get(
                "supported"
            ):
                descriptor["execution"]["interruptible"] = True
            self._devices[name], self._descriptors[name] = device, descriptor
        if not self._devices:
            raise ValueError("device server needs a nonempty registry")
        self.manifest = copy.deepcopy(manifest or {})
        self._lock = execution_lock if execution_lock is not None else threading.RLock()
        self._schedule = segment_scheduler
        self._resolve = resolve or (
            lambda value: value.result() if isinstance(value, Future) else value
        )
        self._queues = {}
        self._active = {}
        self._draining = set()
        self._faults = {}
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
                "world_serialized": False,
                "stop_supported": any(
                    d["actions"].get("stop", {}).get("supported", False)
                    for d in self._descriptors.values()
                ),
                "artifact_transport": "shared-filesystem",
            },
        }

    def call(self, args, *, submitted=False):
        return self._resolve(self.submit(args, submitted=submitted))

    def _resource(self, name):
        descriptor = self.describe(name)
        binding = descriptor.get("binding") or {}
        return binding.get("entity", name)

    def interrupt(self, names=None, *, active=True):
        """Discard queued writes, optionally interrupting the current segments too."""
        with self._lock:
            resources = (
                set(self._queues) | set(self._active)
                if names is None
                else {self._resource(name) for name in names}
            )
            for resource in resources:
                for future, _, _ in self._queues.pop(resource, ()):
                    future.set_exception(
                        DeviceError("operation_interrupted", "queued write interrupted by stop")
                    )
                current = self._active.get(resource)
                if active and current is not None:
                    current.set()

    def submit(self, args, *, submitted=False):
        with self._lock:
            if self._closed:
                raise DeviceError("device_closed", "Device registry has been closed")
            # Validate shape and capability before queueing or interrupting peers.
            if (
                not isinstance(args, dict)
                or set(args) != {"device", "action", "args"}
                or not isinstance(args["args"], dict)
            ):
                raise DeviceError(
                    "invalid_args", "Expected exactly device, action, and object args"
                )
            name, action = args["device"], args["action"]
            descriptor = self.describe(name)
            spec = descriptor["actions"].get(action) if isinstance(action, str) else None
            if spec is None or not spec.get("supported"):
                raise DeviceError(
                    "unsupported_action", (spec or {}).get("reason", "Action not declared")
                )
            try:
                json_value(args["args"])
                self._validators[name, action](args["args"])
            except ValueError as error:
                raise DeviceError("invalid_args", str(error)) from error
            if submitted and action not in READ_ACTIONS:
                raise DeviceError(
                    "already_submitted", "Mutating device calls are frozen after submit"
                )
            resource = self._resource(name)
            if action == "stop":
                names = (
                    descriptor.get("intrinsics", {}).get("stop", {}).get("affected_devices", [name])
                )
                self.interrupt(names)
                try:
                    return self._execute(args, submitted=submitted)
                except Exception as error:
                    for peer in names:
                        self._faults[self._resource(peer)] = error
                    raise
            if action in READ_ACTIONS:
                return self._execute(args, submitted=submitted)
            if resource in self._faults:
                raise DeviceError("device_faulted", "Stop failed; owner recovery is required")
            future = Future()
            self._queues.setdefault(resource, deque()).append(
                (future, copy.deepcopy(args), submitted)
            )
            self._start_next(resource)
            return future

    def _start_next(self, resource):
        if resource in self._draining:
            return
        self._draining.add(resource)
        try:
            # Immediate results can drain a backlog without recursive callbacks.
            while resource not in self._active and self._queues.get(resource):
                self._start(resource)
        finally:
            self._draining.remove(resource)

    def _start(self, resource):
        future, args, submitted = self._queues[resource].popleft()
        cancel = threading.Event()
        self._active[resource] = cancel

        def finish(value=None, error=None):
            with self._lock:
                self._active.pop(resource, None)
                if cancel.is_set():
                    error = DeviceError("operation_interrupted", "device write interrupted by stop")
                if error is None:
                    future.set_result(value)
                else:
                    future.set_exception(error)
                self._start_next(resource)

        try:
            schedule = (
                (lambda complete, snapshot: self._schedule(complete, snapshot, cancel))
                if self._schedule
                else None
            )
            with scheduling(schedule):
                result = self._execute(args, submitted=submitted)
            if isinstance(result, Future):

                def completed(source):
                    try:
                        value = source.result()
                    except Exception as error:
                        finish(error=error)
                    else:
                        finish(value)

                result.add_done_callback(completed)
            else:
                finish(result)
        except Exception as error:
            finish(error=error)

    def _execute(self, args, *, submitted=False):
        with self._lock:
            if self._closed:
                raise DeviceError("device_closed", "Device registry has been closed")
            name, action = args["device"], args["action"]
            read_only = action in READ_ACTIONS
            if submitted and not read_only:
                raise DeviceError(
                    "already_submitted", "Mutating device calls are frozen after submit"
                )
            context = ExecutionContext(read_only=read_only)
            self.sequence += 1
            sequence = self.sequence
            started = time.time_ns()
            try:
                data = self._devices[name].execute(action, copy.deepcopy(args["args"]), context)
            except Exception as exc:
                raise self._failure(exc, context, read_only) from exc

            def finish(value):
                try:
                    self._output_validators[name, action](value)
                    if any(entry.get("ok") is not True for entry in context.calls):
                        raise DeviceError(
                            "device_failed", "Binding suppressed an underlying failure"
                        )
                    if isinstance(value, dict) and value.get("ok") is False:
                        raise DeviceError(
                            "device_failed", str(value.get("error", "Device rejected action"))
                        )
                except Exception as exc:
                    raise self._failure(exc, context, read_only) from exc
                return {
                    "schema_version": DEVICE_CONTRACT,
                    "device": name,
                    "action": action,
                    "sequence": sequence,
                    "started_unix_ns": started,
                    "finished_unix_ns": time.time_ns(),
                    "read_only": read_only,
                    "data": value,
                    "calls": context.calls,
                }

            def failed(error):
                raise self._failure(error, context, read_only) from error

            return map_result(data, finish, failed)

    @staticmethod
    def _failure(error, context, read_only):
        return DeviceError(
            getattr(error, "code", "device_failed"),
            str(error),
            details={"calls": context.calls, "effects": "read_only" if read_only else "unknown"},
        )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.interrupt()
            failures = []
            for device in reversed(list(self._devices.values())):
                try:
                    device.close()
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError("Device cleanup failed: " + "; ".join(failures))
