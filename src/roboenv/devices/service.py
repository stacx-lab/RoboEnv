"""Backend-independent devices/1 service. World ownership is injected, not imported."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future

from roboenv.devices.contract import DEVICE_CONTRACT, READ_ACTIONS
from roboenv.devices.execution import map_result, request_identity, scheduling
from roboenv.devices.model import Device, DeviceError, ExecutionContext
from roboenv.validation import action_validator, json_value


class DeviceService:
    def __init__(
        self,
        devices: Iterable[Device],
        *,
        manifest=None,
        execution_lock=None,
        motion_scheduler=None,
        resolve=None,
        flush=None,
    ):
        self._owners = {}
        self._cancelled = {}  # Bounded tombstones prevent cancellation overtaking admission.
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
            if motion_scheduler is not None and descriptor["actions"].get("stop", {}).get(
                "supported"
            ):
                descriptor["execution"]["interruptible"] = True
            self._devices[name], self._descriptors[name] = device, descriptor
        if not self._devices:
            raise ValueError("device server needs a nonempty registry")
        self.manifest = copy.deepcopy(manifest or {})
        self._lock = execution_lock if execution_lock is not None else threading.RLock()
        self._schedule = motion_scheduler
        self._resolve = resolve or (
            lambda value: value.result() if isinstance(value, Future) else value
        )
        self._flush = flush or (lambda: None)
        self._queues = {}
        self._active = {}
        self._waits = {}
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
                "mode": self.manifest.get("execution_mode", "realtime"),
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

    def cancel(self, args, *, submitted=False):
        if (
            not isinstance(args, dict)
            or set(args) != {"execution_id"}
            or not isinstance(args["execution_id"], str)
            or not args["execution_id"]
        ):
            raise DeviceError("invalid_args", "cancel requires a nonempty execution_id")
        identity = args["execution_id"]
        with self._lock:
            waiting = self._waits.pop(identity, None)
            if waiting is not None:
                waiting.cancel()
                return {"data": {"stopped": False}}
            self._cancelled[identity] = None
            if len(self._cancelled) > 4096:
                self._cancelled.pop(next(iter(self._cancelled)))
            names = [
                name for name in self._devices if self._owners.get(self._resource(name)) == identity
            ]
            self.interrupt(names)
            self._flush()
            for name in names:
                self.submit({"device": name, "action": "stop", "args": {}})
            return {"data": {"stopped": bool(names)}}

    def _claim(self, names):
        identity = request_identity.get()
        if identity in self._cancelled:
            raise DeviceError("operation_interrupted", "motion cancelled before admission")
        for name in names:
            self._owners[self._resource(name)] = identity

    def _validate(self, name, action, args):
        descriptor = self.describe(name)
        spec = descriptor["actions"].get(action) if isinstance(action, str) else None
        if spec is None or not spec.get("supported"):
            raise DeviceError(
                "unsupported_action", (spec or {}).get("reason", "Action not declared")
            )
        try:
            json_value(args)
            self._validators[name, action](args)
            validate = getattr(self._devices[name], "validate", None)
            if validate is not None:
                validate(action, args)
        except ValueError as error:
            raise DeviceError("invalid_args", str(error)) from error
        if action not in READ_ACTIONS | {"stop"} and self._resource(name) in self._faults:
            raise DeviceError("device_faulted", "Stop failed; owner recovery is required")
        return descriptor

    def _enqueue(self, args, submitted=False):
        resource = self._resource(args["device"])
        future = Future()
        self._queues.setdefault(resource, deque()).append((future, copy.deepcopy(args), submitted))
        self._start_next(resource)
        return future

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
            descriptor = self._validate(name, action, args["args"])
            if submitted and action not in READ_ACTIONS:
                raise DeviceError(
                    "already_submitted", "Mutating device calls are frozen after submit"
                )
            if action == "stop":
                names = (
                    descriptor.get("intrinsics", {}).get("stop", {}).get("affected_devices", [name])
                )
                self.interrupt(names)
                for peer in names:
                    self._owners.pop(self._resource(peer), None)
                try:
                    return self._execute(args, submitted=submitted)
                except Exception as error:
                    for peer in names:
                        self._faults[self._resource(peer)] = error
                    raise
            if action == "wait":
                identity = request_identity.get()
                if identity in self._cancelled:
                    raise DeviceError("operation_interrupted", "wait cancelled before admission")
                result = self._execute(args, submitted=submitted)
                if isinstance(result, Future):
                    self._waits[identity] = result
                    result.add_done_callback(lambda source: self._waits.pop(identity, None))
                return result
            if action in READ_ACTIONS:
                return self._execute(args, submitted=submitted)
            self._claim([name])
            self.interrupt([name])
            self._flush()
            return self._enqueue(args, submitted)

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
        if resource in self._faults:
            future.set_exception(DeviceError("device_faulted", "Owner recovery is required"))
            return
        cancel = threading.Event()
        self._active[resource] = cancel
        background = []

        def release(source=None):
            with self._lock:
                if source is not None:
                    error = source.exception()
                    ended = (
                        getattr(error, "code", None) == "episode_finished"
                        and (getattr(error, "details", None) or {}).get("effects")
                        in ("none", "partial")
                    )
                    if (
                        error is not None
                        and getattr(error, "code", None) != "operation_interrupted"
                        and not ended
                    ):
                        self._faults[resource] = error
                if self._active.get(resource) is cancel:
                    self._active.pop(resource, None)
                self._start_next(resource)

        def schedule(motion):
            result = self._schedule(motion, cancel)
            background.append(result)
            return result

        try:
            with scheduling(schedule if self._schedule else None):
                result = self._execute(args, submitted=submitted)
            if isinstance(result, Future):

                def completed(source):
                    try:
                        future.set_result(source.result())
                    except Exception as error:
                        future.set_exception(error)
                    finally:
                        release()

                result.add_done_callback(completed)
            else:
                future.set_result(result)
                if background:
                    background[-1].add_done_callback(release)
                else:
                    release()
        except Exception as error:
            future.set_exception(error)
            cancel.set()
            release()

    def _execute(self, args, *, submitted=False):
        with self._lock:
            if self._closed:
                raise DeviceError("device_closed", "Device registry has been closed")
            name, action = args["device"], args["action"]
            read_only = action in READ_ACTIONS
            if submitted and not read_only:
                raise DeviceError(
                    "already_submitted", "Mutating device calls are frozen after submit",
                    details={"effects": "none"},
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
                if action == "read" and self._resource(name) in self._faults:
                    value = {
                        **value,
                        "faults": [*value.get("faults", []), {"code": "motion_failed"}],
                    }
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
                    "calls": copy.deepcopy(context.calls),
                }

            def failed(error):
                raise self._failure(error, context, read_only) from error

            return map_result(data, finish, failed)

    @staticmethod
    def _failure(error, context, read_only):
        return DeviceError(
            getattr(error, "code", "device_failed"),
            str(error),
            details={
                "calls": context.calls,
                "effects": "read_only" if read_only else "unknown",
                **(getattr(error, "details", None) or {}),
            },
        )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.interrupt()
            for waiting in list(self._waits.values()):
                waiting.cancel()
            self._flush()
            failures = []
            for name, device in self._devices.items():
                if self._resource(name) in self._owners and device.describe()["actions"].get(
                    "stop", {}
                ).get("supported"):
                    try:
                        self._execute({"device": name, "action": "stop", "args": {}})
                    except Exception as error:
                        failures.append(str(error))
            self._closed = True
            for device in reversed(list(self._devices.values())):
                try:
                    device.close()
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError("Device cleanup failed: " + "; ".join(failures))
