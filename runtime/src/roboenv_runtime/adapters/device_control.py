"""Validated persistent targets and atomic control intervals for native backends.

The existing WorldSession owns serialization, scheduling and terminal freeze.
This helper neither creates a world nor runs a thread. Backends supply native
motor application and one fixed control interval.
"""

import copy
import math


class DeviceControl:
    def __init__(self, limits, targets, apply, advance, *, realtime, exhausted, on_write=None):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        if set(limits) != set(targets) or not limits:
            raise ValueError("Every actuator needs limits and an initial target")
        self.limits = copy.deepcopy(limits)
        self.targets = copy.deepcopy(targets)
        self.apply, self.advance = apply, advance
        self.on_write = on_write
        self.realtime, self.exhausted = realtime, exhausted
        self.frame = 0
        self.commands = []
        self.pending = None
        self.last_step_id = -1

    def _target(self, device, target):
        if device not in self.limits:
            raise ValueError("Unknown native actuator")
        bounds = self.limits[device]
        if not isinstance(target, list) or len(target) != len(bounds):
            raise ValueError("Target dimensions differ from the native actuator")
        for value, (lower, upper) in zip(target, bounds, strict=True):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Targets must contain finite numbers")
            if lower is not None and value < lower or upper is not None and value > upper:
                raise ValueError("Target exceeds native joint limits")
        return [float(value) for value in target]

    def write(self, device, target, step=None):
        value = self._target(device, target)
        if self.exhausted():
            raise RuntimeError("Native episode budget exhausted")
        if step is None:
            if self.pending is not None:
                raise ValueError("Complete or stop the pending atomic step first")
            return self._commit({device: value})
        if not isinstance(step, dict) or set(step) != {"id", "devices"}:
            raise ValueError("Atomic step requires exactly id and devices")
        identity, devices = step["id"], step["devices"]
        if type(identity) is not int or identity <= self.last_step_id:
            raise ValueError("Atomic step id must increase within an episode")
        if (
            not isinstance(devices, list) or len(devices) < 2
            or not all(isinstance(name, str) for name in devices)
            or len(set(devices)) != len(devices) or device not in devices
            or set(devices) - self.limits.keys()
        ):
            raise ValueError("Atomic step requires distinct declared actuators including this device")
        expected = sorted(devices)
        if self.pending is not None and (
            self.pending["id"] != identity or self.pending["devices"] != expected
        ):
            raise ValueError("Another atomic step is pending")
        pending = copy.deepcopy(self.pending or {"id": identity, "devices": expected, "values": {}})
        if device in pending["values"]:
            raise ValueError("An actuator can contribute only once to an atomic step")
        pending["values"][device] = value
        if set(pending["values"]) != set(expected):
            self.pending = pending
            return {"accepted": True, "pending": True, "step_id": identity}
        self.pending = None
        self.last_step_id = identity
        return {**self._commit(pending["values"]), "pending": False, "step_id": identity}

    def _commit(self, commands):
        if self.on_write is not None:
            self.on_write(tuple(commands))
        self.targets.update(copy.deepcopy(commands))
        self.commands.append({"frame": self.frame, "op": "write", "commands": commands})
        if not self.realtime:
            self.tick()
        return {"accepted": True}

    def stop(self, device, measured_target):
        # Stop may hold measured soft-limit overshoot. Never project reported state.
        bounds = self.limits.get(device)
        if bounds is None or len(measured_target) != len(bounds):
            raise ValueError("Unknown stop actuator or wrong measured dimensions")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in measured_target):
            raise ValueError("Stop requires finite measured positions")
        if self.pending is not None:
            self.last_step_id = self.pending["id"]
            self.pending = None
        self.targets[device] = list(measured_target)
        # Apply the hold immediately, without advancing physics or changing peers.
        self.apply(copy.deepcopy(self.targets))
        self.commands.append({"frame": self.frame, "op": "stop", "device": device})
        return {"stopped": True, "behavior": "hold", "affected_devices": [device]}

    def tick(self):
        if self.exhausted():
            raise RuntimeError("Native episode budget exhausted")
        self.apply(copy.deepcopy(self.targets))
        self.advance()
        self.frame += 1
