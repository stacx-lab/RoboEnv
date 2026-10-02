"""Validated persistent targets and per-device writes for native backends.

The existing WorldSession owns serialization, scheduling and terminal freeze.
This helper neither creates a world nor runs a thread. Backends supply native
motor application and one fixed control interval.
"""

import copy
import math

from roboenv.devices.model import DeviceError


class DeviceControl:
    def __init__(
        self, limits, targets, apply, advance, *, exhausted, on_write=None, interval_s=None
    ):
        # The world advances references using this native cadence.
        if interval_s is None or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("a controlled world must declare its control interval")
        self.interval_s = interval_s
        if set(limits) != set(targets) or not limits:
            raise ValueError("Every actuator needs limits and an initial target")
        self.limits = copy.deepcopy(limits)
        self.targets = copy.deepcopy(targets)
        self.apply, self.advance = apply, advance
        self.on_write = on_write
        self.exhausted = exhausted
        self.frame = 0
        self.commands = []
        self.on_step = None
        self.frozen = False

    def _target(self, device, target):
        if device not in self.limits:
            raise ValueError("Unknown native actuator")
        bounds = self.limits[device]
        if not isinstance(target, list) or len(target) != len(bounds):
            raise ValueError("Target dimensions differ from the native actuator")
        for value in target:
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Targets must contain finite numbers")
        return [float(value) for value in target]

    def write(self, device, target):
        """Stage validated native references without advancing physics."""
        value = self._target(device, target)
        if self.frozen or self.exhausted():
            raise DeviceError("episode_finished", "Native episode budget exhausted",
                              details={"effects": "none"})
        commands = {device: value}
        if self.on_write is not None:
            self.on_write(tuple(commands))
        self.targets.update(copy.deepcopy(commands))
        self.commands.append({"frame": self.frame, "op": "write", "commands": commands})
        return {"accepted": True, "elapsed_s": 0.0}

    def stop(self, device, measured_target):
        if self.frozen:
            raise RuntimeError("Native episode is frozen")
        # Stop may hold measured soft-limit overshoot. Never project reported state.
        bounds = self.limits.get(device)
        if bounds is None or len(measured_target) != len(bounds):
            raise ValueError("Unknown stop actuator or wrong measured dimensions")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in measured_target):
            raise ValueError("Stop requires finite measured positions")
        self.targets[device] = list(measured_target)
        # Apply the hold immediately, without advancing physics or changing peers.
        self.apply(copy.deepcopy(self.targets))
        self.commands.append({"frame": self.frame, "op": "stop", "device": device})
        return {"stopped": True, "behavior": "hold", "affected_devices": [device]}

    def tick(self):
        if self.frozen or self.exhausted():
            raise DeviceError("episode_finished", "Native episode budget exhausted",
                              details={"effects": "none"})
        self.apply(copy.deepcopy(self.targets))
        self.advance()
        self.frame += 1
        if self.on_step is not None:
            self.on_step()
