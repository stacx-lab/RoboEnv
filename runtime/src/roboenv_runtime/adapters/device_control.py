"""Validated persistent targets and per-device writes for native backends.

The existing WorldSession owns serialization, scheduling and terminal freeze.
This helper neither creates a world nor runs a thread. Backends supply native
motor application and one fixed control interval.
"""

import copy
import math
import time


class DeviceControl:
    def __init__(self, limits, targets, apply, advance, *, realtime, exhausted,
                 on_write=None, interval_s=None):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        # Every write reports the simulated time it spent, so the control period
        # is needed whichever side advances the clock.
        if interval_s is None or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("a controlled world must declare its control interval")
        self.interval_s = interval_s
        if set(limits) != set(targets) or not limits:
            raise ValueError("Every actuator needs limits and an initial target")
        self.limits = copy.deepcopy(limits)
        self.targets = copy.deepcopy(targets)
        self.apply, self.advance = apply, advance
        self.on_write = on_write
        self.realtime, self.exhausted = realtime, exhausted
        self.frame = 0
        self.commands = []

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

    def write(self, device, target, *, for_s=None, arrived=None):
        """Replace this actuator's goal and execute it; report the time spent.

        Where the write is what advances physics it runs exactly one control
        interval. Where a background clock runs the world it waits for the goal,
        bounded by for_s. Either way the caller learns how much simulated time
        this call was worth, which is the only clock a controller reads.
        """
        value = self._target(device, target)
        if self.exhausted():
            raise RuntimeError("Native episode budget exhausted")
        commands = {device: value}
        if self.on_write is not None:
            self.on_write(tuple(commands))
        self.targets.update(copy.deepcopy(commands))
        self.commands.append({"frame": self.frame, "op": "write", "commands": commands})
        started = self.frame
        if not self.realtime:
            self.tick()
        elif for_s:
            self._settle(for_s, arrived)
        return {"accepted": True, "elapsed_s": (self.frame - started) * self.interval_s}

    def _settle(self, for_s, arrived):
        """Wait out the background clock until the goal is met or the budget ends.

        `arrived` is the device binding's to supply: it knows which actuator it
        owns, while this owns the clock.
        """
        frames = max(1, math.ceil(for_s / self.interval_s))
        limit = self.frame + frames
        # A stalled background clock must not hang the caller: bound the wait
        # in wall time too, and report whatever simulated time did pass.
        deadline = time.monotonic() + max(1.0, 4 * frames * self.interval_s)
        while self.frame < limit and not self.exhausted():
            if arrived is not None and arrived():
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(self.interval_s / 4)

    def stop(self, device, measured_target):
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
        if self.exhausted():
            raise RuntimeError("Native episode budget exhausted")
        self.apply(copy.deepcopy(self.targets))
        self.advance()
        self.frame += 1
