"""Independent device targets; one world owns stepping, expiry and evidence."""

import copy
import math
import time

from roboenv.devices.velocity import command as velocity_command

from .profiles import CONTROL_INTERVAL_S, NAMES, profiles


class Control:
    def __init__(self, native, model, *, realtime=True):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        self.native, self.model = native, model
        self.realtime = realtime
        self.events = []
        self.names = NAMES
        control = copy.deepcopy(model["control"])
        self.profiles = profiles({**model, "control": control})
        posture = control["modes"]["locomotion"]["posture"]["components"]
        self.targets = {"base": [0, 0, 0, *[c["default"] for c in posture]]}
        for name in NAMES[1:]:
            part = model["parts"][name]
            self.targets[name] = [part["home"][j["name"]] for j in part["joints"]]

    def validate(self, name, args):
        if name not in NAMES or not isinstance(args, dict):
            raise ValueError("Expected a known device and command object")
        if name == "base":
            value = velocity_command(self.model["control"], args)
            return [value["velocity"][a] for a in ("vx", "vy", "wz")] + [
                value["posture"][a] for a in ("height", "roll", "pitch", "yaw")
            ]
        if set(args) != {"target"}:
            raise ValueError("Joint writes require only target")
        target = args["target"]
        joints = self.model["parts"][name]["joints"]
        if not isinstance(target, list) or len(target) != len(joints):
            raise ValueError("Target dimension mismatch")
        for value, joint in zip(target, joints, strict=True):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Target must contain finite numbers")
            limits = joint["limits"]
            if limits and not limits["min"] <= value <= limits["max"]:
                raise ValueError("Target exceeds limits")
        return list(target)

    def write(self, name, args, *, for_s=None, arrived=None):
        target = self.validate(name, args)
        self.native.require_active()
        self.targets[name] = target
        self.events.append(
            {"frame": self.native.steps, "op": "write", "device": name, "args": copy.deepcopy(args)}
        )
        started = self.native.steps
        if not self.realtime:
            self.native.advance(self.targets)
        elif for_s:
            self._settle(for_s, arrived)
        return {"accepted": True,
                "elapsed_s": (self.native.steps - started) * CONTROL_INTERVAL_S}

    def _settle(self, for_s, arrived):
        """Wait out the background clock until the goal is met or the budget ends.

        `arrived` is the device binding's to supply: it knows which actuator it
        owns, while this owns the clock.
        """
        interval = CONTROL_INTERVAL_S
        frames = max(1, math.ceil(for_s / interval))
        limit = self.native.steps + frames
        # A stalled background clock must not hang the caller: bound the wait
        # in wall time too, and report whatever simulated time did pass.
        deadline = time.monotonic() + max(1.0, 4 * frames * interval)
        while self.native.steps < limit and not self.native.done:
            if arrived is not None and arrived():
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(interval / 4)

    def stop(self, name, *, reason="requested"):
        if name not in NAMES:
            raise ValueError("Unknown device")
        value = (
            [0, 0, 0, *self.targets[name][3:]]
            if name == "base"
            else self.native.read_device(name)["positions"]
        )
        self.targets[name] = list(value)
        if name == "base":
            self.events.append(
            {"frame": self.native.steps, "op": "stop", "device": name, "reason": reason}
        )
        return {"stopped": True, **self.profiles[name]["intrinsics"]["stop"]}

    def tick(self):
        if self.native.done:
            return
        self.native.advance(self.targets)
