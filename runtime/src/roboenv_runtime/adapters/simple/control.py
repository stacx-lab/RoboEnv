"""Independent device targets; one world owns stepping, expiry and evidence."""

import copy
import math
import time

from roboenv.devices.velocity import command as velocity_command

from .profiles import NAMES, profiles


class Control:
    def __init__(self, native, model, *, realtime=True, command_timeout_s=0.5):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        if command_timeout_s is not None and (
            type(command_timeout_s) not in (int, float)
            or not math.isfinite(command_timeout_s)
            or command_timeout_s <= 0
        ):
            raise ValueError("command_timeout_s must be positive and finite, or null")
        self.native, self.model = native, model
        self.realtime, self.command_timeout_s = realtime, command_timeout_s
        self.base_deadline = None
        self.events = []
        self.names = NAMES
        control = copy.deepcopy(model["control"])
        control["execution"] = (
            {"mode": "realtime", "command_timeout_s": command_timeout_s}
            if realtime
            else {"mode": "stepped", "interval_s": 0.02}
        )
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

    def write(self, name, args):
        target = self.validate(name, args)
        self.native.require_active()
        self.targets[name] = target
        self.events.append(
            {"frame": self.native.steps, "op": "write", "device": name, "args": copy.deepcopy(args)}
        )
        if self.realtime and name == "base":
            self.base_deadline = (
                time.monotonic() + self.command_timeout_s
                if self.command_timeout_s is not None
                else None
            )
        if not self.realtime:
            self.native.advance(self.targets)
        return {"accepted": True}

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
            self.base_deadline = None
        self.events.append(
            {"frame": self.native.steps, "op": "stop", "device": name, "reason": reason}
        )
        return {"stopped": True, **self.profiles[name]["intrinsics"]["stop"]}

    def tick(self):
        if self.native.done:
            return
        if self.base_deadline is not None and time.monotonic() >= self.base_deadline:
            self.stop("base", reason="timeout")
        self.native.advance(self.targets)
