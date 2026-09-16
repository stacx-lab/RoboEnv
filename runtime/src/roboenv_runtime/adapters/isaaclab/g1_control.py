"""Validate and latch component targets before one shared native interval."""

import math
import time

from roboenv.devices.velocity import command as base_command

from .g1_profiles import NAMES, profiles


class Control:
    def __init__(self, native, model, *, realtime=False, command_timeout_s=0.5):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        if command_timeout_s is not None and (
            isinstance(command_timeout_s, bool)
            or not isinstance(command_timeout_s, (int, float))
            or not math.isfinite(command_timeout_s)
            or command_timeout_s <= 0
        ):
            raise ValueError("command_timeout_s must be positive and finite, or null")
        control = model["control"]
        if (
            control["mode"] != "motion"
            or control["body_frame"] != "pelvis"
            or set(control["modes"]) != {"locomotion"}
        ):
            raise ValueError("G1 policy requires pelvis control")
        mode = control["modes"]["locomotion"]
        velocity, posture = mode["velocity"], mode["posture"]
        if (
            velocity is None
            or posture is None
            or velocity["frame"] != "body"
            or posture["reference"] != "ground"
            or [(c["axis"], c["units"]) for c in velocity["components"]]
            != [("vx", "m/s"), ("vy", "m/s"), ("wz", "rad/s")]
            or [(c["axis"], c["units"]) for c in posture["components"]] != [("height", "m")]
        ):
            raise ValueError(
                "G1 policy supports only body vx/vy/wz and ground-relative pelvis height"
            )
        height = posture["components"][0]["default"]
        base_command(
            control,
            {
                "mode": "locomotion",
                "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": height},
            },
        )
        self.native = native
        self.model = model
        self.realtime = realtime
        self.command_timeout_s = command_timeout_s
        self.base_deadline = None
        execution = (
            {"mode": "realtime", "command_timeout_s": command_timeout_s}
            if realtime
            else control["execution"]
        )
        self.profiles = profiles({**model, "control": {**control, "execution": execution}})
        self.targets = {"base": [0, 0, 0, height]}
        for name in NAMES[1:]:
            part = model["parts"][name]
            self.targets[name] = [part["home"][joint["name"]] for joint in part["joints"]]

    def validate(self, name, args):
        if name not in NAMES or not isinstance(args, dict):
            raise ValueError("Expected a known device and command object")
        if name == "base":
            value = base_command(self.model["control"], args)
            return [value["velocity"][axis] for axis in ("vx", "vy", "wz")] + [
                value["posture"]["height"]
            ]
        if set(args) != {"target"}:
            raise ValueError("Joint writes require only target")
        target = args["target"]
        bounds = [j["limits"] for j in self.model["parts"][name]["joints"]]
        if not isinstance(target, list) or len(target) != len(bounds):
            raise ValueError("Target dimension mismatch")
        for value, limits in zip(target, bounds, strict=True):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("Target must contain finite numbers")
            if limits is not None and not limits["min"] <= value <= limits["max"]:
                raise ValueError("Target exceeds limits")
        return list(target)

    def write(self, name, args):
        self.native.require_active()
        target = self.validate(name, args)
        combined = {**self.targets, name: target}
        if not self.realtime:
            self.native.advance(combined)
        self.targets = combined
        if self.realtime and name == "base":
            self.base_deadline = (
                time.monotonic() + self.command_timeout_s
                if self.command_timeout_s is not None
                else None
            )
        return {"accepted": True}

    def tick(self):
        """One owner-clock interval; never step a terminal episode."""
        if self.native.done:
            return
        if self.base_deadline is not None and time.monotonic() >= self.base_deadline:
            self.stop("base")
        self.native.advance(self.targets)

    def stop(self, name):
        if name not in NAMES:
            raise ValueError("Unknown device")
        value = (
            [0, 0, 0, self.targets[name][3]]
            if name == "base"
            else self.native.read_device(name)["positions"]
        )
        self.targets[name] = list(value)
        if name == "base":
            self.base_deadline = None
        return {"stopped": True, **self.profiles[name]["intrinsics"]["stop"]}
