"""Independent device targets; one world owns stepping and evidence."""

import copy
import math

from roboenv.devices.velocity import command as velocity_command

from .profiles import NAMES, profiles


class Control:
    def __init__(self, native, model):
        self.native, self.model = native, model
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
        for value in target:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Target must contain finite numbers")
        return list(target)

    def write(self, name, args):
        target = self.validate(name, args)
        self.native.require_active()
        self.targets[name] = target
        self.events.append(
            {"frame": self.native.steps, "op": "write", "device": name, "args": copy.deepcopy(args)}
        )
        return {"accepted": True, "elapsed_s": 0.0}

    def stop(self, name, *, reason="requested"):
        if name not in NAMES:
            raise ValueError("Unknown device")
        value = (
            [0, 0, 0, *self.targets[name][3:]]
            if name == "base"
            else self.native.read_device(name)["positions"]
        )
        self.targets[name] = list(value)
        self.events.append(
            {"frame": self.native.steps, "op": "stop", "device": name, "reason": reason}
        )
        return {"stopped": True, **self.profiles[name]["intrinsics"]["stop"]}

    def tick(self):
        if self.native.done:
            return
        self.native.advance(self.targets)
