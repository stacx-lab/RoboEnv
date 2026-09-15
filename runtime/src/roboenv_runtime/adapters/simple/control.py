"""Record accepted commands and watchdog stops at their simulation boundaries."""

import copy
import time

from roboenv_runtime.adapters.humanoid_control import Control as HumanoidControl

from .profiles import NAMES, POSTURE_AXES, profiles


class Control(HumanoidControl):
    def __init__(self, native, model, *, realtime=False, command_timeout_s=0.5):
        super().__init__(
            native,
            model,
            realtime=realtime,
            command_timeout_s=command_timeout_s,
            profile_factory=profiles,
            names=NAMES,
            posture_axes=POSTURE_AXES,
        )
        self.expiring = False
        self.events = []

    def batch_write(self, commands):
        frame = self.native.steps
        result = super().batch_write(commands)
        self.events.append({"frame": frame, "op": "write", "commands": copy.deepcopy(commands)})
        return result

    def stop(self, name):
        result = super().stop(name)
        self.events.append(
            {
                "frame": self.native.steps,
                "op": "stop",
                "device": name,
                "reason": "timeout" if self.expiring else "requested",
            }
        )
        return result

    def tick(self):
        self.expiring = self.base_deadline is not None and time.monotonic() >= self.base_deadline
        try:
            super().tick()
        finally:
            self.expiring = False
