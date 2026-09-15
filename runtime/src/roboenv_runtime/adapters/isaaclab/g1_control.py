"""Isaac G1 specialization of the shared retained-target humanoid control."""

import time  # noqa: F401 - shared clock exposed for existing control tests

from roboenv_runtime.adapters.humanoid_control import Control as HumanoidControl
from .g1_profiles import profiles


class Control(HumanoidControl):
    def __init__(self, native, model, *, realtime=False, command_timeout_s=0.5):
        super().__init__(native, model, realtime=realtime,
                         command_timeout_s=command_timeout_s, profile_factory=profiles)
