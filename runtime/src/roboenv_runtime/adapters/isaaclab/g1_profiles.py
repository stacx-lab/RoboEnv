"""Isaac G1 profiles with the shared humanoid actuator contracts."""

from roboenv_runtime.adapters.humanoid_profiles import IDENTITY, META, NAMES  # noqa: F401
from roboenv_runtime.adapters.humanoid_profiles import profiles as humanoid_profiles
from .g1_cameras import profile as camera_profile


def profiles(model):
    return humanoid_profiles(model, camera_profile=camera_profile)
