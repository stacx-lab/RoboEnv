"""Defaults for licensed simulator startup; explicit opt-outs take precedence."""

import os


def acceptance_environment(environment=None):
    result = dict(os.environ if environment is None else environment)
    result.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    result.setdefault("ROBOENV_BEHAVIOR_ACCEPT_LICENSE", "YES")
    return result
