"""GPU and license configuration before native application creation."""

import os

from roboenv.provisioning.acceptance import acceptance_environment
from roboenv.provisioning.gpu import configure_gpu


def isaac_config(selection):
    return {"active_gpu": selection["vulkan_index"], "physics_gpu": 0,
            "multi_gpu": False, "max_gpu_count": 1}


def launch_isaaclab(backend, **kwargs):
    os.environ.update(acceptance_environment())
    selection = configure_gpu(backend)
    from isaaclab.app import AppLauncher

    class SingleGPUAppLauncher(AppLauncher):
        def _config_resolution(self, launcher_args):
            super()._config_resolution(launcher_args)
            # AppLauncher otherwise overwrites both indices with the CUDA ordinal.
            self._sim_app_config.update(isaac_config(selection))

    return SingleGPUAppLauncher(device="cuda:0", **kwargs).app
