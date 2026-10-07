"""ABC Blender selection and per-state camera cache; physics stays in ABCEnv."""

import os
import subprocess

from roboenv.provisioning.abc_blender import paths
from roboenv.provisioning.assets import _digest
from roboenv_runtime.paths import repo_root


def settings():
    binary, hdri = paths(repo_root())
    samples = int(os.environ.get("ABC_BLENDER_SAMPLES", "16"))
    denoiser = os.environ.get("ABC_BLENDER_DENOISER", "OPENIMAGEDENOISE")
    if samples < 1 or denoiser not in {"OPENIMAGEDENOISE", "OPTIX", "NONE"}:
        raise ValueError("Invalid ABC Blender samples or denoiser")
    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                             text=True, timeout=30).stdout.splitlines()[0]
    return {
        "blender": str(binary), "hdri": str(hdri), "samples": samples,
        "denoiser": denoiser, "gpu_id": None,
        "identity": {"version": version, "binary_sha256": _digest(binary),
                     "hdri_sha256": _digest(hdri), "samples": samples,
                     "denoiser": denoiser, "gpu_id": None},
    }


class BlenderImages:
    def __init__(self, world, views=None):
        from abc_sim.rendering.blender.live import BlenderCameraProvider

        self.world = world
        options = {key: value for key, value in world.blender_settings.items() if key != "identity"}
        self.provider = BlenderCameraProvider(
            world.env.model, world.env.data, width=640, height=480,
            camera_names=world._camera_names, views=views, **options,
        )
        self.frame = world.control.frame
        self.images = self.provider.initial_frames()

    def read(self):
        if self.frame != self.world.control.frame:
            self.images = self.provider.frames_for_step(self.world.control.frame, self.world.timestamp)
            self.frame = self.world.control.frame
        return self.images

    def close(self):
        self.provider.close()
