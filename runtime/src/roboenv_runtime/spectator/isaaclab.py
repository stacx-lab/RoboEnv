"""Dedicated Isaac Lab products using embodiment-owned view providers.

Render calls preserve physics, but Go2 wrist RGB can exceed the image-isolation
probe tolerance in the shared RTX graph. Recording remains experimental.
"""

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix

from . import CameraSource
from .isaac import Engine as IsaacEngine


class Engine(IsaacEngine):
    warmup_frames = 64

    def __init__(self, adapter):
        self.world, self.backend = adapter.environment, "isaaclab"
        if getattr(self.world, "_spectator_opened", False):
            raise ValueError("Create only one spectator renderer per Isaac Lab world")
        self.sources = {
            name: CameraSource(view, view.width, view.height, fov_y=view.fov_y)
            for name, view in self.world.spectator_sources().items()
        }
        self.products = []

    def create_product(self, rep, path, view):
        # Match the tiled camera pipeline used by the pinned Isaac Lab renderer.
        return rep.create.render_product_tiled(
            cameras=[path], tile_resolution=(view.width, view.height)
        )

    def get_stage(self):
        return self.world.native.sim.stage

    def source_attributes(self, source):
        aperture = 20.0
        return {
            "projection": "perspective",
            "verticalAperture": aperture,
            "horizontalAperture": aperture * source.width / source.height,
            "focalLength": aperture / (2 * np.tan(np.deg2rad(source.native.fov_y / 2))),
            "horizontalApertureOffset": 0.0,
            "verticalApertureOffset": 0.0,
            "clippingRange": (0.01, 100.0),
        }

    def view_transform(self, view):
        pose = self.sources[view.source].native.pose() if view.source else view.pose
        return matrix(pose) @ np.diag([1, -1, -1, 1])

    def render(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        self.world.native.sim.render()
        # Headless Isaac Lab 3 cameras drive RTX on read. Pump its render-only
        # helper for our independent products too; it disables physics updates.
        ensure_isaac_rtx_render_update()

    def open(self, views):
        super().open(views)
        self.world._spectator_opened = True
        # Products share Kit's render graph and remain alive until application close.
        self.world._spectator_resources = self
        # New RTX products invalidate temporal histories for the shared graph.
        # Let them converge before returning control to device-camera readers.
        for _ in range(self.warmup_frames):
            self.capture(views)
