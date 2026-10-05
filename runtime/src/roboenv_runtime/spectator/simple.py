"""Record SIMPLE's native stereo frames without changing its shared render graph."""

import numpy as np

from . import CameraSource


class Engine:
    def __init__(self, adapter):
        self.world = adapter._env
        self.sources = {
            name: CameraSource(spec["native"], spec["width"], spec["height"])
            for name, spec in self.world.model["cameras"].items()
        }

    def open(self, views):
        if any(view.pose is not None for view in views):
            raise ValueError("SIMPLE records existing stereo cameras; fixed spectator poses are unsupported")

    def capture(self, views):
        # Isaac's render() only reads existing native sensor buffers. Calling its
        # step()/orchestrator here would refresh the shared temporal render state.
        native = (self.world.env.isaac if self.world.sim_mode == "mujoco_isaac"
                  else self.world.env.mujoco)
        images = native.render()
        frames = {}
        for view in views:
            source = self.sources[view.source]
            frame = np.asarray(images[source.native])
            if frame.dtype != np.uint8 or frame.shape != (source.height, source.width, 3):
                raise ValueError(f"{view.source}: native renderer returned invalid RGB shape/type")
            if (view.width, view.height) != (source.width, source.height):
                from PIL import Image

                # Explicit output sizing is resampling, not extra scene detail.
                frame = np.asarray(Image.fromarray(frame).resize(
                    (view.width, view.height), resample=Image.Resampling.BILINEAR))
            frames[view.name] = frame
        return frames

    def close(self):
        # The world owns the source cameras and their native buffers.
        pass
