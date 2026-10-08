"""Owner-only spectator cameras. Experimental until the native isolation gate passes.

One renderer belongs to one world generation and its creating thread. Rendering
never schedules physics. Native resources stay separate from device cameras.
"""

import threading
from dataclasses import dataclass, replace
from importlib import import_module

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix


@dataclass(frozen=True)
class ViewSpec:
    """Follow a named camera, or use a world pose (metres, xyzw, OpenCV axes)."""

    name: str
    source: str | None = None
    width: int | None = None
    height: int = 720
    pose: tuple[float, ...] | None = None
    fov_y: float = 45.0

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("View name must contain only letters, digits and underscores")
        if (self.source is None) == (self.pose is None):
            raise ValueError("A view needs either a source camera or a fixed pose")
        for value in (self.height, self.width if self.width is not None else 2):
            if type(value) is not int or value < 2:
                raise ValueError("View dimensions must be integers >= 2")
        if self.pose is not None:
            matrix(self.pose)
        if not np.isfinite(self.fov_y) or not 0 < self.fov_y < 180:
            raise ValueError("Vertical field of view must be between 0 and 180 degrees")


@dataclass(frozen=True)
class CameraSource:
    native: object
    width: int
    height: int
    fov_y: float | None = None


class SpectatorRenderer:
    """Common view resolution and RGB output contract; engines own native details."""

    def __init__(self, engine, views):
        self._thread = threading.get_ident()
        self._engine = engine
        self._closed = False
        sources = engine.sources
        self.views = tuple(views) if views is not None else tuple(
            ViewSpec(name, source=name) for name in sources
        )
        try:
            if not self.views or len({v.name for v in self.views}) != len(self.views):
                raise ValueError("Select at least one view, with distinct names")
            resolved = []
            for view in self.views:
                if view.source is not None and view.source not in sources:
                    raise ValueError(f"Unknown camera {view.source!r}; available: {list(sources)}")
                source = sources.get(view.source)
                aspect = source.width / source.height if source else 16 / 9
                width = view.width or max(2, round(view.height * aspect / 2) * 2)
                fov_y = source.fov_y if source and source.fov_y is not None else view.fov_y
                resolved.append(replace(view, width=width, fov_y=fov_y))
            self.views = tuple(resolved)
            engine.open(self.views)
        except BaseException:
            engine.close()
            raise

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Spectator cameras must run on the world's owner thread")

    def capture(self):
        self._check_thread()
        if self._closed:
            raise RuntimeError("Spectator renderer is closed")
        frames = self._engine.capture(self.views)
        for view in self.views:
            frame = frames[view.name]
            if frame.dtype != np.uint8 or frame.shape != (view.height, view.width, 3):
                raise ValueError(f"{view.name}: native renderer returned invalid RGB shape/type")
        # Detach from native buffers before handing images to another process.
        return {name: frame.copy() for name, frame in frames.items()}

    def close(self):
        self._check_thread()
        if not self._closed:
            self._closed = True
            self._engine.close()


def create_spectator_renderer(adapter, views=None):
    """Create once after reset; close before resetting or destroying the world."""
    engines = {"abc": "mujoco", "libero": "mujoco", "robocasa": "mujoco", "robotwin": "sapien",
               "robodojo": "isaac", "behavior": "isaac", "simple": "simple", "isaaclab": "isaaclab"}
    if adapter.name not in engines:
        raise ValueError(f"Spectator rendering is not supported for {adapter.name}")
    if adapter.name == "abc" and adapter._env.camera_backend == "blender":
        engines["abc"] = "abc_blender"
    module = import_module(f"{__name__}.{engines[adapter.name]}")
    engine = module.Engine(adapter)
    if adapter.name == "simple" and views is None:
        views = [ViewSpec(name, source=name, width=source.width, height=source.height)
                 for name, source in engine.sources.items()]
    return SpectatorRenderer(engine, views)
