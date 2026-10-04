"""Separate MuJoCo GL context and framebuffer, reading the live model/data."""

from contextlib import contextmanager

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix

from . import CameraSource


class Engine:
    def __init__(self, adapter):
        self.world = adapter._env
        height, width = self.world._camera_size
        names = ({"third_person": "agentview", "wrist": "robot0_eye_in_hand"}
                 if adapter.name == "libero" else
                 {"third_person": "robot0_agentview_left", "wrist": "robot0_eye_in_hand"})
        self.sources = {role: CameraSource(name, width, height)
                        for role, name in names.items() if name in self.world._camera_names}
        self.gl = self.context = None

    def open(self, views):
        import mujoco as mj

        self.model, self.data = self.world.sim.model._model, self.world.sim.data._data
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.camera, self.options = mj.MjvCamera(), mj.MjvOption()
        self.perturb = mj.MjvPerturb()
        # Keep the established scene visibility, without borrowing its context.
        original = self.world.sim._render_context_offscreen
        if original is not None:
            for field in ("geomgroup", "sitegroup", "jointgroup", "tendongroup",
                          "actuatorgroup", "flags"):
                getattr(self.options, field)[:] = getattr(original.vopt, field)
        width, height = max(v.width for v in views), max(v.height for v in views)
        # Reuse the owner's EGL display/device selection, but create a new context.
        context_type = type(original.gl_ctx) if original is not None else mj.GLContext
        self.gl = context_type(width, height)
        with self._current():
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_100)
            # Resize this framebuffer, never model.vis or robosuite's shared renderer.
            mj.mjr_resizeOffscreen(width, height, self.context)
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

    @contextmanager
    def _current(self):
        self.gl.make_current()
        try:
            yield
        finally:
            original = self.world.sim._render_context_offscreen
            if original is not None:
                original.gl_ctx.make_current()

    def capture(self, views):
        import mujoco as mj

        frames = {}
        with self._current():
            for view in views:
                self.camera.type = mj.mjtCamera.mjCAMERA_FREE
                if view.source:
                    self.camera.type = mj.mjtCamera.mjCAMERA_FIXED
                    self.camera.fixedcamid = mj.mj_name2id(
                        self.model, mj.mjtObj.mjOBJ_CAMERA, self.sources[view.source].native)
                mj.mjv_updateScene(self.model, self.data, self.options, self.perturb,
                                   self.camera, mj.mjtCatBit.mjCAT_ALL, self.scene)
                if view.pose is not None:
                    transform = matrix(view.pose)
                    for camera in self.scene.camera:
                        camera.pos[:] = transform[:3, 3]
                        camera.forward[:] = transform[:3, 2]
                        camera.up[:] = -transform[:3, 1]
                        top = camera.frustum_near * np.tan(np.deg2rad(view.fov_y) / 2)
                        camera.frustum_top, camera.frustum_bottom = top, -top
                        camera.frustum_center = camera.frustum_width = 0
                rgb = np.empty((view.height, view.width, 3), dtype=np.uint8)
                viewport = mj.MjrRect(0, 0, view.width, view.height)
                mj.mjr_render(viewport, self.scene, self.context)
                mj.mjr_readPixels(rgb, None, viewport, self.context)
                frames[view.name] = rgb[::-1]
        return frames

    def close(self):
        if self.gl is not None:
            self.gl.make_current()
            if self.context is not None:
                self.context.free()
            self.gl.free()
            self.gl = self.context = None
            original = self.world.sim._render_context_offscreen
            if original is not None:
                original.gl_ctx.make_current()
