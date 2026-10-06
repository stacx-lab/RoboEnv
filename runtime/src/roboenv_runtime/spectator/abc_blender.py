"""Independent Blender views through the existing spectator lifecycle."""

import numpy as np

from roboenv_runtime.adapters.abc.rendering import BlenderImages
from roboenv_runtime.adapters.urdf_geometry import matrix

from . import CameraSource


class Engine:
    def __init__(self, adapter):
        self.world = adapter._env
        self.images = None
        self.sources = {role: CameraSource(name, 640, 480) for role, name in
                        {"third_person": "top", "wrist": "left", "right_wrist": "right"}.items()}

    def open(self, views):
        specs = []
        model = self.world.env.model
        for view in views:
            spec = {"name": view.name, "width": view.width, "height": view.height}
            if view.source:
                name = self.sources[view.source].native
                spec.update(source=name, fov_y=float(model.cam_fovy[model.cam(name).id]))
            else:
                transform = matrix(view.pose)
                # ViewSpec is OpenCV; Blender and MuJoCo cameras use OpenGL axes.
                rotation = transform[:3, :3] @ np.diag([1, -1, -1])
                spec.update(fov_y=view.fov_y,
                            matrix_pose=[*transform[:3, 3], *rotation.ravel()])
            specs.append(spec)
        self.images = BlenderImages(self.world, specs)

    def capture(self, views):
        return self.images.read()

    def close(self):
        if self.images is not None:
            self.images.close()
            self.images = None
