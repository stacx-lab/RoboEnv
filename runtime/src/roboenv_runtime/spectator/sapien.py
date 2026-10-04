"""Independent SAPIEN cameras, with no task render/randomization callbacks."""

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

from . import CameraSource


class Engine:
    def __init__(self, adapter):
        self.world = adapter._env
        self.scene = self.world.task.scene
        names = {"third_person": "head_camera", "left_wrist": "left_camera",
                 "right_wrist": "right_camera"}
        self.sources = {}
        for role, name in names.items():
            camera = self.world._camera(name)
            self.sources[role] = CameraSource(camera, camera.get_width(), camera.get_height())
        self.cameras = {}

    def open(self, views):
        import sapien

        for view in views:
            source = self.sources.get(view.source)
            near, far = ((source.native.get_near(), source.native.get_far())
                         if source else (0.01, 100.0))
            camera = self.scene.add_camera(
                "roboenv_spectator_" + view.name, view.width, view.height,
                np.deg2rad(view.fov_y), near, far)
            self.cameras[view.name] = camera
            if source:
                k = source.native.get_intrinsic_matrix().copy()
                k[0] *= view.width / source.width
                k[1] *= view.height / source.height
                camera.set_perspective_parameters(
                    near, far, k[0, 0], k[1, 1], k[0, 2], k[1, 2], k[0, 1])
            else:
                # SAPIEN cameras: forward +X, left +Y, up +Z.
                optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]])
                pose = matrix_pose(matrix(view.pose) @ np.linalg.inv(optical))
                camera.entity.set_pose(sapien.Pose(pose[:3], [pose[6], *pose[3:6]]))

    def capture(self, views):
        # Read live wrist mounts directly. Updating the task's source cameras
        # would mutate agent rendering state; _update_render also changes RNG.
        for view in views:
            if view.source:
                if view.source in {"left_wrist", "right_wrist"}:
                    side = view.source.split("_")[0]
                    mount = getattr(self.world.task.robot, side + "_camera")
                    pose = mount.get_pose()
                else:
                    pose = self.sources[view.source].native.get_global_pose()
                self.cameras[view.name].entity.set_pose(pose)
        self.scene.update_render()
        frames = {}
        for view in views:
            camera = self.cameras[view.name]
            camera.take_picture()
            pixels = np.asarray(camera.get_picture("Color"))
            frames[view.name] = (np.clip(pixels[..., :3], 0, 1) * 255).astype(np.uint8)
        return frames

    def close(self):
        for camera in self.cameras.values():
            self.scene.remove_camera(camera)
        self.cameras.clear()
