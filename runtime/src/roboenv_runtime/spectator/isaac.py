"""Isaac spectator products share the native render graph, never agent products.

Products live until world shutdown. This path needs the native feasibility gate;
independent SyntheticData teardown is intentionally not attempted during control.
"""

import secrets
from contextlib import nullcontext

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix

from . import CameraSource


class Engine:
    def __init__(self, adapter):
        self.world, self.backend = adapter._env, adapter.name
        self.sources, self.products = {}, []
        if self.backend == "robodojo":
            names = {"third_person": "head_camera", "left_wrist": "left_camera",
                     "right_wrist": "right_camera"}
            for role, name in names.items():
                camera = self.world._camera(name)
                width, height = camera.get_resolution()
                self.sources[role] = CameraSource(camera, width, height)
        else:
            roles = {"left_arm": "left_wrist", "right_arm": "right_wrist", "trunk": "third_person"}
            for name, sensor_name in self.world.cameras.items():
                parent = self.world.device_facts[name]["extrinsics"]["mount"]["parent"]
                role = roles.get(parent)
                if role is None or role in self.sources:
                    raise ValueError(f"No unique spectator role for camera mounted on {parent}")
                sensor = self.world.robot.sensors[sensor_name]
                self.sources[role] = CameraSource(sensor, sensor.image_width, sensor.image_height)

    def open(self, views):
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        self.stage = get_current_stage()
        self.cameras, self.annotators = {}, {}
        self.root = "/RoboEnvSpectator_" + secrets.token_hex(8)
        context = nullcontext()
        if self.backend == "behavior":
            import omnigibson as og
            context = og.sim.editing_usd()
        with context:
            for view in views:
                path = self.root + "/" + view.name
                camera = UsdGeom.Camera.Define(self.stage, path)
                source = self.sources.get(view.source)
                if source:
                    original = UsdGeom.Camera.Get(self.stage, str(source.native.prim_path))
                    for name in ("projection", "focalLength", "horizontalAperture",
                                 "verticalAperture", "horizontalApertureOffset",
                                 "verticalApertureOffset", "clippingRange"):
                        value = original.GetPrim().GetAttribute(name).Get()
                        camera.GetPrim().GetAttribute(name).Set(value)
                else:
                    aperture = 20.0
                    camera.GetVerticalApertureAttr().Set(aperture)
                    camera.GetHorizontalApertureAttr().Set(aperture * view.width / view.height)
                    focal = aperture / (2 * np.tan(np.deg2rad(view.fov_y) / 2))
                    camera.GetFocalLengthAttr().Set(focal)
                    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100))
                self.cameras[view.name] = UsdGeom.Xformable(camera).AddTransformOp()
                product = rep.create.render_product(path, (view.width, view.height), force_new=True)
                self.products.append(product)
                rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                rgb.attach(product)
                self.annotators[view.name] = rgb

    def capture(self, views):
        from pxr import Gf

        for view in views:
            if view.source:
                source = self.sources[view.source].native
                if self.backend == "robodojo":
                    position, q = source.get_world_pose(camera_axes="usd")
                    pose = [*position, *q[1:], q[0]]
                else:
                    position, q = source.get_position_orientation()
                    pose = [*position.cpu().numpy(), *q.cpu().numpy()]
                transform = matrix(pose)
            else:
                transform = matrix(view.pose) @ np.diag([1, -1, -1, 1])
            self.cameras[view.name].Set(Gf.Matrix4d(transform.T.tolist()))
        if self.backend == "robodojo":
            self.world._env.render()
        else:
            import omnigibson as og
            og.sim.render()
        return {v.name: np.asarray(self.annotators[v.name].get_data())[..., :3] for v in views}

    def close(self):
        # The owner must close the world after this renderer. Never detach a
        # live shared SyntheticData graph or rewrite its global render settings.
        pass
