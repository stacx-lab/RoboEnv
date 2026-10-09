"""IsaacLab cameras over the author's exact kinematic workcell geometry.

MuJoCo owns forward kinematics and the original collision checks. IsaacLab owns
the USD scene and RGBD rendering. Rendering never advances the task clock.
"""

import numpy as np


class ReachingScene:
    def __init__(self, task, profiles):
        from roboenv_runtime.adapters.gpu import launch_isaaclab

        self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=True)
        import isaaclab.sim as sim_utils
        import mujoco
        import torch
        from isaaclab.sensors import Camera, CameraCfg
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.Gf = Gf
        self.task = task
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=.1, device="cuda:0"))
        stage = self.sim.stage
        light = sim_utils.DomeLightCfg(intensity=500, color=(.85, .9, 1.))
        light.func("/World/Light", light)
        materials = {}

        def paint(shape, color):
            key = tuple(round(float(v), 3) for v in color)
            if key not in materials:
                path = f"/World/Materials/m{len(materials)}"
                material = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.65)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[key] = material
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(materials[key])

        self.transforms = {}
        task.world._update_kinematics()
        model = task.world._model
        for i in range(model.ngeom):
            if model.geom_group[i] == 3:
                continue
            path = f"/World/Geometry/g{i}"
            parent = UsdGeom.Xform.Define(stage, path)
            self.transforms[i] = parent.AddTransformOp()
            kind = int(model.geom_type[i])
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[i]
                va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                shape = UsdGeom.Mesh.Define(stage, path + "/shape")
                shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(model.mesh_vert[va:va + vn]))
                shape.CreateFaceVertexCountsAttr([3] * int(fn))
                shape.CreateFaceVertexIndicesAttr(model.mesh_face[fa:fa + fn].ravel().tolist())
                shape.CreateSubdivisionSchemeAttr("none")
                na, nn = model.mesh_normaladr[mesh_id], model.mesh_normalnum[mesh_id]
                normals = model.mesh_normal[na:na + nn][model.mesh_facenormal[fa:fa + fn]]
                shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3)))
                shape.SetNormalsInterpolation("faceVarying")
            elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
                shape = UsdGeom.Cube.Define(stage, path + "/shape")
                shape.CreateSizeAttr(2)
                shape.AddScaleOp().Set(Gf.Vec3f(3, 3, .01))
            else:
                raise ValueError(f"Unsupported source visual geometry {kind}")
            material = int(model.geom_matid[i])
            color = model.mat_rgba[material, :3] if material >= 0 else model.geom_rgba[i, :3]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                color = [.18, .2, .24]
            UsdGeom.Gprim(shape.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
            paint(shape, color)
        self.target_shapes = []
        for i, target in enumerate(task.targets):
            marker = UsdGeom.Xform.Define(stage, f"/World/Targets/t{i}")
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = target["rotation"], target["position"]
            marker.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
            shapes = []
            for axis in range(3):
                shape = UsdGeom.Cylinder.Define(stage, f"/World/Targets/t{i}/axis{axis}")
                shape.CreateRadiusAttr(.0025)
                shape.CreateHeightAttr(.045)
                shape.CreateAxisAttr("XYZ"[axis])
                offset = [0., 0., 0.]
                offset[axis] = .0225
                shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
                color = [float(k == axis) for k in range(3)]
                shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                paint(shape, color)
                shapes.append(shape.CreateDisplayOpacityAttr([1.]))
            self.target_shapes.append(shapes)
        self.cameras = {}
        for name, profile in profiles.items():
            if profile["type"] != "rgbd":
                continue
            mount = profile["extrinsics"]["mount"]
            camera = Camera(CameraCfg(
                prim_path=f"/World/{name}", update_period=0., height=900, width=1600,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=38.4,
                                               clipping_range=(.05, 15.)),
                offset=CameraCfg.OffsetCfg(pos=tuple(mount["translation_m"]),
                                          rot=tuple(mount["rotation_xyzw"]), convention="ros")))
            self.cameras[name] = camera
        self.sim.reset()
        for camera in self.cameras.values():
            camera.set_intrinsic_matrices(torch.tensor(
                [[[1000., 0, 800.], [0, 1000., 450.], [0, 0, 1.]]], device=self.sim.device))
        self.sync()

    def sync(self):
        self.task.world._update_kinematics()
        data = self.task.world._data
        for i, operation in self.transforms.items():
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = data.geom_xmat[i].reshape(3, 3), data.geom_xpos[i]
            operation.Set(self.Gf.Matrix4d(pose.T.tolist()))
        for i, shapes in enumerate(self.target_shapes):
            for shape in shapes:
                shape.Set([.2 if self.task.solved_by[i] >= 0 else 1.])

    def capture(self, name, directory):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        from ...common.observations import camera_state

        self.sync()
        for _ in range(3):
            self.sim.render()
            ensure_isaac_rtx_render_update()
        return camera_state(self.cameras[name], directory, self.task.frame * .1)

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None
