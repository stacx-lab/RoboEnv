"""URDF visual geometry in Isaac Lab, driven by the instrument process clock.

This instrument-command mode has no robot or contact dynamics. Unsupported
state-to-joint mappings are recorded explicitly rather than guessed from prose.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def material(stage, shape, mesh, directory, index):
    """Preserve URDF vertex colors and mesh UV textures in the Isaac stage."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    visual = mesh.visual
    native = getattr(visual, "material", None)
    color = getattr(visual, "main_color", getattr(native, "main_color", [180, 180, 180, 255]))
    color = np.asarray(color, dtype=float)[:3] / 255.0
    shape.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    image = getattr(native, "image", None)
    if image is None:
        image = getattr(native, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is None or uv is None:
        if visual.kind in {"vertex", "face"}:
            colors = visual.vertex_colors if visual.kind == "vertex" else visual.face_colors
            shape.CreateDisplayColorAttr(
                Vt.Vec3fArray.FromNumpy(np.asarray(colors[:, :3] / 255.0, dtype=np.float32))
            )
            shape.GetDisplayColorPrimvar().SetInterpolation(
                "vertex" if visual.kind == "vertex" else "uniform"
            )
        return
    path = Path(directory) / f"{index}.png"
    image.save(path)
    UsdGeom.PrimvarsAPI(shape).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
    base = str(shape.GetPath()) + "/material"
    mat = UsdShade.Material.Define(stage, base)
    surface = UsdShade.Shader.Define(stage, base + "/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    reader = UsdShade.Shader.Define(stage, base + "/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, base + "/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(path))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)


class InstrumentScene:
    def __init__(self, environment):
        from roboenv_runtime.adapters.gpu import launch_isaaclab

        self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=True)
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom, Vt
        from scipy.spatial.transform import Rotation
        from yourdfpy import URDF

        self.env, self.Gf, self.models = environment, Gf, {}
        self.textures = TemporaryDirectory(prefix="chemlab-textures-")
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=environment.control_interval, device="cuda:0")
        )
        light = sim_utils.DomeLightCfg(intensity=200.0)
        light.func("/World/Light", light)
        floor = UsdGeom.Mesh.Define(self.sim.stage, "/World/Floor")
        floor.CreatePointsAttr(
            [(-15.0, -15.0, -0.01), (15.0, -15.0, -0.01), (15.0, 15.0, -0.01), (-15.0, 15.0, -0.01)]
        )
        floor.CreateFaceVertexCountsAttr([4])
        floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.15, 0.2)])
        floor.CreateSubdivisionSchemeAttr("none")
        self.mapping_report = {}
        for i, (name, instrument) in enumerate(environment.instruments.items()):
            model = URDF.load(str(environment.depot / instrument.row["urdf"]))
            operations = {}
            for j, node in enumerate(model.scene.graph.nodes_geometry):
                _, geometry = model.scene.graph[node]
                mesh = model.scene.geometry[geometry]
                parent = UsdGeom.Xform.Define(self.sim.stage, f"/World/{name}/part{j}")
                operation = parent.AddTransformOp()
                shape = UsdGeom.Mesh.Define(self.sim.stage, f"/World/{name}/part{j}/shape")
                shape.CreatePointsAttr(
                    Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32))
                )
                shape.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
                shape.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
                shape.CreateSubdivisionSchemeAttr("none")
                material(self.sim.stage, shape, mesh, self.textures.name, f"{i}-{j}")
                operations[node] = operation
            mount = environment.profiles.get(name, {}).get("extrinsics", {}).get("mount", {})
            if (
                mount.get("parent", "world") != "world"
                or mount.get("parent_frame", "world") != "world"
            ):
                raise ValueError("ChemLab instrument mounts must be expressed in world coordinates")
            pose = np.eye(4)
            pose[:3, 3] = mount.get("translation_m", [0.0, 0.0, 0.0])
            pose[:3, :3] = Rotation.from_quat(
                mount.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            ).as_matrix()
            self.models[name] = (model, operations, pose)
            self.mapping_report[name] = {"synchronized": [], "unsupported": [], "unbound": []}
        self.sim.reset()
        self.sync()

    def spectator_pose(self):
        from itertools import product

        from ...common.spectator import look_at

        bounds = []
        for model, _, pose in self.models.values():
            corners = np.array(list(product(*model.scene.bounds.T)))
            corners = corners @ pose[:3, :3].T + pose[:3, 3]
            bounds.append([corners.min(axis=0), corners.max(axis=0)])
        bounds = np.asarray(bounds)
        lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
        center = (lower + upper) * 0.5
        extent = max(float(np.max(upper - lower)), 0.1)
        return look_at((center + extent * np.array([1.4, -1.8, 1.1])).tolist(), center.tolist())

    def sync(self):
        for name, (model, operations, pose) in self.models.items():
            instrument = self.env.instruments[name]
            state = instrument.read(owner=True)
            config = {}
            report = {"synchronized": [], "unsupported": [], "unbound": []}
            for key, binding in instrument.contract.get("model_bindings", {}).items():
                if binding.get("channel") != "joint":
                    continue
                slot, target = binding.get("state_id"), binding.get("target")
                if slot is None:
                    report["unbound"].append(key)
                    continue
                value = state.get(slot, {}).get("value")
                unit = instrument.contract["state_slots"].get(slot, {}).get("unit")
                transform = instrument.config.get("joint_transforms", {}).get(target)
                if (
                    target not in model.joint_map
                    or type(value) not in (int, float)
                    or not np.isfinite(value)
                    or (
                        transform is None
                        and (unit != binding.get("unit") or unit not in {"m", "rad"})
                    )
                ):
                    report["unsupported"].append(key)
                    continue
                transform = transform or {}
                config[target] = value * transform.get("scale", 1.0) + transform.get("offset", 0.0)
                report["synchronized"].append(key)
            model.update_cfg(config)
            for node, operation in operations.items():
                transform = pose @ model.scene.graph[node][0]
                operation.Set(self.Gf.Matrix4d(transform.T.tolist()))
            self.mapping_report[name] = report
        self.sim.render()

    def close(self):
        self.textures.cleanup()
        if self.app is not None:
            self.app.close()
            self.app = None
