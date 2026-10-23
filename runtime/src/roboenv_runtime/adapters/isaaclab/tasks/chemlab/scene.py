"""URDF visual geometry in Isaac Lab, driven by the instrument process clock.

This instrument-command mode has no robot or contact dynamics. Unsupported
state-to-joint mappings are recorded explicitly rather than guessed from prose.
"""

import hashlib
import json
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .geometry import binding_report, component_links, joint_bindings, screen_quad
from .readouts import capture, displays, readout_bindings


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
    base = str(shape.GetPath()) + "/material"
    mat = UsdShade.Material.Define(stage, base)
    surface = UsdShade.Shader.Define(stage, base + "/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    properties = getattr(native, "kwargs", {})

    def scalar(name, default):
        value = properties.get(name, default)
        return float(value[0] if isinstance(value, (list, tuple)) else value)

    opacity = scalar("d", 1.0)
    factor = getattr(native, "baseColorFactor", None)
    if factor is not None:
        opacity *= float(factor[3]) / 255.0
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shape.CreateDisplayOpacityAttr([opacity])
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(scalar("pr", 0.6))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(scalar("pm", 0.0))
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if image is None or uv is None:
        return surface
    path = Path(directory) / f"{index}.png"
    image.save(path)
    UsdGeom.PrimvarsAPI(shape).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
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
    return surface


class InstrumentScene:
    def __init__(self, environment):
        from roboenv_runtime.adapters.gpu import launch_isaaclab

        self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=True)
        import isaaclab.sim as sim_utils
        from pxr import Gf, Sdf, Tf, UsdGeom, UsdLux, UsdShade, Vt
        from scipy.spatial.transform import Rotation

        self.env, self.Gf, self.models = environment, Gf, {}
        self.panels, self.lamps, self.joints = {}, {}, {}
        self.physics, self.colors, self.properties = {}, {}, {}
        self.panel_failures, self.visual_evidence = {}, {}
        self.dashboard_points = []
        self.textures = TemporaryDirectory(prefix="chemlab-textures-")
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=environment.control_interval / 10, device="cuda:0")
        )
        light = sim_utils.DomeLightCfg(intensity=500.0)
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
            model = environment.projections[name].model
            operations, shapes, surfaces, nodes_by_link = {}, {}, {}, {}
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
                surfaces[node] = material(
                    self.sim.stage, shape, mesh, self.textures.name, f"{i}-{j}"
                )
                operations[node] = operation
                shapes[node] = shape
                link = model.scene.graph.transforms.parents[node]
                nodes_by_link.setdefault(link, []).append(node)
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
            from .physics import InstrumentPhysics

            self.physics[name] = InstrumentPhysics(self, name, instrument, pose)
            self.joints[name] = joint_bindings(instrument, model)
            self.panels[name], self.lamps[name] = {}, {}
            self.colors[name], self.properties[name] = {}, {}
            self.panel_failures[name] = {}
            self.visual_evidence[name] = {
                "panels": {},
                "joints": {},
                "lamps": {},
                "supplements": environment.projections[name].supplements,
            }

            def component_nodes(component, instrument=instrument, nodes_by_link=nodes_by_link):
                return [
                    node
                    for link in component_links(instrument.contract, component)
                    for node in nodes_by_link.get(link, [])
                ]

            for display_index, (display, spec) in enumerate(displays(instrument).items()):
                component = spec.get("component_id")
                candidates = component_nodes(component)
                # A component may contain a screen and decorative text meshes.
                node = (
                    max(
                        candidates, key=lambda n: model.scene.geometry[model.scene.graph[n][1]].area
                    )
                    if candidates
                    else None
                )
                if node is not None:
                    mesh = model.scene.geometry[model.scene.graph[node][1]]
                    corners, normal = screen_quad(mesh)
                    parent = str(shapes[node].GetPath()).rsplit("/", 1)[0]
                else:
                    # Some released panel components explicitly have no mesh.
                    # Keep their pixels on a separate inspection board instead
                    # of claiming a guessed placement is manufacturer geometry.
                    parent = f"/World/{name}/panel_mirrors"
                    mirror = UsdGeom.Xform.Define(self.sim.stage, parent)
                    if not mirror.GetOrderedXformOps():
                        mirror.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
                    x = model.scene.bounds[1, 0] + 0.08 + (display_index % 3) * 0.22
                    z = 0.15 + (display_index // 3) * 0.16
                    corners = np.array(
                        [[x, 0, z], [x + 0.20, 0, z], [x + 0.20, 0, z + 0.12], [x, 0, z + 0.12]]
                    )
                    normal = np.array([0.0, -1.0, 0.0])
                    self.dashboard_points.extend(corners @ pose[:3, :3].T + pose[:3, 3])
                quad = UsdGeom.Mesh.Define(self.sim.stage, parent + f"/screen_{display}")
                quad.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(corners.astype(np.float32)))
                quad.CreateFaceVertexCountsAttr([4])
                quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                quad.CreateSubdivisionSchemeAttr("none")
                quad.CreateDoubleSidedAttr(True)
                UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                base = parent + f"/panel_material_{display}"
                mat = UsdShade.Material.Define(self.sim.stage, base)
                shader = UsdShade.Shader.Define(self.sim.stage, base + "/surface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
                reader = UsdShade.Shader.Define(self.sim.stage, base + "/uv")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                texture = UsdShade.Shader.Define(self.sim.stage, base + "/texture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.ConnectableAPI(), "result"
                )
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb"
                )
                shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)
                self.panels[name][display] = {
                    "texture": texture.CreateInput("file", Sdf.ValueTypeNames.Asset),
                    "node": node,
                    "corners": corners,
                    "normal": normal,
                    "component": component,
                    "kind": spec["kind"],
                    "spec": spec,
                    "placement": "instrument" if node is not None else "inspection_board",
                }

            panel_components = {panel["component"] for panel in self.panels[name].values()}
            for key, binding in instrument.contract.get("model_bindings", {}).items():
                slot, channel = binding.get("state_id"), binding.get("channel")
                component = binding.get("component_id")
                nodes = component_nodes(component)
                if channel == "material" and nodes:
                    self.colors[name][key] = [surfaces[node] for node in nodes]
                if binding.get("attribute", "").startswith("custom_property:") and slot:
                    property_name = binding["attribute"].split(":", 1)[1]
                    prim = self.sim.stage.GetPrimAtPath(f"/World/{name}")
                    attribute = prim.CreateAttribute(
                        "chemlab:" + Tf.MakeValidIdentifier(property_name),
                        Sdf.ValueTypeNames.String,
                        custom=True,
                    )
                    self.properties[name][key] = (slot, attribute)
                if (
                    channel in {"display", "material"}
                    and instrument.contract["state_slots"]
                    .get(slot, {})
                    .get("value_schema", {})
                    .get("type")
                    == "boolean"
                    and nodes
                    and component not in panel_components
                ):
                    lamps = []
                    for node in nodes:
                        lamp = UsdLux.SphereLight.Define(
                            self.sim.stage,
                            str(shapes[node].GetPath()).rsplit("/", 1)[0]
                            + "/lamp_"
                            + Tf.MakeValidIdentifier(key),
                        )
                        mesh = model.scene.geometry[model.scene.graph[node][1]]
                        position = mesh.centroid.copy()
                        # Place the emitter outside the underside, rather than
                        # inside the opaque luminous component that occludes it.
                        position[2] = mesh.bounds[0, 2] - 0.01
                        lamp.AddTranslateOp().Set(Gf.Vec3d(*position))
                        lamp.CreateRadiusAttr(0.025)
                        lamp.CreateIntensityAttr(0.0)
                        lamps.append((surfaces[node], lamp))
                    self.lamps[name][key] = (slot, lamps)
        self.sim.reset()
        self.bind_projections()
        self.sync()

    def bind_projections(self):
        for name, projection in self.env.projections.items():
            _, operations, pose = self.models[name]
            self.models[name] = (projection.model, operations, pose)
            projection.commit = lambda name=name: self.sync_geometry(name)

    def sync_geometry(self, name):
        from pxr import UsdGeom

        projection = self.env.projections[name]
        _, operations, pose = self.models[name]
        for node, operation in operations.items():
            transform = pose @ projection.node_transform(node)
            operation.Set(self.Gf.Matrix4d(transform.T.tolist()))
            if not np.allclose(np.array(operation.Get()).T, transform, atol=1e-9):
                raise RuntimeError("Isaac instrument transform readback differs")
            shape = UsdGeom.Imageable(operation.GetAttr().GetPrim())
            shape.CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited
                if projection.node_visible(node)
                else UsdGeom.Tokens.invisible
            )
        self.physics[name].sync(projection.config)

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
        if self.dashboard_points:
            lower = np.minimum(lower, np.min(self.dashboard_points, axis=0))
            upper = np.maximum(upper, np.max(self.dashboard_points, axis=0))
        center = (lower + upper) * 0.5
        extent = max(float(np.max(upper - lower)), 0.1)
        return look_at((center + extent * np.array([0.55, -1.5, 0.75])).tolist(), center.tolist())

    def panel_pose(self, name, display):
        from ...common.spectator import fit_view

        model, _, pose = self.models[name]
        panel = self.panels[name][display]
        transform = pose @ model.scene.graph[panel["node"]][0] if panel["node"] else pose
        corners = panel["corners"] @ transform[:3, :3].T + transform[:3, 3]
        normal = transform[:3, :3] @ panel["normal"]
        center = corners.mean(axis=0)
        # Avoid a vertical optical axis so the shared look_at has a defined up.
        normal = normal + np.array([0.0, -0.03, 0.0])
        extent = max(float(np.linalg.norm(corners[1] - corners[0])), 0.05)
        return fit_view(center + normal * extent, center, corners, margin=0.08)

    def sync(self):
        for name in self.models:
            instrument = self.env.instruments[name]
            state = instrument.read(owner=True)
            projection = self.env.projections[name]
            config = projection.config
            active = []
            for key, mapping in self.joints[name].items():
                value = state.get(mapping["state"], {}).get("value")
                if type(value) in (float, int) and np.isfinite(value):
                    active.append(key)
            self.sync_geometry(name)
            self.visual_evidence[name]["joints"] = config
            self.visual_evidence[name]["physics"] = self.physics[name].evidence
            for display, panel in self.panels[name].items():
                captured, mode = capture(instrument, display, panel["spec"])
                path = Path(captured["path"]).resolve()
                if not path.is_relative_to(instrument.capture_directory.name):
                    raise RuntimeError("Unowned native panel texture")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != captured["sha256"]:
                    raise RuntimeError("Native panel texture checksum differs")
                panel["texture"].Set(str(path))
                if panel["texture"].Get().path != str(path):
                    raise RuntimeError("Isaac texture readback differs")
                self.visual_evidence[name]["panels"][display] = {
                    "sha256": digest,
                    "width": captured["width"],
                    "height": captured["height"],
                    "bound": True,
                    "mode": mode,
                    "placement": panel["placement"],
                }
            lamps, visible = [], []
            colors = projection.colors()
            for key, color in colors.items():
                surfaces = self.colors[name].get(key, [])
                for surface in surfaces:
                    surface.GetInput("diffuseColor").DisconnectSource()
                    surface.GetInput("diffuseColor").Set(self.Gf.Vec3f(*color))
                if surfaces:
                    lamps.append(key)
                    self.visual_evidence[name]["lamps"][key] = color
            for key, (slot, attribute) in self.properties[name].items():
                if slot in state:
                    attribute.Set(json.dumps(state[slot]["value"], sort_keys=True))
                    lamps.append(key)
            for key, (slot, bindings) in self.lamps[name].items():
                if key in colors:
                    continue
                value = state.get(slot, {}).get("value")
                if type(value) is bool:
                    for surface, light in bindings:
                        color = surface.GetInput("diffuseColor").Get()
                        surface.GetInput("emissiveColor").Set(color * (2.0 if value else 0.0))
                        light.GetIntensityAttr().Set(
                            2000.0 if value and slot == "lamp.lit" else 0.0
                        )
                    lamps.append(key)
                    self.visual_evidence[name]["lamps"][key] = value
            native_panels, readout_keys = set(), set()
            for display, panel in self.panels[name].items():
                mode = self.visual_evidence[name]["panels"][display]["mode"]
                if mode == "native" and panel["placement"] == "instrument":
                    native_panels.add(panel["component"])
                elif not mode.startswith("unavailable:"):
                    readout_keys.update(readout_bindings(instrument, panel["spec"]))
            report = binding_report(
                instrument,
                set(active) | projection.binding_keys(),
                native_panels,
                lamps,
                visible,
                readout_keys,
            )
            report["panels"] = sorted(self.panels[name])
            report["panel_failures"] = self.panel_failures[name]
            self.mapping_report[name] = report
        self.sim.render()

    def step(self):
        for _ in range(10):
            self.sim.step(render=False)
        self.sync()

    def close(self):
        self.textures.cleanup()
        if self.app is not None:
            error = sys.exc_info()[1]
            if error is not None:
                traceback.print_exc()
            self.app.close(exit_code=1 if error is not None else 0)
            self.app = None
