"""Detailed native mechanisms and task fixtures for the complete HSSD house.

Render meshes follow the existing collision parts and articulated links.
Small edge radii soften manufactured edges; cavities remain open. Source
depots are immutable and generated USD lives in the owner-local directory.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, PhysxSchema

from . import household_usd
from .assets import generated_directory
from .household_models import box, model
from .asset_bindings import SCANS
from .imported_assets import _scan, _textures, _furniture_asset


def mesh(stage, path, points, faces, material, normals=None, uv=None):
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr("none")
    if normals is not None:
        shape.CreateNormalsAttr([Gf.Vec3f(*map(float, n)) for n in normals])
        shape.SetNormalsInterpolation("vertex")
    if uv is not None:
        UsdGeom.PrimvarsAPI(shape).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                               "vertex").Set(uv)
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    return shape


def rounded_box(stage, path, part, material):
    """A rounded cuboid within the collider, with at most 2 mm edge deviation."""
    half = np.asarray(part.size) / 2
    radius = min(.002, float(half.min()) * .22)
    core = half - radius
    points, normals, uv, faces = [], [], [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        # Concentrate vertices at the edge radius instead of subdividing flats.
        coords = [np.r_[-half[k], -core[k] - radius * np.array([.8, .4]),
                        -core[k], core[k], core[k] + radius * np.array([.4, .8]), half[k]]
                  for k in (u, v)]
        for sign in (-1, 1):
            offset = len(points)
            for a in coords[0]:
                for b in coords[1]:
                    p = np.zeros(3)
                    p[axis], p[u], p[v] = sign * half[axis], a, b
                    nearest = np.clip(p, -core, core)
                    normal = p - nearest
                    normal /= np.linalg.norm(normal)
                    points.append(nearest + radius * normal)
                    normals.append(normal)
                    uv.append((float(a * 3), float(b * 3)))
            n = len(coords[1])
            for i in range(len(coords[0]) - 1):
                for j in range(n - 1):
                    q = [offset + i*n+j, offset+(i+1)*n+j,
                         offset+(i+1)*n+j+1, offset+i*n+j+1]
                    faces.append(q if sign > 0 else q[::-1])
    shape = mesh(stage, path, points, faces, material, normals, uv)
    household_usd.transform(shape.GetPrim(), part.pos, part.rpy)


def lathe(stage, path, profile, material, pos=(0, 0, 0), rpy=(0, 0, 0)):
    """Closed radial section, including an open interior when the profile has one."""
    count = 96
    points, uv, faces = [], [], []
    for j, (radius, z) in enumerate(profile):
        for i in range(count + 1):
            a = 2 * math.pi * i / count
            points.append((radius * math.cos(a), radius * math.sin(a), z))
            uv.append((i / count, j / (len(profile) - 1)))
    for j in range(len(profile) - 1):
        for i in range(count):
            a = j * (count + 1) + i
            faces.append([a, a + 1, a + count + 2, a + count + 1])
    shape = mesh(stage, path, points, faces, material, uv=uv)
    household_usd.transform(shape.GetPrim(), pos, rpy)
    return shape


def _finish(stage, definition):
    for link in definition.links:
        root = f"/Asset/{link.name}"
        rings = {}
        for part in link.parts:
            prefix = part.name.rsplit("_", 1)[0]
            if part.shape == "box" and prefix in ("rim", "wall"):
                rings.setdefault(prefix, []).append(part)
                continue
            material = UsdShade.Material.Get(stage, f"/Asset/Materials/{part.material}")
            prim = stage.GetPrimAtPath(f"{root}/{part.name}")
            if part.shape == "box":
                UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
                rounded_box(stage, f"{root}/{part.name}_finish", part, material)
            elif part.shape == "cylinder":
                UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
                radius, height = part.size
                edge = min(.001, height * .10, radius * .1)
                profile = [(0, -height/2), (radius-edge, -height/2),
                           (radius, -height/2+edge), (radius, height/2-edge),
                           (radius-edge, height/2), (0, height/2)]
                lathe(stage, f"{root}/{part.name}_finish", profile, material, part.pos, part.rpy)
        for prefix, parts in rings.items():
            first = parts[0]
            radius = math.hypot(*first.pos[:2])
            thickness, _, height = first.size
            inner, outer = radius-thickness/2, radius+thickness/2
            bottom, top = first.pos[2]-height/2, first.pos[2]+height/2
            edge = min(.001, thickness/5)
            profile = [(inner, bottom), (outer-edge, bottom), (outer, bottom+edge),
                       (outer, top-edge), (outer-edge, top), (inner+edge, top),
                       (inner, top-edge), (inner, bottom)]
            for part in parts:
                UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/{part.name}")).CreateVisibilityAttr("invisible")
            lathe(stage, f"{root}/{prefix}_finish", profile,
                  UsdShade.Material.Get(stage, f"/Asset/Materials/{first.material}"))
        handles = [p for p in link.parts if p.name.startswith("handle_") and p.shape == "sphere"]
        if len(handles) == 40:
            points, normals, faces = [], [], []
            for i in range(96):
                a = 2 * math.pi * i / 96
                center = np.array([.070 + .032*math.cos(a), 0, .060 + .039*math.sin(a)])
                radial = np.array([math.cos(a)/.032, 0, math.sin(a)/.039])
                radial /= np.linalg.norm(radial)
                for j in range(16):
                    b = 2*math.pi*j/16
                    normal = math.cos(b)*radial + math.sin(b)*np.array([0, 1, 0])
                    points.append(center + .008*normal)
                    normals.append(normal)
                    faces.append([i*16+j, ((i+1)%96)*16+j,
                                  ((i+1)%96)*16+(j+1)%16, i*16+(j+1)%16])
            for part in handles:
                UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/{part.name}")).CreateVisibilityAttr("invisible")
            mesh(stage, root + "/handle_finish", points, faces,
                 UsdShade.Material.Get(stage, "/Asset/Materials/ceramic"), normals)


def make_asset(name, assets):
    assets = Path(assets)
    definition = model(name)
    identity = hashlib.sha256(Path(__file__).read_bytes() + repr(definition).encode()
                              + str(assets.resolve()).encode()).hexdigest()[:16]
    folder = generated_directory() / "hssd_household" / identity
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.usdc"
    if path.exists():
        return path
    layer = Sdf.Layer.OpenAsAnonymous(str(household_usd.make_asset(name, definition)))
    stage = Usd.Stage.Open(layer)
    if name == "balance_scale":
        # The light beam has a deliberately weak gravitational restoring torque.
        # Default articulation sleeping can freeze a balanced load while tilted.
        # Keep this passive mechanism awake until its physical motion settles.
        physics = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetDefaultPrim())
        physics.CreateSleepThresholdAttr(1e-8)
        physics.CreateStabilizationThresholdAttr(0.0)
    if name not in SCANS:
        _finish(stage, definition)
    _textures(stage, "/Asset", assets)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Asset/Materials")):
        shader = UsdShade.Shader(prim)
        if shader and shader.GetIdAttr().Get() == "UsdUVTexture":
            # Rounded surfaces use metric UVs, including negative coordinates.
            # The USD default border mode would turn those faces black.
            for axis in ("wrapS", "wrapT"):
                shader.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
    # Match manufactured ceramic and coated metal to the source house palette.
    for key, color, roughness in (("ceramic", (.88, .87, .83), .19),
                                  ("teal", (.16, .24, .23), .38),
                                  ("metal", (.52, .54, .56), .24)):
        shader = UsdShade.Shader.Get(stage, f"/Asset/Materials/{key}/Surface")
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
        shader.GetInput("roughness").Set(roughness)
    if name in SCANS:
        _scan(stage, name, assets)
    layer.Export(str(path))
    return path


def island_colliders(stage, scene, material_path):
    """Bake the mirrored island into outward-wound world-space collision meshes.

    PhysX triangle meshes must not inherit the source's negative scale. Keep
    the render meshes and their normals/UVs untouched, and preserve every
    collision vertex (including the thin countertop and the cabinet below).
    """
    relative = next(path for path, entity in scene.collider_bindings.items()
                    if entity == "kitchen_island")
    root = stage.GetPrimAtPath("/World/Scene/" + relative)
    sources = [UsdGeom.Mesh(p) for p in Usd.PrimRange(root)
               if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
    collision_root = UsdGeom.Xform.Define(stage, str(root.GetPath()) + "/Collision")
    collision_root.SetResetXformStack(True)
    material = UsdShade.Material.Get(stage, material_path)
    for index, source in enumerate(sources):
        transform = np.asarray(source.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        points = np.asarray(source.GetPointsAttr().Get())
        points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        counts = list(source.GetFaceVertexCountsAttr().Get())
        indices = list(source.GetFaceVertexIndicesAttr().Get())
        reflected = np.linalg.det(transform[:3, :3]) < 0
        if reflected != (source.GetOrientationAttr().Get() == "leftHanded"):
            offset = 0
            for count in counts:
                indices[offset:offset + count] = reversed(indices[offset:offset + count])
                offset += count
        target = UsdGeom.Mesh.Define(stage, str(collision_root.GetPath()) + f"/mesh_{index}")
        target.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
        target.CreateFaceVertexCountsAttr(counts)
        target.CreateFaceVertexIndicesAttr(indices)
        target.CreateSubdivisionSchemeAttr("none")
        target.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(source.GetPrim()).CreateCollisionEnabledAttr(False)
        UsdPhysics.CollisionAPI.Apply(target.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(target.GetPrim()).CreateApproximationAttr("none")
        UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(material, materialPurpose="physics")


def fixtures(stage, scene):
    """Small functional additions; the house and original furniture stay intact."""
    root = "/World/Household"
    UsdGeom.Xform.Define(stage, root)
    household_usd.materials(stage, root)
    for name in ("parcel_console", "entry_console"):
        surface = scene.surfaces[name]
        xf = UsdGeom.Xform.Define(stage, root + "/" + name)
        xf.GetPrim().GetReferences().AddReference(str(_furniture_asset("console", scene.asset_directory.parent)))
        xf.AddTranslateOp().Set(Gf.Vec3d(*surface.center, 0.0))
        xf.AddRotateZOp().Set(math.degrees(surface.yaw))
        for prim in Usd.PrimRange(xf.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    UsdShade.Material.Get(stage, root + "/Materials/physics"), materialPurpose="physics")
    for name, definition in scene.portal["frame"].items():
        household_usd.part(stage, f"{root}/door_frame/{name}",
                           box(name, definition["size"], definition["pos"], "oak_light"), root)
    shelf = scene.surfaces["book_shelf"]
    cx, cy = shelf.center
    parts = [box("back", (.9, .025, 1.65), (cx, cy+.16, .825), "walnut")]
    parts += [box(f"upright_{i}", (.025, .34, 1.65), (cx+side*.4375, cy, .825), "oak")
              for i, side in enumerate((-1, 1))]
    parts += [box(f"shelf_{i}", (.85, .34, .025), (cx, cy, z-.0125), "oak_light")
              for i, z in enumerate((.12, shelf.height, 1.18, 1.65))]
    UsdGeom.Xform.Define(stage, root + "/book_shelf")
    for part in parts:
        household_usd.part(stage, root + "/book_shelf/" + part.name, part, root)
    bay = scene.cart_bay
    sx, sy = bay["size"]
    bay_root = UsdGeom.Xform.Define(stage, root + "/cart_bay")
    bay_root.AddTranslateOp().Set(Gf.Vec3d(*bay["center"], 0.))
    bay_root.AddRotateZOp().Set(math.degrees(bay["yaw"]))
    # Three low rails leave the room-facing edge open for the rolling cart.
    for name, size, pos in (
        ("left", (.018, sy+.036, .018), (-sx/2-.009, 0, .009)),
        ("right", (.018, sy+.036, .018), (sx/2+.009, 0, .009)),
        ("back", (sx, .018, .018), (0, -sy/2-.009, .009)),
    ):
        household_usd.part(stage, f"{root}/cart_bay/{name}", box(name, size, pos, "metal"), root)
    _textures(stage, root, scene.asset_directory.parent)
