"""Compose pinned SIMPLE assets with native household mechanisms, without depot writes."""

import hashlib
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, PhysxSchema

from .assets import generated_directory
from .household_models import SURFACES
from . import household_usd
from .simple_household import FURNITURE, SCANS, SURFACE_ASSETS, TEXTURES, model


def _folder(assets):
    identity = hashlib.sha256((str(assets.resolve()) + repr((SCANS, FURNITURE, TEXTURES))).encode()
                              + Path(__file__).read_bytes()).hexdigest()[:16]
    folder = generated_directory() / "simple_household" / identity
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _mesh_cube(stage, prim):
    """Give an existing box explicit face UVs without changing its collision solid."""
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        original = prim
        visual_path = str(prim.GetPath()) + "_visual"
        Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), visual_path)
        prim = stage.GetPrimAtPath(visual_path)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        UsdGeom.Imageable(original).CreateVisibilityAttr("invisible")
    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
    points = [(x * size / 2, y * size / 2, z * size / 2)
              for x, y, z in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                              (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
    prim.SetTypeName("Mesh")
    mesh = UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set([(0,0),(1,0),(1,1),(0,1)] * 6)


def _textures(stage, root, assets):
    for name, number in TEXTURES.items():
        surface = UsdShade.Shader.Get(stage, f"{root}/Materials/{name}/Surface")
        if not surface:
            continue
        shader = UsdShade.Shader.Define(stage, f"{root}/Materials/{name}/Texture")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(
            assets / "scenes/hssd/102344022/textures" / f"102344022_texture{number}.png"))
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        reader = UsdShade.Shader.Define(stage, f"{root}/Materials/{name}/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        surface.GetInput("diffuseColor").ConnectToSource(shader.ConnectableAPI(), "rgb")
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        if prim.IsA(UsdGeom.Cube):
            _mesh_cube(stage, prim)


def _copy_materials(source, target, old_root, new_root):
    looks = source.GetPrimAtPath(old_root + "/Looks")
    if looks:
        Sdf.CopySpec(source.GetRootLayer(), looks.GetPath(), target.GetRootLayer(), new_root + "/Looks")
    for prim in target.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if any(str(p).startswith(old_root + "/Looks/") for p in targets):
                rel.SetTargets([Sdf.Path(str(p).replace(old_root + "/Looks/", new_root + "/Looks/", 1)) for p in targets])


def _scan(stage, name, assets):
    number, degrees, size = SCANS[name]
    source_path = assets / "scenes/graspnet" / f"{number:03d}_ruled.usd"
    source = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(source_path)))
    UsdUtils.ModifyAssetPaths(source.GetRootLayer(),
        lambda a: str((source_path.parent / a).resolve()) if a.startswith("./") else a)
    visual = next(p for p in source.Traverse() if p.IsA(UsdGeom.Mesh) and p.GetName() == "visual")
    target_path = "/Asset/body/scan"
    Sdf.CopySpec(source.GetRootLayer(), visual.GetPath(), stage.GetRootLayer(), target_path)
    _copy_materials(source, stage, str(source.GetDefaultPrim().GetPath()), "/Asset")
    mesh = UsdGeom.Mesh.Get(stage, target_path)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle),-math.sin(angle),0],
                         [math.sin(angle),math.cos(angle),0],[0,0,1]])
    points = points @ rotation.T
    lo, hi = points.min(0), points.max(0)
    points = (points - np.array([*(lo[:2] + hi[:2]) / 2, lo[2]])) * (np.array(size) / (hi-lo))
    mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, p)) for p in (points.min(0),points.max(0))])
    # Recompute shading after the axis/scale conversion. Old authored normals
    # are in the source frame; leaving them produces incorrect illumination.
    mesh.GetNormalsAttr().Clear()
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(mesh).ClearXformOpOrder()
    for prim in Usd.PrimRange(mesh.GetPrim()):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAPI(UsdPhysics.MassAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    # Native contacts use the calibrated compound solids. Imported render-only
    # geometry must never introduce a second body or seal the bowl's opening.
    for prim in stage.GetPrimAtPath("/Asset/body").GetChildren():
        if prim.GetName() != "scan" and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")


def make_asset(name, assets):
    assets = Path(assets)
    definition = model(name)
    path = _folder(assets) / (name + "-" + hashlib.sha256(repr(definition).encode()).hexdigest()[:12] + ".usdc")
    if path.is_file():
        return path
    source = household_usd.make_asset(name, definition)
    layer = Sdf.Layer.OpenAsAnonymous(str(source))
    stage = Usd.Stage.Open(layer)
    _textures(stage, "/Asset", assets)
    if name in SCANS:
        _scan(stage, name, assets)
    layer.Export(str(path))
    return path


def _furniture_asset(key, assets):
    folder = _folder(assets)
    path = folder / f"furniture-{key}.usdc"
    if path.is_file():
        return path
    from .scenes import load_scene, house_usd

    scene = load_scene("kitchen")
    scene.asset_directory = assets / "scenes"
    source = Usd.Stage.Open(str(house_usd(scene)))
    while instances := [p for p in Usd.PrimRange(source.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                        if p.IsInstance()]:
        for prim in instances:
            prim.SetInstanceable(False)
    # Flattening resolves references and inherited materials before selecting
    # the actual furniture subtree. Nothing is written to the source depot.
    source = Usd.Stage.Open(source.Flatten())
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    geom = UsdGeom.Xform.Define(stage, "/Asset/Geometry")
    # The selected furniture subtree is already Z-up. Its omitted HSSD
    # ancestor carried -90 degrees X for the whole-house Y-up convention.
    source_root = str(source.GetDefaultPrim().GetPath())
    Sdf.CopySpec(source.GetRootLayer(), source_root + "/furniture/" + FURNITURE[key],
                 stage.GetRootLayer(), "/Asset/Geometry/Mesh")
    _copy_materials(source, stage, source_root, "/Asset")
    # Furniture meshes are static triangle colliders, including open leg space.
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    if not np.all(np.isfinite(hi - lo)) or np.any(hi <= lo) or np.max(hi-lo) > 10:
        raise ValueError(f"Invalid imported furniture bounds: {key}")
    # Center the selected furniture at floor level.
    geom.ClearXformOpOrder()
    geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, -np.array([*(lo[:2]+hi[:2])/2, lo[2]]))))
    stage.GetRootLayer().Save()
    return path


def room(stage, assets):
    assets = Path(assets)
    root = household_usd.room(stage)
    _textures(stage, root, assets)
    # The task layout and portal stay fixed; imported furniture replaces the
    # corresponding authored solid, never an invisible duplicate support.
    placements = {n: (key, yaw, (*SURFACES[n][0], 0.), (*SURFACES[n][1], SURFACES[n][2]))
                  for n, (key, yaw) in SURFACE_ASSETS.items()}
    placements["sofa"] = ("sofa", 0, (0., 0., 0.), (2.15, 0.85, 0.93))
    for name, (key, yaw, pos, size) in placements.items():
        path = _furniture_asset(key, assets)
        source = Usd.Stage.Open(str(path))
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(source.GetDefaultPrim()).ComputeAlignedRange()
        extent = np.array(bound.GetMax()) - np.array(bound.GetMin())
        if yaw == 90:
            extent = extent[[1,0,2]]
        target = f"{root}/{name}"
        stage.RemovePrim(target)
        outer = UsdGeom.Xform.Define(stage, target)
        outer.AddTranslateOp().Set(Gf.Vec3d(*pos))
        outer.AddScaleOp().Set(Gf.Vec3f(*map(float, np.array(size)/extent)))
        mesh = UsdGeom.Xform.Define(stage, target + "/furniture")
        mesh.GetPrim().GetReferences().AddReference(str(path))
        mesh.AddRotateZOp().Set(float(yaw))
        for prim in Usd.PrimRange(outer.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    UsdShade.Material.Get(stage, root + "/Materials/physics"), materialPurpose="physics")
    return root
