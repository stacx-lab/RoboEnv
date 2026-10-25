"""Build USD collision bodies, passive joints and reading lights for the household."""

import hashlib
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

from .household_models import PALETTE, model
from .assets import generated_directory


def transform(prim, pos=(0, 0, 0), rpy=(0, 0, 0)):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if any(rpy):
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*[math.degrees(v) for v in rpy]))


def materials(stage, root):
    for name, (color, roughness, metallic) in PALETTE.items():
        material = UsdShade.Material.Define(stage, f"{root}/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    phys = UsdShade.Material.Define(stage, f"{root}/Materials/physics")
    api = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    api.CreateStaticFrictionAttr(0.65)
    api.CreateDynamicFrictionAttr(0.5)
    api.CreateRestitutionAttr(0.0)


def part(stage, path, definition, material_root):
    p = definition
    if p.shape == "box":
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        transform(shape.GetPrim(), p.pos, p.rpy)
        UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*p.size))
    elif p.shape == "cylinder":
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(p.size[0])
        shape.CreateHeightAttr(p.size[1])
        shape.CreateAxisAttr("Z")
        transform(shape.GetPrim(), p.pos, p.rpy)
    elif p.shape == "hull":
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr([Gf.Vec3f(*v) for v in p.size])
        shape.CreateFaceVertexCountsAttr([4] * 6)
        shape.CreateFaceVertexIndicesAttr([0,1,3,2,4,6,7,5,0,4,5,1,2,3,7,6,0,2,6,4,1,5,7,3])
        shape.CreateSubdivisionSchemeAttr("none")
        transform(shape.GetPrim(), p.pos, p.rpy)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    else:
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(p.size[0])
        transform(shape.GetPrim(), p.pos, p.rpy)
        UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*p.size[1:]))
    prim = shape.GetPrim()
    if p.collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(0.001)
        api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material.Get(stage, f"{material_root}/Materials/physics"),
            materialPurpose="physics",
        )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, f"{material_root}/Materials/{p.material}")
    )
    shape.CreateDisplayColorAttr([Gf.Vec3f(*PALETTE[p.material][0])])
    return prim


def make_asset(name, definition=None):
    """Write a source-defined asset, never changing downloaded depots."""
    m = definition if definition is not None else model(name)
    identity = hashlib.sha256(
        Path(__file__).read_bytes() + repr((m, PALETTE)).encode()
    ).hexdigest()[:16]
    folder = generated_directory() / "household" / identity
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.usda"
    if target.is_file():
        return target
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    materials(stage, "/Asset")
    for link in m.links:
        path = f"/Asset/{link.name}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        transform(prim, link.pos)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(link.mass)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        api.CreateMaxDepenetrationVelocityAttr(1.0)
        api.CreateLinearDampingAttr(link.linear_damping)
        api.CreateAngularDampingAttr(link.angular_damping)
        api.CreateEnableCCDAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        for p in link.parts:
            part(stage, f"{path}/{p.name}", p, "/Asset")
    if m.joints:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
        if m.fixed:
            fixed = UsdPhysics.FixedJoint.Define(stage, "/Asset/anchor")
            fixed.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/{m.links[0].name}")])
        positions = {link.name: np.asarray(link.pos) for link in m.links}
        for j in m.joints:
            cls = UsdPhysics.RevoluteJoint if j.kind == "revolute" else UsdPhysics.PrismaticJoint
            joint = cls.Define(stage, f"/Asset/joints/{j.name}")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Asset/{j.parent}")])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/{j.child}")])
            joint.CreateAxisAttr(j.axis)
            joint.CreateLocalPos0Attr(
                Gf.Vec3f(*map(float, np.asarray(j.anchor) - positions[j.parent]))
            )
            joint.CreateLocalPos1Attr(
                Gf.Vec3f(*map(float, np.asarray(j.anchor) - positions[j.child]))
            )
            factor = 180 / math.pi if j.kind == "revolute" else 1.0
            joint.CreateLowerLimitAttr(j.limits[0] * factor)
            joint.CreateUpperLimitAttr(j.limits[1] * factor)
            joint.CreateCollisionEnabledAttr(False)
            api = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            api.CreateJointFrictionAttr(j.friction)
            drive = UsdPhysics.DriveAPI.Apply(
                joint.GetPrim(), "angular" if j.kind == "revolute" else "linear"
            )
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(j.damping)
    stage.GetRootLayer().Save()
    return target


def reading_light(stage, name):
    # Fabric moves articulation links without updating their USD transforms.
    # A world-space light is synchronized explicitly from the native head pose.
    UsdGeom.Xform.Define(stage, "/World/TaskLights")
    light = UsdLux.DiskLight.Define(stage, f"/World/TaskLights/{name}")
    light.CreateRadiusAttr(0.06)
    light.CreateNormalizeAttr(True)
    light.CreateIntensityAttr(0.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.84, 0.61))
    shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr(42.0)
    shaping.CreateShapingConeSoftnessAttr(0.3)
    xf = UsdGeom.Xformable(light)
    return light, xf.AddTranslateOp(), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
