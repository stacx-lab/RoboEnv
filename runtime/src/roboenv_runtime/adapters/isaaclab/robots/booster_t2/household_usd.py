"""Build the household's reusable USD assets and furnished room locally."""

import hashlib
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

from .household_models import (
    CART_BAY_INTERIOR,
    PALETTE,
    SURFACES,
    box,
    cylinder,
    model,
    sphere,
    table_parts,
)
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


def room(stage):
    """Connected living/dining, kitchen and entry, with actual support/obstacles."""
    root = "/World/Household"
    UsdGeom.Xform.Define(stage, root)
    materials(stage, root)

    def solid(name, p):
        return part(stage, f"{root}/{name}", p, root)

    solid("floor", box("floor", (12.8, 9.0, 0.10), (1.2, 0, -0.05), "oak_light"))
    # Narrow boards and staggered seams, above the single physical floor.
    for row in range(36):
        for col in range(7):
            x = -5.1 + col * 1.85 + (row % 2) * 0.42
            if x + 1.8 > 7.55:
                continue
            solid(
                f"floor_board_{row}_{col}",
                box(
                    "board",
                    (1.84, 0.247, 0.001),
                    (x + 0.92, -4.375 + row * 0.25, 0.0005),
                    "oak_light" if (row + col) % 3 else "oak",
                    collision=False,
                ),
            )
    for name, size, pos in (
        ("north", (12.8, 0.12, 2.8), (1.2, 4.5, 1.4)),
        ("south", (12.8, 0.12, 2.8), (1.2, -4.5, 1.4)),
        ("west", (0.12, 9, 2.8), (-5.2, 0, 1.4)),
        ("east", (0.12, 9, 2.8), (7.6, 0, 1.4)),
        ("portal_south", (0.12, 3.7, 2.8), (5.0, -2.65, 1.4)),
        ("portal_north", (0.12, 3.75, 2.8), (5.0, 2.625, 1.4)),
        ("portal_lintel", (0.12, 1.55, 0.55), (5.0, -0.025, 2.525)),
    ):
        solid(name, box(name, size, pos, "cream"))
    for name, (xy, size, height) in SURFACES.items():
        for p in table_parts(size, height, "oak_light"):
            p.pos = (p.pos[0] + xy[0], p.pos[1] + xy[1], p.pos[2])
            solid(f"{name}/{p.name}", p)
    # Bookcase around its accessible shelf, with space above the task book.
    for name, size, pos in (
        ("back", (1.35, 0.035, 1.75), (0, 3.9, 0.875)),
        ("left", (0.04, 0.55, 1.75), (-0.67, 3.64, 0.875)),
        ("right", (0.04, 0.55, 1.75), (0.67, 3.64, 0.875)),
        ("upper", (1.35, 0.55, 0.035), (0, 3.64, 1.55)),
    ):
        solid(f"bookcase/{name}", box(name, size, pos, "walnut"))
    for i in range(7):
        solid(
            f"bookcase/decorative_book_{i}",
            box(
                "book",
                (0.065, 0.18, 0.23),
                (-0.48 + i * 0.095, 3.70, 1.68),
                "teal" if i % 2 else "red",
            ),
        )
    # Sofa: a physical obstruction, not a camera backdrop.
    for name, size, pos, mat in (
        ("base", (2.15, 0.85, 0.22), (0, 0, 0.27), "walnut"),
        ("back", (2.15, 0.18, 0.62), (0, 0.35, 0.62), "fabric"),
        ("arm_l", (0.19, 0.85, 0.50), (-1.0, 0, 0.46), "fabric"),
        ("arm_r", (0.19, 0.85, 0.50), (1.0, 0, 0.46), "fabric"),
        ("seat_l", (0.84, 0.69, 0.17), (-0.45, -0.04, 0.45), "fabric"),
        ("seat_r", (0.84, 0.69, 0.17), (0.45, -0.04, 0.45), "fabric"),
    ):
        solid(f"sofa/{name}", box(name, size, pos, mat))
    for i, x in enumerate((-0.55, 0.55)):
        solid(
            f"sofa/cushion_{i}",
            box(
                "cushion",
                (0.38, 0.13, 0.36),
                (x, 0.21, 0.73),
                "cream",
                (0.15, 0, 0.1 if i else -0.1),
            ),
        )
    solid("rug", box("rug", (2.8, 1.7, 0.004), (0, -1.0, 0.002), "teal", collision=False))
    # A real parking bay opens northward into the room.
    (left, back), (right, front) = CART_BAY_INTERIOR
    thickness = 0.05
    for name, size, pos in (
        (
            "left",
            (thickness, front - back, 0.55),
            (left - thickness / 2, (back + front) / 2, 0.275),
        ),
        (
            "right",
            (thickness, front - back, 0.55),
            (right + thickness / 2, (back + front) / 2, 0.275),
        ),
        (
            "back",
            (right - left + 2 * thickness, thickness, 0.55),
            ((left + right) / 2, back - thickness / 2, 0.275),
        ),
    ):
        solid(f"cart_bay/{name}", box(name, size, pos, "walnut"))
    # Wall panels, skirting and framed artwork make the room legible at head height.
    for i, x in enumerate((-3.8, -1.0, 2.0, 4.0)):
        solid(f"art/frame_{i}", box("frame", (0.82, 0.035, 0.65), (x, 4.42, 1.75), "walnut"))
        solid(
            f"art/picture_{i}",
            box(
                "picture",
                (0.73, 0.008, 0.56),
                (x, 4.395, 1.75),
                "teal" if i % 2 else "oak_light",
                collision=False,
            ),
        )
        solid(
            f"art/motif_{i}", sphere("motif", 0.20, (x, 4.38, 1.76), "paper", (1, 0.03, 0.7), False)
        )
    for x in (-5.12, 7.52):
        solid(
            f"skirting_{str(x).replace('-', 'n').replace('.', '_')}",
            box("trim", (0.025, 9.0, 0.11), (x, 0, 0.055), "paper"),
        )
    for i, (x, y) in enumerate(((-4.35, 3.65), (4.35, 3.65), (6.9, -2.5))):
        solid(f"plant_{i}/pot", cylinder("pot", 0.16, 0.27, (x, y, 0.135), "ceramic"))
        for k in range(9):
            a = 2 * math.pi * k / 9
            solid(
                f"plant_{i}/leaf_{k}",
                sphere(
                    "leaf",
                    0.12,
                    (x + 0.12 * math.cos(a), y + 0.12 * math.sin(a), 0.45 + 0.04 * (k % 3)),
                    "green",
                    (0.8, 0.6, 2.1),
                    False,
                ),
            )
    for i, (x, y) in enumerate(((-2, 2), (2, 2), (-2, -2.5), (2, -2.3), (6.5, 0))):
        light = UsdLux.RectLight.Define(stage, f"{root}/Lighting/key_{i}")
        transform(light.GetPrim(), (x, y, 2.65))
        light.CreateWidthAttr(1.3)
        light.CreateHeightAttr(1.0)
        light.CreateIntensityAttr(250.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.79))
    return root
