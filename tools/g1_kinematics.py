"""Extract G1 component chains from USD physics joints, without starting Kit."""

import math


def extract(stage):
    from pxr import Gf, UsdGeom, UsdPhysics
    from roboenv_runtime.adapters.isaaclab.geometry import rotation

    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("G1 USD must use metres")
    parents, moving = {}, {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) != 1 or len(b) != 1:
            continue  # A root/world joint is not part of a component chain.
        if str(b[0]) in parents:
            raise ValueError("G1 needs an acyclic tree with unique child bodies")
        parents[str(b[0])] = (str(a[0]), joint)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            moving[prim.GetName()] = UsdPhysics.RevoluteJoint(prim)

    def metadata(name):
        joint = moving[name]
        low = math.radians(joint.GetLowerLimitAttr().Get())
        high = math.radians(joint.GetUpperLimitAttr().Get())
        if not math.isfinite(low) or not math.isfinite(high) or not low < high:
            raise ValueError(f"G1 requires finite joint limits: {name}")
        return {
            "name": name,
            "type": "revolute",
            "limits": {"min": low, "max": high},
            "max_velocity": None,
            "max_effort": None,
        }

    def local(joint, side):
        p = getattr(joint, f"GetLocalPos{side}Attr")().Get()
        q = getattr(joint, f"GetLocalRot{side}Attr")().Get().GetNormalized()
        return [*map(float, p), *map(float, q.GetImaginary()), float(q.GetReal())]

    def link(name):
        paths = {
            path
            for child, (parent, _) in parents.items()
            for path in (child, parent)
            if path.rsplit("/", 1)[-1] == name
        }
        if len(paths) != 1:
            raise ValueError(f"G1 requires one body named {name}")
        return paths.pop()

    def chain(base, tip):
        current, root = link(tip), link(base)
        selected, seen = [], set()
        while current != root:
            if current in seen or current not in parents:
                raise ValueError("Component is not a descendant of its base")
            seen.add(current)
            current, joint = parents[current]
            selected.append(joint)
        links, joints = [], []
        for joint in reversed(selected):
            parent, child = local(joint, 0), local(joint, 1)
            item = {"origin": parent}
            prim = joint.GetPrim()
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()
                axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                item.update(
                    joint=len(joints),
                    axis=[float(axis == a) for a in "XYZ"],
                    anchor=[0, 0, 0],
                    reference=0,
                )
                joints.append(metadata(name))
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError("Only fixed and revolute G1 joints are supported")
            links.append(item)
            p = (-rotation(child[3:]).T @ child[:3]).tolist()
            links.append({"origin": [*p, *[-v for v in child[3:6]], child[6]]})
        return {
            "base": base,
            "tip": tip,
            "joints": joints,
            "chain": links,
            "home": {joint["name"]: 0.0 for joint in joints},
        }

    parts = {"waist": chain("pelvis", "torso_link")}
    for side in ("left", "right"):
        parts[f"{side}_arm"] = chain("torso_link", f"{side}_wrist_yaw_link")
        joints = [metadata(name) for name in sorted(moving) if name.startswith(side + "_hand_")]
        if len(joints) != 7:
            raise ValueError("Expected seven joints in each G1 TriHand")
        parent, joint = parents[link(f"{side}_hand_palm_link")]
        if parent != link(f"{side}_wrist_yaw_link") or not joint.GetPrim().IsA(
            UsdPhysics.FixedJoint
        ):
            raise ValueError("Expected a fixed wrist-to-palm mount")
        transforms = []
        for index in (0, 1):
            value = local(joint, index)
            transforms.append(
                Gf.Matrix4d(
                    Gf.Rotation(Gf.Quatd(value[6], Gf.Vec3d(*value[3:6]))),
                    Gf.Vec3d(*value[:3]),
                )
            )
        # Gf matrices use row vectors: child^-1 followed by parent.
        mount = transforms[1].GetInverse() * transforms[0]
        quaternion = mount.ExtractRotationQuat().GetNormalized()
        parts[f"{side}_hand"] = {
            "joints": joints,
            "home": {joint["name"]: 0.0 for joint in joints},
            "mount": [
                *mount.ExtractTranslation(),
                *quaternion.GetImaginary(),
                quaternion.GetReal(),
            ],
        }
    if len(parts["waist"]["joints"]) != 3 or any(
        len(parts[side + "_arm"]["joints"]) != 7 for side in ("left", "right")
    ):
        raise ValueError("Expected three waist and seven joints per arm")
    owned = [joint["name"] for part in parts.values() for joint in part["joints"]]
    if len(set(owned)) != len(owned):
        raise ValueError("Component joints overlap")
    return parts
