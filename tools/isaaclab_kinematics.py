"""Extract scalar-joint chains from the pinned USD's physics joint frames."""


def extract_chains(stage, joint_names, feet):
    from pxr import UsdPhysics
    from roboenv_runtime.adapters.isaaclab.geometry import rotation

    def local(joint, side):
        position = getattr(joint, f"GetLocalPos{side}Attr")().Get()
        quaternion = getattr(joint, f"GetLocalRot{side}Attr")().Get().GetNormalized()
        return [*map(float, position), *map(float, quaternion.GetImaginary()),
                float(quaternion.GetReal())]

    parents = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if len(bodies[0]) != 1 or len(bodies[1]) != 1:
            raise ValueError("Go2 joints require one parent and one child body")
        child = str(bodies[1][0])
        if child in parents:
            raise ValueError("Go2 joints require unique child bodies")
        parents[child] = (str(bodies[0][0]), joint)
    indexes = {name: i for i, name in enumerate(joint_names)}
    chains = {}
    for foot in feet:
        current = f"/go2_description/{foot}"
        selected, seen = [], set()
        while current != "/go2_description/base":
            if current in seen or current not in parents:
                raise ValueError("Foot must descend from Go2 body through an acyclic joint chain")
            seen.add(current)
            current, joint = parents[current]
            selected.append(joint)
        chain = []
        for joint in reversed(selected):
            parent, child = local(joint, 0), local(joint, 1)
            link = {"origin": parent}
            if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                name = joint.GetPrim().GetName()
                if name not in indexes:
                    raise ValueError(f"Undeclared Go2 actuator: {name}")
                axis = UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get()
                link.update(joint=indexes[name], axis=[float(axis == a) for a in "XYZ"],
                            anchor=[0, 0, 0], reference=0)
            elif not joint.GetPrim().IsA(UsdPhysics.FixedJoint):
                raise ValueError("Go2 extraction supports only fixed and revolute joints")
            chain.append(link)
            inverse_position = (-rotation(child[3:]).T @ child[:3]).tolist()
            chain.append({"origin": [*inverse_position, *[-v for v in child[3:6]], child[6]]})
        chains[foot] = chain
    return chains
