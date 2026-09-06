"""Counted backend actions for libero-panda/arm; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"hand_pose": sim.get_hand_pose()}

    @action
    def state(args):
        return {
            "flange_pose": sim.get_hand_pose(),
            "tip": sim.tcp(),
            "budget": sim.get_step_budget(),
        }

    @action
    def move(args):
        target = args.get("target")
        if not target or len(target) < 3:
            return {
                "ok": False,
                "error": "move needs target [x, y, z] or [x, y, z, qw, qx, qy, qz]",
            }
        quat = target[3:7] if len(target) >= 7 else None
        rep = sim.goto_pose(target[:3], quat)
        return {"reported": rep, "tip": sim.tcp()}

    @action
    def displace(args):
        deltas = {k[1]: float(args[k]) for k in ("dx", "dy", "dz") if k in args}
        if not deltas:
            return {"ok": False, "error": "displace needs dx/dy/dz"}
        return {"reported": sim.move_by_displacement(**deltas), "tip": sim.tcp()}

    @action
    def pick(args):
        obj = args.get("object")
        if not obj:
            return {"ok": False, "error": "pick needs {object, ...grasp_object kwargs}"}
        before = sim.get_object_pose(obj)
        kwargs = {k: v for k, v in args.items() if k != "object"}
        rep = sim.grasp_object(obj, **kwargs)
        return {"reported": rep, "pose_before": before, "pose_after": sim.get_object_pose(obj)}

    @action
    def place(args):
        obj = args.get("object")
        if not obj:
            return {"ok": False, "error": "place needs {object} + region/target kwargs"}
        if args.get("region"):
            rep = (
                sim.place_in_region(obj, args["region"])
                if args.get("inside")
                else sim.place_on_region(obj, args["region"])
            )
        elif args.get("target"):
            rep = sim.place_at(obj, args["target"])
        else:
            rep = sim.place_gently(obj)
        return {"reported": rep, "pose": sim.get_object_pose(obj)}

    @action
    def open_articulated(args):
        kwargs = dict(args)
        joint = kwargs.pop("joint", None)
        if not joint:
            return {"ok": False, "error": "open_articulated needs {joint, ...kwargs}"}
        return {"reported": sim.open_articulated(joint, **kwargs), "joints": sim.get_joint_states()}

    @action
    def close_articulated(args):
        kwargs = dict(args)
        joint = kwargs.pop("joint", None)
        if not joint:
            return {"ok": False, "error": "close_articulated needs {joint, ...kwargs}"}
        return {
            "reported": sim.close_articulated(joint, **kwargs),
            "joints": sim.get_joint_states(),
        }

    install(ACTIONS, backend, manipulator=True, dispatch=True, native_wxyz=True)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_hand_pose", "get_step_budget", "tcp"],
    "state": ["get_hand_pose", "get_step_budget", "tcp"],
    "move": ["goto_pose", "tcp"],
    "displace": ["move_by_displacement", "tcp"],
    "pick": ["get_object_pose", "grasp_object"],
    "place": ["get_object_pose", "place_at", "place_gently", "place_in_region", "place_on_region"],
    "open_articulated": ["get_joint_states", "open_articulated"],
    "close_articulated": ["close_articulated", "get_joint_states"],
    "stop": [],
    "write": ["goto_pose", "tcp"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
