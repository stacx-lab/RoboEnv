"""Counted robocasa arm actions with explicit controller injection."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"gripper_site_pose": sim.get_gripper_pose()}

    @action
    def state(args):
        return {
            "flange_pose": sim.get_gripper_pose(),
            "gripper_opening": sim.get_gripper_opening(),
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
        rep = sim.goto_pose(target[:3], quat, args.get("tolerance", 0.02))
        return {"reported": rep, "pose": sim.get_gripper_pose()}

    @action
    def displace(args):
        deltas = {k[1]: float(args[k]) for k in ("dx", "dy", "dz") if k in args}
        if not deltas:
            return {"ok": False, "error": "displace needs dx/dy/dz"}
        return {"reported": sim.move_by_displacement(**deltas), "pose": sim.get_gripper_pose()}

    @action
    def pick(args):
        obj = args.get("object")
        if not obj:
            return {"ok": False, "error": "pick needs {object}"}
        before = sim.get_object_pose(obj)
        rep = sim.pick_object(obj)
        return {"reported": rep, "pose_before": before, "pose_after": sim.get_object_pose(obj)}

    @action
    def place(args):
        obj = args.get("object")
        if not obj:
            return {"ok": False, "error": "place needs {object}"}
        if {"x", "y", "surface_z"} <= set(args):
            rep = sim.place_object(
                obj,
                x=args["x"],
                y=args["y"],
                surface_z=args["surface_z"],
                clearance_m=args.get("clearance_m", 0.03),
                gentle=True,
            )
        elif args.get("fixture"):
            rep = sim.place_object(obj, fixture=args["fixture"], gentle=True)
        else:
            rep = sim.place_gently(obj)
        return {"reported": rep, "pose": sim.get_object_pose(obj)}

    @action
    def door_open(args):
        if not args.get("fixture"):
            return {"ok": False, "error": "door_open needs {fixture}"}
        return {"reported": sim.open_door(args["fixture"])}

    @action
    def door_close(args):
        if not args.get("fixture"):
            return {"ok": False, "error": "door_close needs {fixture}"}
        return {"reported": sim.close_door(args["fixture"])}

    install(ACTIONS, backend, manipulator=True, dispatch=True, native_wxyz=True)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_opening", "get_gripper_pose", "get_step_budget"],
    "state": ["get_gripper_opening", "get_gripper_pose", "get_step_budget"],
    "move": ["get_gripper_pose", "goto_pose"],
    "displace": ["get_gripper_pose", "move_by_displacement"],
    "pick": ["get_object_pose", "pick_object"],
    "place": ["get_object_pose", "place_gently", "place_object"],
    "door_open": ["open_door"],
    "door_close": ["close_door"],
    "stop": [],
    "write": ["get_gripper_pose", "goto_pose"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
