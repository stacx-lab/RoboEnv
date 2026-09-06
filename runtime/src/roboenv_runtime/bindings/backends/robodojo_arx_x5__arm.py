"""Counted backend actions for robodojo-arx-x5/arm; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"gripper_pose": sim.get_gripper_pose()}

    @action
    def state(args):
        return {
            "flange_pose": sim.get_gripper_pose(),
            "gripper": sim.get_gripper_state(),
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
        rep = sim.goto_pose(target[:3], quat) if quat else sim.goto_pose(target[:3])
        return {"reported": rep, "pose": sim.get_gripper_pose()}

    @action
    def displace(args):
        deltas = {k[1]: float(args[k]) for k in ("dx", "dy", "dz") if k in args}
        if not deltas:
            return {"ok": False, "error": "displace needs dx/dy/dz"}
        return {"reported": sim.move_by_displacement(**deltas)}

    @action
    def pick(args):
        obj = args.get("object")
        if not obj:
            return {"ok": False, "error": "pick needs {object, ...pick kwargs}"}
        before = sim.get_object_pose(obj)
        kwargs = {k: v for k, v in args.items() if k != "object"}
        rep = sim.pick(obj, **kwargs)
        return {"reported": rep, "pose_before": before, "pose_after": sim.get_object_pose(obj)}

    @action
    def place(args):
        kwargs = dict(args)
        return {"reported": sim.place_at(**kwargs)}

    @action
    def home(args):
        return {"reported": sim.go_home(args.get("arm", "left"))}

    install(ACTIONS, backend, manipulator=True, dispatch=True, native_wxyz=True)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_pose", "get_gripper_state", "get_step_budget"],
    "state": ["get_gripper_pose", "get_gripper_state", "get_step_budget"],
    "move": ["get_gripper_pose", "goto_pose"],
    "displace": ["move_by_displacement"],
    "pick": ["get_object_pose", "pick"],
    "place": ["place_at"],
    "home": ["go_home"],
    "stop": [],
    "write": ["get_gripper_pose", "goto_pose"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
