"""Packaged backend actions with explicit controller injection."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"gripper_pose": sim.get_gripper_pose_left()}

    @action
    def state(args):
        return {"flange_pose": sim.get_gripper_pose_left()}

    @action
    def move(args):
        target = args.get("target")
        if not target or len(target) < 3:
            return {
                "ok": False,
                "error": "move needs target [x, y, z] or [x, y, z, qw, qx, qy, qz]",
            }
        quat = target[3:7] if len(target) >= 7 else None
        rep = sim.goto_pose_left(target[:3], quat) if quat else sim.goto_pose_left(target[:3])
        return {"reported": rep, "pose": sim.get_gripper_pose_left()}

    @action
    def displace(args):
        deltas = {k[1]: float(args[k]) for k in ("dx", "dy", "dz") if k in args}
        if not deltas:
            return {"ok": False, "error": "displace needs dx/dy/dz"}
        return {"reported": sim.move_by_displacement_left(**deltas)}

    @action
    def home(args):
        return {"reported": sim.back_to_origin_left()}

    install(ACTIONS, backend, manipulator=True, dispatch=True, native_wxyz=True)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_pose_left"],
    "state": ["get_gripper_pose_left"],
    "move": ["get_gripper_pose_left", "goto_pose_left"],
    "displace": ["move_by_displacement_left"],
    "home": ["back_to_origin_left"],
    "stop": [],
    "write": ["get_gripper_pose_left", "goto_pose_left"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
