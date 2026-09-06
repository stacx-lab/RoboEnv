"""Counted backend actions for robocasa-panda-omron/base; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"pose": sim.get_base_pose()}

    @action
    def state(args):
        return {"pose": sim.get_base_pose(), "budget": sim.get_step_budget()}

    @action
    def move(args):
        target = args.get("target")
        if not target or len(target) < 2:
            return {"ok": False, "error": "move needs target [x, y, yaw?] (device action space)"}
        yaw = target[2] if len(target) > 2 else 0.0
        return {"reported": sim.base_move(target[0], target[1], yaw), "pose": sim.get_base_pose()}

    @action
    def dock(args):
        if "fixture" not in args:
            return {"ok": False, "error": "dock needs {fixture}"}
        return {"reported": sim.drive_to_fixture(args["fixture"]), "pose": sim.get_base_pose()}

    install(ACTIONS, backend, manipulator=False, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_base_pose", "get_step_budget"],
    "state": ["get_base_pose", "get_step_budget"],
    "move": ["base_move", "get_base_pose"],
    "dock": ["drive_to_fixture", "get_base_pose"],
    "stop": [],
    "write": ["base_move", "get_base_pose"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
