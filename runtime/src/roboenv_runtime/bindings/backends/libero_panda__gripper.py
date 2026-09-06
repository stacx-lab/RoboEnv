"""Counted backend actions for libero-panda/gripper; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"opening": sim.get_gripper_opening()}

    ACTIONS["state"] = ACTIONS["read"]

    @action
    def move(args):
        target = args.get("target")
        if not target:
            return {"ok": False, "error": "move needs target [grip] (0 closed .. 1 open)"}
        rep = sim.open_gripper() if float(target[0]) >= 0.5 else sim.close_gripper()
        return {"reported": rep, "opening": sim.get_gripper_opening()}

    install(ACTIONS, backend, manipulator=False, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_opening"],
    "move": ["close_gripper", "get_gripper_opening", "open_gripper"],
    "stop": [],
    "state": ["get_gripper_opening"],
    "write": ["close_gripper", "get_gripper_opening", "open_gripper"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
