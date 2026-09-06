"""Counted backend actions for robodojo-arx-x5/right_gripper; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    ARM = "right"

    @action
    def read(args):
        return {"state": sim.get_gripper_state(ARM)}

    ACTIONS["state"] = ACTIONS["read"]

    @action
    def move(args):
        target = args.get("target")
        if not target:
            return {"ok": False, "error": "move needs target [grip] (0 closed .. 1 open)"}
        fraction = float(target[0])
        rep = (
            sim.open_gripper(ARM, fraction) if fraction >= 0.5 else sim.close_gripper(ARM, fraction)
        )
        return {"reported": rep}

    install(ACTIONS, backend, manipulator=False, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_state"],
    "move": ["close_gripper", "open_gripper"],
    "stop": [],
    "state": ["get_gripper_state"],
    "write": ["close_gripper", "open_gripper"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
