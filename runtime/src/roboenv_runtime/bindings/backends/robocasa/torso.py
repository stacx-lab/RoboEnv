"""Counted robocasa torso actions with explicit controller injection."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        return {"gripper_site_z": sim.get_gripper_pose()[2]}

    ACTIONS["state"] = ACTIONS["read"]

    @action
    def move(args):
        target = args.get("target")
        if not target:
            return {"ok": False, "error": "move needs target [height_m]"}
        return {"reported": sim.set_torso_height(float(target[0]))}

    install(ACTIONS, backend, manipulator=True, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": ["get_gripper_pose"],
    "move": ["set_torso_height"],
    "stop": [],
    "state": ["get_gripper_pose"],
    "write": ["set_torso_height"],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
