"""Adapt the existing simulator drivers to Complex's current action interface.

Native primitives retain their existing parameters. write uses the profile's
declared action order. Kinematics and flange transforms are never invented.
"""

from . import client as simlink


def install(actions, *, manipulator=False, dispatch=False, native_wxyz=False):
    """Install standard lifecycle/read/write actions around an existing driver."""
    native_read = actions.get("state", actions.get("read"))
    native_move = actions.get("move")

    def read(args):
        args = dict(args)
        if args.get("file_name"):
            args["name"] = args.pop("file_name")
        result = dict(native_read(args) or {})
        # These older drivers report a simulator gripper/TCP pose, NOT a known
        # base-to-flange transform. Do not let complex.frames compose it as one.
        if "flange_pose" in result:
            result["native_tool_pose"] = result.pop("flange_pose")
            result["pose_convention"] = "simulator native frame; quaternion wxyz"
        return result

    def write(args):
        if native_move is None:
            raise RuntimeError("write is unsupported; use this device's documented action")
        command = dict(args)
        if native_wxyz and "target" in command:
            target = list(command["target"])
            if len(target) not in (3, 7):
                raise ValueError("write.target must be xyz or xyz + quaternion xyzw")
            if len(target) == 7:
                command["target"] = [*target[:3], target[6], *target[3:6]]
        return native_move(command)

    def status(args):
        health = simlink.rpc("health")
        ready = health.get("worker", {}).get("state") == "ready"
        if not ready:
            raise RuntimeError(f"simulator is not ready: {health.get('worker')}")
        return {"loaded": True, "service": health["worker"]}

    def unload(args):
        return {"unloaded": True, "note": "client deactivated; Harbor owns the shared episode"}

    def unsupported(args):
        raise RuntimeError(
            "unsupported_kinematics: this bridge exposes native Cartesian primitives, "
            "not general joint-space IK/FK"
        )

    def invoke(args):
        name = args.get("tool_name", args.get("name"))
        if not isinstance(name, str) or name not in simlink.api().get("tools", {}):
            raise ValueError(f"unknown public simulator tool: {name}")
        return {"result": simlink.call(name, **(args.get("kwargs") or {}))}

    def policy_run(args):
        request = {"n_actions": int(args.get("max_steps", 50))}
        if args.get("task"):
            request["instruction"] = args["task"]
        return {"report": simlink.rpc("vla", request, timeout=1800)}

    actions.update(read=read, load=status, status=status, unload=unload)
    if "state" in actions:
        actions["state"] = read
    if native_move is not None:
        actions["write"] = write
    if manipulator:
        actions.update(ik=unsupported, fk=unsupported)
    if dispatch:
        actions.update(invoke=invoke, policy_run=policy_run)
    else:
        actions.pop("invoke", None)
    actions["catalog"] = lambda args: simlink.api()
