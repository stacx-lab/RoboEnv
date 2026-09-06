"""Shared truthful action normalization for counted simulator-controller bindings."""


def install(actions, backend, *, manipulator=False, dispatch=False, native_wxyz=False):
    original_read = actions.get("state", actions.get("read"))
    move = actions.get("move")

    def read(args):
        args = dict(args)
        if args.get("file_name"):
            args["name"] = args.pop("file_name")
        result = dict(original_read(args) or {})
        if "flange_pose" in result:
            result["native_tool_pose"] = result.pop("flange_pose")
            result["pose_convention"] = "simulator native frame; quaternion wxyz"
        return result

    def write(args):
        command = dict(args)
        if native_wxyz and "target" in command:
            target = list(command["target"])
            if len(target) not in (3, 7):
                raise ValueError("write.target must be xyz or xyz + quaternion xyzw")
            if len(target) == 7:
                command["target"] = [*target[:3], target[6], *target[3:6]]
        return move(command)

    def status(args):
        health = backend.rpc("health")
        if health.get("worker", {}).get("state") != "ready":
            raise RuntimeError("simulator is not ready")
        return {"loaded": True, "service": health["worker"]}

    def invoke(args):
        name = args.get("tool_name", args.get("name"))
        return {"result": backend.call(name, **(args.get("kwargs") or {}))}

    actions.update(read=read, load=status, status=status, catalog=lambda args: backend.api())
    if "state" in actions:
        actions["state"] = read
    if move is not None:
        actions["write"] = write
    if dispatch:
        actions["invoke"] = invoke
    else:
        actions.pop("invoke", None)
    # No simulator binding can claim a real interrupt or generic joint IK/FK
    # just because a historical profile listed those names.
    for name in ("stop", "ik", "fk", "unload", "policy_run"):
        actions.pop(name, None)
