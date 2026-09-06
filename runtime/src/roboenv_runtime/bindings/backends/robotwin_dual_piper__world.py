"""Counted backend actions for robotwin-dual-piper/world; no catalog execution or RPC client."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    simlink = backend
    sim = backend.functions
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    @action
    def read(args):
        objects = {}
        for name in sim.list_objects():
            # Functional annotations are not universal scene properties: the
            # table, for example, is a plain SAPIEN entity without such points.
            # Keep this read to universally supported poses. Annotated objects
            # remain queryable through the explicit functional-pose primitive.
            objects[name] = {"pose": sim.get_object_pose(name)}
        return {
            "objects": objects,
            "grippers": {
                "left": sim.get_gripper_pose_left(),
                "right": sim.get_gripper_pose_right(),
            },
            "budget": simlink.rpc("observe").get("budget"),
            "score": simlink.rpc("score.peek"),
        }

    ACTIONS["state"] = ACTIONS["read"]

    @action
    def api(args):
        tools = simlink.api().get("tools", {})
        if args.get("name"):
            entry = tools.get(args["name"])
            if entry is None:
                return {"ok": False, "error": f"no such function {args['name']!r}"}
            return {args["name"]: entry}
        return {
            name: {
                "signature": entry.get("signature"),
                "doc": (entry.get("doc") or "").splitlines()[0] if entry.get("doc") else "",
                "read_only": entry.get("read_only"),
            }
            for name, entry in sorted(tools.items())
        }

    @action
    def invoke(args):
        name = args.get("name")
        if not name:
            return {"ok": False, "error": "invoke needs {name, kwargs?}"}
        return {"result": simlink.call(name, **args.get("kwargs") or {})}

    install(ACTIONS, backend, manipulator=False, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": [
        "get_gripper_pose_left",
        "get_gripper_pose_right",
        "get_object_pose",
        "list_objects",
    ],
    "api": [],
    "invoke": [],
    "state": [
        "get_gripper_pose_left",
        "get_gripper_pose_right",
        "get_object_pose",
        "list_objects",
    ],
    "load": [],
    "status": [],
    "catalog": [],
}
