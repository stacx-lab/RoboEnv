"""Counted libero world actions with explicit controller injection."""

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
        poses = sim.get_all_object_poses()
        return {
            "goal": sim.get_goal_predicates(),
            "objects": {name: {"pose": pose} for name, pose in poses.items()},
            "regions": sim.get_region_boxes(),
            "articulations": sim.get_joint_states(),
            "hand_pose": sim.get_hand_pose(),
            "budget": sim.get_step_budget(),
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
        "get_all_object_poses",
        "get_goal_predicates",
        "get_hand_pose",
        "get_joint_states",
        "get_region_boxes",
        "get_step_budget",
    ],
    "api": [],
    "invoke": [],
    "state": [
        "get_all_object_poses",
        "get_goal_predicates",
        "get_hand_pose",
        "get_joint_states",
        "get_region_boxes",
        "get_step_budget",
    ],
    "load": [],
    "status": [],
    "catalog": [],
}
