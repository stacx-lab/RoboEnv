"""Counted backend actions for robocasa-panda-omron/world; no catalog execution or RPC client."""

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
            objects[name] = {
                "pose": sim.get_object_pose(name),
                "extent": sim.get_object_extent(name),
            }
        fixtures = sim.get_task_fixtures()
        regions = {}
        for _role, fixture in (fixtures or {}).items():
            try:
                regions[fixture] = sim.get_fixture_regions(fixture)
            except Exception as exc:
                regions[fixture] = f"(no regions: {exc})"
        return {
            "instruction": sim.get_instruction(),
            "objects": objects,
            "fixtures": fixtures,
            "fixture_regions": regions,
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
        "get_fixture_regions",
        "get_instruction",
        "get_object_extent",
        "get_object_pose",
        "get_step_budget",
        "get_task_fixtures",
        "list_objects",
    ],
    "api": [],
    "invoke": [],
    "state": [
        "get_fixture_regions",
        "get_instruction",
        "get_object_extent",
        "get_object_pose",
        "get_step_budget",
        "get_task_fixtures",
        "list_objects",
    ],
    "load": [],
    "status": [],
    "catalog": [],
}
