"""Counted shared camera actions with explicit controller injection."""

from roboenv_runtime.bindings.actions import install


def create_actions(backend):
    simlink = backend
    ACTIONS = {}

    def action(fn):
        ACTIONS[fn.__name__.rstrip("_")] = fn
        return fn

    import time

    @action
    def read(args):
        name = args.get("name") or f"frame-{int(time.time() * 1000)}.png"
        if not name.endswith(".png"):
            name += ".png"
        return {"file": simlink.capture(name)}

    @action
    def ping(args):
        return {"note": "frames accumulate on action steps; read after the first motion"}

    install(ACTIONS, backend, manipulator=False, dispatch=False, native_wxyz=False)
    return ACTIONS


REQUIRES = {
    "action": [],
    "read": [],
    "ping": [],
    "state": [],
    "load": [],
    "status": [],
    "catalog": [],
    "invoke": [],
}
