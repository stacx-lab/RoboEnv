"""Legged adapter: fenced RoboEnv transport and Kinex profile validation."""

import sys
from functools import cache

# Importing the support file must not modify the copied catalog.
sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


@cache
def model():
    from kinex.legged import Legged

    return Legged(settings(), transport.directory)


def handle(action, args):
    if action not in {"read", "write", "stop", "load", "unload", "status", "fk", "ik"}:
        return {"ok": False, "error": f"Unsupported legged action: {action}"}
    try:
        from kinex.legged import command, reading

        profile = settings()
        if action not in {"write", "fk", "ik"} and args:
            raise ValueError(f"{action} takes no arguments")
        if action == "fk":
            if set(args) != {"joints"}:
                raise ValueError("fk requires only joints")
            return {"ok": True, **model().fk(args["joints"])}
        if action == "ik":
            if set(args) - {"pose", "seed"} or "pose" not in args:
                raise ValueError("ik requires pose and optional seed")
            solver = model()
            seed = args.get("seed")
            if seed is None:
                seed = solver.project(remote("read", {})["positions"])
            return {"ok": True, **solver.ik(args["pose"], seed)}
        if action == "write":
            value = command(profile, args, model().model)
            return {"ok": True, **remote(action, value)}
        if action == "read":
            return {"ok": True, **reading(profile, remote(action, args), model().model)}
        if action in {"stop", "unload"}:
            result = remote("stop", {})
            if result.get("stopped") is not True:
                raise ValueError("Remote legged device did not acknowledge stop")
            if action == "unload":
                return {"ok": True, **remote(action, args)}
            return {"ok": True, **result}
        return {"ok": True, **remote(action, args)}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
