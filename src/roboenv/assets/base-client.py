"""Base adapter: fenced RoboEnv transport and Kinex profile validation."""

import sys

# Importing the support file must not modify the copied catalog.
sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    if action not in {"read", "write", "stop", "load", "unload", "status"}:
        return {"ok": False, "error": f"Unsupported base action: {action}"}
    try:
        from kinex.base import command, reading

        profile = settings()
        if action != "write" and args:
            raise ValueError(f"{action} takes no arguments")
        if action == "write":
            value = command(profile, args)
            return {"ok": True, **remote(action, value)}
        if action == "read":
            return {"ok": True, **reading(profile, remote(action, args))}
        if action in {"stop", "unload"}:
            result = remote("stop", {})
            if result.get("stopped") is not True:
                raise ValueError("Remote base did not acknowledge stop")
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
