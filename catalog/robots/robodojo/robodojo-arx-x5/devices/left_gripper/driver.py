"""Submit flat end-effector action sequences through the fenced RoboEnv transport."""

import sys
from pathlib import Path

# Import shared collection support without modifying a read-only catalog.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    if action not in {"read", "write", "stop", "load", "unload", "status"}:
        return {"ok": False, "error": f"Unsupported end-effector action: {action}"}
    try:
        from kinex.scalar_control import command, reading, stopped

        if action != "write" and args:
            raise ValueError(f"{action} takes no arguments")
        profile = settings()
        if action == "write":
            value = command(profile, args)
            if "native_speed" in value:
                raise ValueError("This device does not support speedFraction")
            return {"ok": True, **remote(action, value)}
        if action == "read":
            return {"ok": True, **reading(profile, remote(action, args))}
        if action == "stop":
            expected = stopped(profile)
            result = remote(action, args)
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Remote stop result differs from profile scope/behavior")
            return {"ok": True, **result}
        if action == "unload":
            result = handle("stop", {})
            if not result["ok"]:
                return result
        return {"ok": True, **remote(action, args)}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
