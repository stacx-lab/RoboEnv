"""End-effector adapter: fenced RoboEnv transport and Kinex profile validation."""

import sys

# Importing the support file must not modify the copied catalog.
sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    if action not in {"read", "write", "stop", "load", "unload", "status"}:
        return {"ok": False, "error": f"Unsupported end-effector action: {action}"}
    try:
        from kinex.end_effector import command, reading, stopped

        if action != "write" and args:
            raise ValueError(f"{action} takes no arguments")
        profile = settings()
        if action == "write":
            value = command(profile, args)
            if "native_speed" in value:
                raise ValueError("RoboCasa does not support speedFraction")
            return {"ok": True, **remote(action, value)}
        if action == "read":
            return {"ok": True, **reading(profile, remote(action, args))}
        if action == "stop":
            expected = stopped(profile)
            result = remote(action, args)
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Remote stop result differs from profile scope/behavior")
            return {"ok": True, **result}
        return {"ok": True, **remote(action, args)}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle)
