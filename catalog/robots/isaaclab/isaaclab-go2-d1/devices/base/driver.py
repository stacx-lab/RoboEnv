"""Mobile base adapter: fenced transport and velocity/posture validation."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    if action not in {
        "read",
        "write",
        "stop",
        "load",
        "unload",
        "status",
    }:
        return {"ok": False, "error": f"Unsupported base action: {action}"}
    try:
        if not isinstance(args, dict) or (action not in {"write"} and args):
            raise ValueError(f"Invalid arguments for {action}")
        if action in {"stop", "unload"}:
            result = remote("stop", {})
            if result.get("stopped") is not True:
                raise ValueError("Remote base did not acknowledge stop")
            return {"ok": True, **(remote("unload", {}) if action == "unload" else result)}
        if action in {"load", "status"}:
            return {"ok": True, **remote(action, args)}
        profile = settings()
        from kinex.velocity import command, reading

        if action == "write":
            value = command(profile, args)
            return {"ok": True, **remote(action, value)}
        return {"ok": True, **reading(profile, remote("read", {}))}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
