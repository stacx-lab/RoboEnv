"""RoboEnv camera: profile-selected remote device and selectable image output."""

import sys
from pathlib import Path

# Import shared collection support without modifying a read-only catalog.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _camera import camera_read, output_mode  # noqa: E402
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    if action not in {"read", "load", "unload", "status"}:
        return {"ok": False, "error": f"Unsupported camera action: {action}"}
    try:
        if action == "read":
            if set(args) - {"output"}:
                raise ValueError("camera read accepts only output")
            output = output_mode(args.get("output", "file"))
            data = remote(action, {}, image_output=output)
        else:
            if args:
                raise ValueError(f"{action} takes no arguments")
            data = remote(action, args)
        if action == "read":
            data = camera_read(data, settings())
        return {**data, "ok": True}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
