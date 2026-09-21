"""Generic remote device adapter; transport and dialog are shared with typed drivers."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__, device=Path(__file__).parent.name)
remote = transport.remote


def handle(action, args):
    try:
        if action in {"load", "unload", "status"} and args:
            raise ValueError(f"{action} takes no arguments")
        # Kinex owns this client connection; RoboEnv keeps the shared world alive.
        if action in {"load", "status"}:
            remote("read", {})
            return {"ok": True, "loaded": True}
        if action == "unload":
            if transport.can_stop:
                stopped = remote("stop", {})
                if stopped.get("stopped") is not True:
                    raise ValueError("Remote device did not acknowledge stop")
            transport.close()
            transport.unloaded = True
            return {"ok": True, "unloaded": True}
        data = remote(action, args)
        return {**(data if isinstance(data, dict) else {"result": data}), "ok": True}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
