"""Portable world clock driver."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)


def handle(action, args):
    try:
        if action not in {"read", "wait", "load", "unload", "status"}:
            raise ValueError(f"Unsupported world action: {action}")
        return {"ok": True, **transport.remote(action, args)}
    except WireError as error:
        return {"ok": False, "error": str(error), "code": error.code}
    except Exception as error:
        return {"ok": False, "error": str(error)}


if __name__ == "__main__":
    main(handle, transport)
