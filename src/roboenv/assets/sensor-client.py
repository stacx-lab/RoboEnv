"""Read-only IMU/contact adapter. Kinex dispatch applies calibration bias once."""

import sys

sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)


def handle(action, args):
    try:
        if transport.settings()["type"] not in {"imu", "force_torque"}:
            raise ValueError("Expected an imu or force_torque device")
        if action not in {"read", "load", "unload", "status"} or not isinstance(args, dict) or args:
            raise ValueError("Sensors accept read/load/unload/status with no arguments")
        return {**transport.remote(action, args), "ok": True}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle)
