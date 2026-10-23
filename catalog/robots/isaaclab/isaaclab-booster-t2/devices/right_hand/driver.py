"""Portable device driver: stdlib transport and Kinex's local kinematics."""

import sys
from functools import cache
from pathlib import Path

# Import shared collection support without modifying a read-only catalog.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


@cache
def model():
    from kinex.joints import JointDevice

    from hand_fk import grasp_center

    profile = settings()
    return JointDevice(profile, transport.directory,
                       native_fk=lambda joints: grasp_center(profile, transport.directory, joints))


def handle(action, args):
    try:
        profile = settings()
        if action in {"read", "stop", "load", "unload", "status"} and args:
            raise ValueError(f"{action} takes no arguments")
        if action == "unload":
            result = handle("stop", {})
            if not result["ok"]:
                return result
        if action in {"load", "unload", "status"}:
            return {"ok": True, **remote(action, args)}
        if action == "stop":
            result = remote(action, args)
            expected = {"stopped": True, **profile["intrinsics"]["stop"]}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Remote stop result differs from profile scope/behavior")
            return {"ok": True, **result}
        joint = model()
        if action == "fk":
            if set(args) - {"joints", "frame", "reference_frame"} or "joints" not in args:
                raise ValueError("fk requires joints and optional frame/reference_frame")
            return {"ok": True, **joint.fk(**args)}
        if action == "ik":
            if set(args) - {"pose", "seed", "frame", "reference_frame"} or "pose" not in args:
                raise ValueError("ik requires pose and optional seed/frame/reference_frame")
            seed = args.get("seed")
            if seed is None:
                seed = joint.project(remote("read", {})["positions"])
            return {
                "ok": True,
                **joint.ik(args["pose"], seed, args.get("frame"), args.get("reference_frame")),
            }
        if action == "write":
            joint.command(args)
            return {"ok": True, **transport.write(args, joint.reading)}
        if action == "read":
            data = remote(action, args)
            return {
                "ok": True,
                **joint.reading(data),
            }
        raise ValueError(f"Unsupported joint action: {action}")
    except WireError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "code": exc.code,
            "request_id": exc.request_id,
            "details": exc.details,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
