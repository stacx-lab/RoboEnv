"""Portable device driver: stdlib transport and Kinex's local kinematics."""

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
    from kinex.joints import JointDevice

    return JointDevice(settings(), transport.directory)


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
            if set(args) != {"joints"}:
                raise ValueError("fk requires only joints")
            return {"ok": True, **joint.fk(args["joints"])}
        if action == "ik":
            if set(args) - {"pose", "seed"} or "pose" not in args:
                raise ValueError("ik requires pose and optional seed")
            seed = args.get("seed")
            if seed is None:
                seed = joint.project(remote("read", {})["positions"])
            return {"ok": True, **joint.ik(args["pose"], seed)}
        if action == "write":
            value = joint.command(args)
            data = remote(action, value)
            if data.get("accepted") is not True:
                raise ValueError("write did not acknowledge acceptance")
            return {"ok": True, **data}
        if action == "read":
            data = remote(action, args)
            return {
                "ok": True,
                **joint.reading({**data, "frame_poses": data.get("frame_poses", {})}),
            }
        raise ValueError(f"Unsupported joint action: {action}")
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle, transport)
