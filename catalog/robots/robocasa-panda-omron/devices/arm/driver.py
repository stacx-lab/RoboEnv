"""Portable device driver: stdlib transport and Kinex's local kinematics."""

import sys

# Importing the support file must not modify the copied catalog.
sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def handle(action, args):
    try:
        from kinex.manipulator import Manipulator

        profile = settings()
        if action in {"read", "stop", "load", "unload", "status"} and args:
            raise ValueError(f"{action} takes no arguments")
        if action in {"load", "unload", "status"}:
            return {"ok": True, **remote(action, args)}
        if action == "stop":
            result = remote(action, args)
            expected = {"stopped": True, **profile["intrinsics"]["stop"]}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Remote stop result differs from profile scope/behavior")
            return {"ok": True, **result}
        model = Manipulator(profile, transport.directory)
        if action == "fk":
            if set(args) != {"joints"}:
                raise ValueError("fk requires only joints")
            return {"ok": True, **model.fk(args["joints"])}
        if action == "ik":
            if set(args) - {"pose", "seed"} or "pose" not in args:
                raise ValueError("ik requires pose and optional seed")
            seed = args.get("seed")
            if seed is None:
                seed = model.project(remote("read", {})["positions"])
            return {"ok": True, **model.ik(args["pose"], seed)}
        if action == "write":
            value = model.command(args)
            data = remote(action, value)
            if data.get("accepted") is not True:
                raise ValueError("write did not acknowledge acceptance")
            return {"ok": True, **data}
        if action == "read":
            data = remote(action, args)
            return {"ok": True, **model.reading({**data, "reported_pose": data["flange_pose"]})}
        raise ValueError(f"Unsupported manipulator action: {action}")
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle)
