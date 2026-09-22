"""Test-only probe for device observations, local IK and concurrent writes."""

from kinex import args, device, devices, finish, frames, profile
from kinex.geometry import compose_rt, pose_to_rt, rt_to_pose, xyzq_to_rt
from kinex.joints import JointDevice
from kinex.motion import pose_error


def readings(results):
    """Collect successful device observations; timestamps remain per device."""
    if any(not result["ok"] for result in results):
        raise ValueError("A device read failed")
    return {result["device"]: result["data"] for result in results}


def main():
    options = args()
    names = ["base", "left_arm", "right_arm", "left_hand", "right_hand"]
    if profile("left_arm")["extrinsics"]["mount"]["parent"] == "waist":
        names.insert(1, "waist")
    goals = options.get("goals", {})
    if set(goals) - {"left_arm", "right_arm"}:
        raise ValueError("goals may name left_arm and right_arm")
    body = options.get(
        "body",
        {
            "mode": "locomotion",
            "velocity": {"vx": 0, "vy": 0, "wz": 0},
            "posture": {
                c["axis"]: c["default"] for c in
                profile("base")["intrinsics"]["control"]["modes"]["locomotion"]["posture"]["components"]
            },
        },
    )
    reads = [{"device": name, "action": "read"} for name in names]
    initial = readings(devices(reads))
    retained = {
        name: JointDevice(profile(name)).project(initial[name]["positions"])
        for name in names[1:]
    }
    errors = {}
    # Each write runs the goal for this much of the device's own clock and reports
    # what it actually spent, so the loop needs no rate and no clock of its own.
    segment_s = options.get("segment_s", 0.1)
    elapsed = 0.0

    def stop():
        results = devices([{"device": name, "action": "stop"} for name in names])
        if any(not result["ok"] for result in results):
            raise ValueError("A component stop failed")

    try:
        for _ in range(options.get("steps", 50)):
            before = readings(devices(reads))
            if any(state.get("faults") for state in before.values()):
                raise ValueError("Device fault")
            targets = {"base": body, **retained}
            for arm, goal in goals.items():
                transform = frames.pose(goal["frame"], arm, readings=before)
                value = rt_to_pose(compose_rt(pose_to_rt(transform), xyzq_to_rt(goal["pose"])))
                desired = [*value["translation"], *value["rotation"]]
                targets[arm] = device(
                    arm, "ik", pose=desired,
                    seed=JointDevice(profile(arm)).project(before[arm]["positions"]),
                )["data"]["joints"]
            results = devices(
                [
                    {
                        "device": name,
                        "action": "write",
                        "args": {**(target if name == "base" else {"target": target}),
                                 "for_s": segment_s},
                    }
                    for name, target in targets.items()
                ],
            )
            if any(not result["ok"] for result in results):
                raise ValueError("A component write failed")
            elapsed += float((results[0].get("data") or {}).get("elapsed_s") or 0.0)
            # Errors describe the observations from this iteration, not write receipts.
            for arm, goal in goals.items():
                actual = frames.pose(f"{arm}:flange", goal["frame"], readings=before)
                measured = [*actual["translation"], *actual["rotation"]]
                errors[arm] = pose_error(goal["pose"], measured)
    finally:
        stop()
    finish({"steps": options.get("steps", 50), "elapsed_s": elapsed,
            "errors": errors, "grasp_confirmed": False})


if __name__ == "__main__":
    main()
