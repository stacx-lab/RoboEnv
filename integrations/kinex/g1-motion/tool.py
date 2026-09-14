"""One explicit loop composing snapshots, local IK and synchronized writes."""

from kinex import args, device, devices, finish, frames, loop, profile
from kinex.geometry import compose_rt, pose_to_rt, rt_to_pose, xyzq_to_rt
from kinex.motion import pose_error


def acquisition(results):
    """Adapt synchronized devices() results for frame-graph calculations."""
    return {
        "sample_time_s": results[0]["data"]["sample_time_s"],
        "states": {result["device"]: result["data"] for result in results},
    }


def main():
    options = args()
    names = ["base", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    goals = options.get("goals", {})
    if set(goals) - {"left_arm", "right_arm"}:
        raise ValueError("goals may name left_arm and right_arm")
    body = options.get(
        "body",
        {
            "mode": "locomotion",
            "velocity": {"vx": 0, "vy": 0, "wz": 0},
            "posture": {
                "height": profile("base")["intrinsics"]["control"]["modes"]["locomotion"][
                    "posture"
                ]["components"][0]["default"]
            },
        },
    )
    realtime = profile("base")["intrinsics"]["control"]["execution"]["mode"] == "realtime"
    reads = [{"device": name, "action": "read"} for name in names]
    initial = acquisition(devices(reads, synchronized=True))
    retained = {name: initial["states"][name]["positions"] for name in names[1:]}
    errors = {}
    last_time = None

    def tick(index):
        nonlocal last_time
        before = acquisition(devices(reads, synchronized=True))
        stamp = before["sample_time_s"]
        if last_time is not None and stamp < last_time:
            raise ValueError("Simulation clock moved backwards")
        if realtime and stamp == last_time:
            return False
        last_time = stamp
        if any(state.get("faults") for state in before["states"].values()):
            raise ValueError("Device fault")
        targets = {"base": body, **retained}
        desired = {}
        for arm, goal in goals.items():
            transform = frames.pose(goal["frame"], f"{arm}:base", snapshot=before)
            value = rt_to_pose(compose_rt(pose_to_rt(transform), xyzq_to_rt(goal["pose"])))
            desired[arm] = [*value["translation"], *value["rotation"]]
            targets[arm] = device(
                arm, "ik", pose=desired[arm], seed=before["states"][arm]["positions"]
            )["data"]["joints"]
        results = devices(
            [
                {
                    "device": name,
                    "action": "write",
                    "args": target if name == "base" else {"target": target},
                }
                for name, target in targets.items()
            ],
            synchronized=True,
        )
        # Realtime receipts acknowledge targets; they are not post-step feedback.
        after = before if realtime else acquisition(results)
        for arm, goal in goals.items():
            actual = frames.pose(f"{arm}:flange", goal["frame"], snapshot=after)
            errors[arm] = pose_error(goal["pose"], [*actual["translation"], *actual["rotation"]])
        return False  # The explicit step budget controls the motion duration.

    def stop():
        results = devices([{"device": name, "action": "stop"} for name in names])
        if any(not result["ok"] for result in results):
            raise ValueError("A component stop failed")

    result = loop(
        tick,
        max_steps=options.get("steps", 50),
        timeout_s=100,
        on_stop=stop,
        rate_hz=options.get("rate_hz", 20) if realtime else None,
    )
    finish({**result, "errors": errors, "grasp_confirmed": False})


if __name__ == "__main__":
    main()
