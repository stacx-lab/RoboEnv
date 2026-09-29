"""A small normal controller tool, independent of simulator and owner state.

RoboEnv 0.9 actuators accept timed segments: `write {actions, duration_s}` with
joint vectors in model order. The world clock executes them: `world wait` steps a
stepped world and waits for simulated time in a realtime one.
"""

import time

from kinex import args, device, finish, profile

STEP = 0.10  # Commanded change of the first joint (radians or metres).
MOVED = 0.05  # Measured change that proves effective motion.
# Native position control may settle with a small offset (about 0.009 rad on RoboTwin).
TOLERANCE = 0.02
QUIET = 0.02
WAYPOINTS = 10
SEGMENT_S = 1.0
SETTLE_S = 0.25
SETTLE_ROUNDS = 40
HOLD_S = 0.4
HOLD_ROUNDS = 8


def clamp(value, joint):
    limits = joint.get("limits")
    return min(limits["max"], max(limits["min"], value)) if limits else value


def arrived(state, target):
    error = max(abs(a - b) for a, b in zip(state["positions"], target, strict=True))
    # Velocities are empty when the arm does not report them.
    velocity = max((abs(value) for value in state.get("velocities") or []), default=0.0)
    return error < TOLERANCE and velocity < QUIET


def read(name):
    state = device(name)["data"]
    if state.get("faults"):
        raise RuntimeError(f"{name} reported faults: {state['faults']}")
    return state


def main():
    arguments = args()
    arm, camera = arguments["arm"], arguments["camera"]
    world = arguments.get("world", "world")
    device(camera)
    arm_profile = profile(arm)
    joints = (arm_profile.get("resolved_model") or arm_profile["intrinsics"]["model"])["joints"]
    initial = read(arm)
    start = [clamp(value, joint) for joint, value in zip(joints, initial["positions"], strict=True)]
    target = list(start)
    bounds = joints[0].get("limits")
    target[0] += STEP if bounds is None or target[0] + STEP < bounds["max"] else -STEP
    # One smooth segment from the measured configuration; other joints hold.
    actions = [
        [a + (b - a) * (index + 1) / WAYPOINTS for a, b in zip(start, target, strict=True)]
        for index in range(WAYPOINTS)
    ]
    elapsed = 0.0
    reached = False
    try:
        write = device(arm, "write", actions=actions, duration_s=SEGMENT_S)["data"]
        elapsed += device(world, "wait", duration_s=SEGMENT_S)["data"]["elapsed_s"]
        measured = read(arm)
        waits = 1
        while not arrived(measured, target) and waits <= SETTLE_ROUNDS:
            elapsed += device(world, "wait", duration_s=SETTLE_S)["data"]["elapsed_s"]
            measured = read(arm)
            waits += 1
        reached = arrived(measured, target)
        if not reached:
            raise RuntimeError("Native arm did not converge within the probe limits")
        if abs(measured["positions"][0] - initial["positions"][0]) < MOVED:
            raise RuntimeError("Driver accepted commands without effective measured motion")
        # Hold while the world advances, for several recorded camera acquisitions.
        for _ in range(HOLD_ROUNDS):
            started = time.monotonic()
            elapsed += device(world, "wait", duration_s=HOLD_S)["data"]["elapsed_s"]
            final_image = device(camera)
            time.sleep(max(0.0, HOLD_S - (time.monotonic() - started)))
    finally:
        device(arm, "stop")
    finish(
        {
            "completed": True,
            "reached": reached,
            "iterations": waits,
            "joint": joints[0]["name"],
            "segment": {
                "actions": len(actions),
                "duration_s": SEGMENT_S,
                "target": target,
                "execution_id": write.get("execution_id"),
            },
            "world_elapsed_s": elapsed,
            "before": initial,
            "after": measured,
        },
        file=final_image.get("file"),
    )


if __name__ == "__main__":
    main()
