"""A small normal controller tool, independent of simulator and owner state."""

import time
from kinex import args, device, finish, profile

arguments = args()
arm, camera = arguments["arm"], arguments["camera"]
initial_image = device(camera)
initial = device(arm)["data"]
joints = profile(arm)["intrinsics"]["kinematics"]["model"]["joints"]
target = [min(joint["limits"]["max"], max(joint["limits"]["min"], value))
          if joint["limits"] else value
          for joint, value in zip(joints, initial["positions"])]
bounds = joints[0]["limits"]
target[0] += .10 if bounds is None or target[0] + .10 < bounds["max"] else -.10
try:
    for iteration in range(80):
        device(arm, "write", target=target)
        measured = device(arm)["data"]
        reached = (max(abs(a-b) for a, b in zip(measured["positions"], target)) < .01
                   and max(abs(value) for value in measured["velocities"]) < .02)
        if reached:
            break
        time.sleep(.03)
    if not reached:
        raise RuntimeError("Native arm did not converge within the probe limits")
    if abs(measured["positions"][0] - initial["positions"][0]) < .05:
        raise RuntimeError("Driver accepted commands without effective measured motion")
    # Keep enough wall time for several recorded camera acquisitions.
    for _ in range(8):
        final_image = device(camera)
        time.sleep(.4)
finally:
    device(arm, "stop")
finish({"reached": True, "iterations": iteration + 1,
        "before": initial, "after": measured}, file=final_image.get("file"))
