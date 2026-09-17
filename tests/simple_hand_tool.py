"""Run inside Kinex: hand prediction, named frames, and grasp-to-flange conversion."""

from kinex import device, finish, frames, profile
from kinex.geometry import compose_rt, invert_rt, pose_to_rt, xyzq_to_rt
from kinex.hand import forward
from kinex.joints import JointDevice
from kinex.motion import flat, pose_error

results = {}
for side in ("left", "right"):
    name, arm = side + "_hand", side + "_arm"
    hand = profile(name)
    state = device(name)["data"]
    predicted = forward(hand, state["positions"])
    for frame, pose in predicted.items():
        error = pose_error(pose, state["frame_poses"][frame])
        assert error["position_m"] < 1e-6 and error["rotation_rad"] < 1e-6, error
        resolved = frames.pose(name + ":" + frame, name, readings={name: state})
        error = pose_error(pose, flat(pose_to_rt(resolved)))
        assert error["position_m"] < 1e-6 and error["rotation_rad"] < 1e-6, error
    geometry = hand["intrinsics"]["geometry"]
    reference = frames.pose(name + ":" + geometry["reference_frame"], arm + ":flange")
    arm_profile = profile(arm)
    joint = JointDevice(arm_profile)
    seed = joint.project(device(arm)["data"]["positions"])
    wrist = joint.fk(seed)["pose"]
    desired_reference = compose_rt(xyzq_to_rt(wrist), pose_to_rt(reference))
    desired_wrist = flat(compose_rt(desired_reference, invert_rt(pose_to_rt(reference))))
    solution = joint.ik(desired_wrist, seed)
    error = pose_error(joint.fk(solution["joints"])["pose"], wrist)
    assert error["position_m"] < .001 and error["rotation_rad"] < .01, error
    results[name] = {"frames": predicted, "grasp_reference_in_flange": reference,
                     "arm_conversion_error": error}
finish({"passed": True, "hands": results})
