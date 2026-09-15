"""SIMPLE G1 profiles from pinned native geometry."""

from roboenv_runtime.adapters.humanoid_profiles import profiles as humanoid_profiles


NAMES = ("base", "left_arm", "right_arm", "left_hand", "right_hand")
POSTURE_AXES = (("height", "m"), ("roll", "rad"), ("pitch", "rad"), ("yaw", "rad"))

def camera_profile(spec):
    return {
        "type": "rgb",
        "intrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "color": {
                "width": spec["width"],
                "height": spec["height"],
                "fps": None,
                "pixel_format": "rgb8",
                "image_format": "png",
                "projection": spec["projection"],
                "distortion": {"model": "none"},
            },
        },
        "extrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "mount": spec["mount"],
        },
    }


def profiles(model):
    result = humanoid_profiles(model, camera_profile=camera_profile)
    del result["waist"]
    result["base"]["extrinsics"]["frames"] = ["torso"]
    for name in ("left_arm", "right_arm", *model["cameras"]):
        mount = result[name]["extrinsics"]["mount"]
        mount.update(parent="base", parent_frame="torso")
    return result
