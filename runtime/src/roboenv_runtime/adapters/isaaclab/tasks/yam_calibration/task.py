"""Open-ended practice factories: no evaluator or calibration submission API."""

import copy
from pathlib import Path

TASKS = {
    "yam-calibrate-overhead-camera": ("camera_top",),
    "yam-calibrate-wrist-cameras": ("camera_left", "camera_right"),
    "yam-calibrate-dual-arm": ("right_arm",),
}
BOARD = {
    "squares_x": 7,
    "squares_y": 5,
    "square_m": 0.03,
    "marker_m": 0.0225,
    "dictionary": "DICT_4X4_50",
}


class CalibrationTask:
    embodiment = "yam"
    source_directories = (Path(__file__).parent,)

    @classmethod
    def profiles(cls):
        return load_profiles(cls.name)

    @classmethod
    def validate_profiles(cls, public, physical):
        validate_profiles(public, physical, cls.name)

    @classmethod
    def create_environment(cls, **kwargs):
        from ...robots.yam.environment import YamEnv

        return YamEnv(task=cls, **kwargs)


def robot_name(task_name):
    if task_name not in TASKS:
        raise ValueError(f"Unknown YAM calibration task: {task_name}")
    return "isaaclab-" + task_name


def load_profiles(task_name):
    import yaml

    from roboenv_runtime.paths import repo_root

    directory = repo_root() / "catalog/robots/isaaclab" / robot_name(task_name) / "devices"
    return {
        p.parent.name: {
            k: v
            for k, v in yaml.safe_load(p.read_text()).items()
            if k in {"type", "intrinsics", "extrinsics"}
        }
        for p in sorted(Path(directory).glob("*/profile.yaml"))
    }


def validate_profiles(public, physical, task_name):
    """Calibration extrinsics are editable; hardware/control facts are not."""
    expected = load_profiles(task_name)
    if public.keys() != expected.keys():
        raise ValueError("YAM device inventory differs from the selected exercise")
    for name, profile in public.items():
        if profile["type"] != expected[name]["type"] or (
            profile["intrinsics"] != expected[name]["intrinsics"]
        ):
            raise ValueError(f"YAM hardware profile differs: {name}")
    for name, profile in physical.items():
        if name not in expected:
            raise ValueError("Unexpected physical device in YAM bundle")
        intrinsic = copy.deepcopy(expected[name]["intrinsics"])
        if task_name == "yam-calibrate-dual-arm" and name.endswith("_gripper"):
            intrinsic["geometry"]["description"] = profile["intrinsics"]["geometry"]["description"]
        if intrinsic != profile["intrinsics"]:
            raise ValueError(f"Prepared YAM hardware differs from public facts: {name}")
    if set(physical) != {
        "world",
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
        "camera_top",
        "camera_left",
        "camera_right",
    }:
        raise ValueError("Incomplete physical YAM inventory")
