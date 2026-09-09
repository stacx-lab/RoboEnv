"""Export live device facts and thin Kinex drivers, without a harness dependency."""

import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import yaml

DRIVERS = {
    "rgb": "camera-client.py", "rgbd": "camera-client.py",
    "end_effector": "gripper-client.py", "base": "base-client.py",
    "manipulator": "manipulator-client.py", "force_torque": "sensor-client.py",
    "imu": "sensor-client.py", "legged": "legged-client.py",
}
PORTABLE_TYPES = frozenset(DRIVERS)


def driver_access(kind, name, actions, endpoint, *, control_mode=None, timeout_s=None):
    driver = {"entry": "driver.py", "interpreter": "python3",
              "persistent": False, "actions": actions}
    if timeout_s is not None:
        driver["timeout_s"] = timeout_s
    if kind in DRIVERS:
        return {
            "source": "user", "method": "unknown", "calibrated_at": None,
            "connection": {"kind": "url", "url": endpoint,
                           "camera" if kind in {"rgb", "rgbd"} else "device": name},
            "driver": driver,
        }
    return {"source": "user", "kind": "url", "url": endpoint, "driver": driver,
            **({"control_mode": control_mode} if control_mode is not None else {})}


def write_driver(directory, kind):
    """Copy the same portable adapter bundle for catalog generation and live export."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assets = files("roboenv").joinpath("assets")
    (directory / "driver.py").write_bytes(
        assets.joinpath(DRIVERS.get(kind, "device-client.py")).read_bytes()
    )
    if kind in PORTABLE_TYPES:
        (directory / "_roboenv.py").write_bytes(assets.joinpath("_roboenv.py").read_bytes())


def export_devices(profiles, directory, connection):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for descriptor in profiles:
        name = descriptor["name"]
        destination = directory / name
        destination.mkdir(exist_ok=True)
        profile = {key: descriptor[key] for key in ("name", "type", "intrinsics", "extrinsics")}
        timestamp = datetime.now(timezone.utc).isoformat()
        profile["meta"] = {"created_at": timestamp, "updated_at": timestamp}
        if descriptor["type"] == "world":
            profile.pop("extrinsics", None)
        if descriptor["type"] != "world":
            actions = [key for key, value in descriptor["actions"].items() if value["supported"]]
            if descriptor["type"] == "manipulator" and descriptor["intrinsics"].get("kinematics"):
                actions += [action for action in ("fk", "ik") if action not in actions]
            profile["access"] = driver_access(
                descriptor["type"], name, actions, connection["endpoint"],
                control_mode=descriptor["control_mode"],
            )
            write_driver(destination, descriptor["type"])
            (destination / "connection.json").write_text(json.dumps(connection) + "\n")
        (destination / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
