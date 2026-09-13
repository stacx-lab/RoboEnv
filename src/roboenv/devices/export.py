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


def driver_resources(directory):
    """Return validated, explicitly portable files from a static device bundle.

    Existing catalogs contain only top-level Python support. A resource manifest
    additionally permits nested local geometry without exporting episode data.
    """
    directory = Path(directory)
    manifest = directory / "driver-resources.json"
    if manifest.is_symlink():
        raise ValueError("Driver resource manifest must not be a symlink")
    names = sorted(path.name for path in directory.glob("*.py"))
    if manifest.exists():
        content = json.loads(manifest.read_text())
        if not isinstance(content, dict) or set(content) != {"schema", "files"}:
            raise ValueError("Invalid driver resource manifest")
        if content["schema"] != "roboenv/driver-resources/1":
            raise ValueError("Unsupported driver resource manifest")
        names = content["files"]
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("Driver resources must be relative file names")
        if len(set(names)) != len(names) or "driver.py" not in names:
            raise ValueError("Driver resources must include driver.py exactly once")
    resources = []
    for name in names:
        relative = Path(name)
        if (
            not name or relative.is_absolute() or ".." in relative.parts
            or str(relative) != name or relative.suffix not in {".py", ".json", ".urdf"}
            or relative.name in {"connection.json", "owner-record.json", "verdict.json"}
            or any(part.startswith(".") for part in relative.parts)
        ):
            raise ValueError(f"Invalid portable driver resource: {name}")
        source = directory / relative
        if any(path.is_symlink() for path in (source, *source.parents)):
            raise ValueError("Device support files must not contain symlinks")
        if not source.is_file() or not source.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Missing portable driver resource: {name}")
        resources.append(relative)
    if manifest.exists():
        resources.append(Path(manifest.name))
    return resources


def driver_access(kind, name, actions, endpoint, *, timeout_s=None):
    driver = {"entry": "driver.py", "interpreter": "python3",
              "persistent": True, "actions": actions}
    if timeout_s is not None:
        driver["timeout_s"] = timeout_s
    if kind in DRIVERS:
        return {
            "source": "user", "method": "unknown", "calibrated_at": None,
            "connection": {"kind": "url", "url": endpoint,
                           "camera" if kind in {"rgb", "rgbd"} else "device": name},
            "driver": driver,
        }
    return {"source": "user", "connection": {"kind": "url", "url": endpoint, "device": name},
            "driver": driver}


def write_driver(directory, kind):
    """Copy the same portable adapter bundle for catalog generation and live export."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assets = files("roboenv").joinpath("assets")
    (directory / "driver.py").write_bytes(
        assets.joinpath(DRIVERS.get(kind, "device-client.py")).read_bytes()
    )
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
            if (descriptor["type"] in {"manipulator", "legged"}
                    and descriptor["intrinsics"].get("kinematics")):
                actions += [action for action in ("fk", "ik") if action not in actions]
            profile["access"] = driver_access(
                descriptor["type"], name, actions, connection["endpoint"],
            )
            write_driver(destination, descriptor["type"])
            (destination / "connection.json").write_text(json.dumps(connection) + "\n")
        (destination / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
