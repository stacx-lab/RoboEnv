"""Export live device facts and thin Complex drivers, without a harness dependency."""

import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import yaml


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
            if descriptor["type"] == "manipulator" and "chain" in descriptor["intrinsics"].get(
                "kinematics", {}
            ):
                actions += [action for action in ("fk", "ik") if action not in actions]
            profile["access"] = {
                "source": "user",
                "kind": "url",
                "url": connection["endpoint"],
                **(
                    {"control_mode": descriptor["control_mode"]}
                    if descriptor["control_mode"] is not None
                    else {}
                ),
                "driver": {
                    "entry": "driver.py",
                    "interpreter": "python3",
                    "persistent": False,
                    "actions": actions,
                },
            }
            (destination / "driver.py").write_bytes(
                files("roboenv").joinpath("assets/device-client.py").read_bytes()
            )
            (destination / "connection.json").write_text(json.dumps(connection) + "\n")
        (destination / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
