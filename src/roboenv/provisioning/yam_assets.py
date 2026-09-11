"""Reproduce the YAM arm description from the public, pinned manufacturer source."""

import copy
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

import yaml
from roboenv.provisioning.assets import inventory

PREFIX = "i2rt/robot_models/arm/yam/v1/"
STATION = "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf"


def source_inventory(directory):
    return [
        entry
        for entry in inventory(directory)
        if entry["path"] != "YAM.usd"
        and not entry["path"].startswith(("native-usd/", "native-usd-history/"))
    ]


def verify_source(directory, *, source_cache):
    """Reconstruct expected pinned geometry without overwriting an existing depot."""
    with tempfile.TemporaryDirectory(prefix="verify-yam-") as temporary:
        expected = Path(temporary) / "yam"
        prepare_source(expected, source_cache=source_cache)
        if source_inventory(directory) != inventory(expected):
            raise ValueError("Existing YAM geometry differs from pinned manufacturer inputs")
    return True


def prepare_source(destination, *, source_cache):
    """Download verified source inputs and create a new data-only robot directory.

    The camera link uses the manufacturer's optical frame. Native conversion and
    physical validation are separate; this function never imports Isaac or accepts
    a license. The original visual meshes become native collision geometry during
    explicit USD conversion, with that choice recorded in the conversion receipt.
    """
    destination, cache = Path(destination), Path(source_cache)
    if destination.exists():
        raise ValueError("YAM output exists; preserve it and select a new destination")
    manifest = json.loads(files("roboenv").joinpath("assets/yam-source.json").read_text())
    for relative, expected in manifest["files"].items():
        path = cache / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                pending = Path(temporary) / "download"
                url = f"https://raw.githubusercontent.com/i2rt-robotics/i2rt/{manifest['revision']}/{relative}"
                with urlopen(url, timeout=60) as response:
                    pending.write_bytes(response.read())
                verify_blob(pending, expected)
                pending.rename(path)
        verify_blob(path, expected)
    tree = ET.parse(cache / PREFIX / "yam.urdf")
    robot = tree.getroot()
    station = ET.parse(cache / STATION).getroot()
    camera_joint = next(
        joint
        for joint in station.findall("joint")
        if joint.find("parent").get("link") == "left_gripper"
        and "camera" in joint.find("child").get("link")
    )
    camera_joint = copy.deepcopy(camera_joint)
    camera_joint.set("name", "wrist_camera_joint")
    camera_joint.find("parent").set("link", "gripper")
    camera_joint.find("child").set("link", "camera")
    robot.append(camera_joint)
    # A sensor frame has no collision body or task-specific resources.
    ET.SubElement(robot, "link", name="camera")
    gripper_joint = next(joint for joint in robot.findall("joint") if joint.get("name") == "joint7")
    lower = float(gripper_joint.find("limit").get("lower"))
    upper = float(gripper_joint.find("limit").get("upper"))
    config = {
        "urdf_path": "YAM.urdf",
        "ee_joints": "joint6",
        "ee_link": "gripper",
        "ee_type": "gripper",
        "base_link": "base",
        "arm_joints_name": [f"joint{i}" for i in range(1, 7)],
        "gripper_joints_name": ["joint7", "joint8"],
        "save_gripper_joints_name": [f"joint{i}" for i in range(1, 9)],
        # Manufacturer fingers meet at q=0; negative travel opens the jaws.
        "gripper_move": {"base": "joint7", "sign": -1.0, "mimic": ["joint8", 1.0, 0.0]},
        "gripper_scale": [lower, upper],
        # Typed control uses the measured flange; no approximate planner TCP.
        "gripper_bias": 0.0,
        "dual_arm": False,
        "delta_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "global_trans_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "grasp_camera_reference_axis": [0, 0, -1],
        "camera": [
            {
                "link": "camera",
                "name": "cam_wrist",
                "type": "d435",
                "mesh": "pinhole",
                "pos": [0, 0, 0],
                "ori": [180, 0, 0],
            }
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="yam-") as temporary:
        output = Path(temporary) / "yam"
        output.mkdir()
        for mesh in robot.findall(".//mesh"):
            relative = mesh.get("filename")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((cache / PREFIX / relative).read_bytes())
        tree.write(output / "YAM.urdf", encoding="utf-8", xml_declaration=True)
        (output / "robot_config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        (output / "LICENSE").write_bytes((cache / "LICENSE").read_bytes())
        manifest["adaptation"] = {
            "camera": "manufacturer wrist optical frame; RoboDojo simulated d435 pinhole",
            "controller": "absolute joints; planner disabled for target arms",
            "native_conversion_verified": False,
        }
        (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)
    return {"directory": str(destination), "revision": manifest["revision"]}


def verify_blob(path, expected):
    data = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if len(data) != expected["bytes"] or digest != expected["git_blob_sha1"]:
        raise ValueError("YAM source file differs from its pinned Git blob")
