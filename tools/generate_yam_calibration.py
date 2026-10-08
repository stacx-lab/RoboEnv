"""Generate incomplete public Isaac YAM profiles from the shared ABC hardware catalog."""

import argparse
import copy
from pathlib import Path

import yaml
from roboenv.devices.export import driver_access, write_driver
from roboenv_runtime.adapters.isaaclab.tasks.yam_calibration.task import TASKS, robot_name

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = {
    "yam-calibrate-overhead-camera": """Calibrate camera_top extrinsics
using eye-to-hand camera calibration.
A 7 x 5 ChArUco board (DICT_4X4_50, squares 0.03 m, markers 0.0225 m)
is rigidly attached to the left-arm flange, with its printed face along flange -Z.
Move the left arm through diverse board-visible poses and pair images with measured
left_arm flange poses in the arm base frame. The board attachment need not be known
to the hand-eye solver. Compose the solved camera-in-base transform with the known
left_arm world mount and save camera_top's world-relative extrinsics.
Keep the right arm parked. Retain observations and solver evidence in the workspace.
This is ungraded practice: save the profile locally; there is no result submission
or automatic completion. Camera reads and joint reads are sequential, not synchronized.""",
    "yam-calibrate-wrist-cameras": """Calibrate BOTH camera_left and
camera_right extrinsics in this exercise.
A fixed, upward-facing 7 x 5 ChArUco board (DICT_4X4_50, squares 0.03 m,
markers 0.0225 m) is on the table. Both wrist-camera extrinsics start unknown.
For each camera, collect diverse views paired with its own arm's measured flange
pose in that arm's base frame, then use eye-in-hand calibration. Save camera_left
relative to left_arm/flange and camera_right relative to right_arm/flange.
Collect independently or interleave the two datasets; synchronous captures and
simultaneous arm motion are not required. Keep the other arm parked while moving.
Retain observations and solver evidence. This is ungraded practice: update both
workspace profiles locally, with no submission or automatic completion.
Camera reads and joint reads are sequential, not synchronized.""",
    "yam-calibrate-dual-arm": """Calibrate right_arm's base relative
to left_arm's base using dual-arm alignment.
Both grippers carry pointed probes whose tips coincide with their calibrated TCPs.
Keep the grippers open while using these rigid attachments.
Four fixed fiducials A (red), B (green), C (blue), D (yellow) have noncoplanar top
centers. Touch the same top-center point of each fiducial with each probe, recording
TCP positions in the respective arm BASE frame. Use left_contact/right_contact
force readings as simulated contact confirmation; they report fixture-filtered
normal forces on the probe-bearing flange in the gripper TCP axes, not contact
positions or hidden transforms.
Only one arm moves at a time. Park the other arm and retract before switching.
Fit second-base-in-anchor-base and save right_arm extrinsics with parent left_arm,
parent_frame left_arm. Gripper mounts and TCPs are already calibrated; do not
calibrate individual world mounts. Retain contact observations and solver evidence.
This is ungraded practice: save the workspace profile, with no submission or
automatic completion. Contact feedback confirms contact, not exact centering.""",
}


def contact_profile(side):
    return {
        "name": side + "_contact",
        "type": "force_torque",
        "intrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "measurement": {
                "quantity": "contact_normal",
                "target": {"device": side + "_gripper", "frame": "tcp"},
                "force": {
                    "axes": ["x", "y", "z"],
                    "range": None,
                    "resolution": None,
                    "bias": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "torque": None,
            },
            "sample_rate_hz": 1 / 0.034,
        },
        "extrinsics": {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "mount": {
                "parent": side + "_gripper",
                "parent_frame": "tcp",
                "translation_m": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
            },
        },
        "access": driver_access(
            "force_torque",
            side + "_contact",
            ["load", "unload", "status", "read"],
            "http://127.0.0.1:8123",
        ),
        "meta": {"created_at": "2026-10-07T00:00:00Z", "updated_at": "2026-10-07T00:00:00Z"},
    }


def generate(catalog, bindings_out):
    source = ROOT / "catalog/robots/abc/abc-yam"
    originals = {
        p.parent.name: yaml.safe_load(p.read_text())
        for p in (source / "devices").glob("*/profile.yaml")
    }
    for task_name, unknowns in TASKS.items():
        name = robot_name(task_name)
        robot = Path(catalog) / "robots/isaaclab" / name
        robot.mkdir(parents=True, exist_ok=True)
        workspace = yaml.safe_load((source / "workspace.yaml").read_text())
        workspace["identity"].update(id=name, name=name, model="abc-yam")
        workspace["sim"]["simulator"] = "isaaclab"
        (robot / "workspace.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
        profiles = copy.deepcopy(originals)
        if task_name == "yam-calibrate-dual-arm":
            profiles["right_arm"]["extrinsics"]["mount"].update(
                parent="left_arm", parent_frame="left_arm"
            )
            for side in ("left", "right"):
                profiles[side + "_contact"] = contact_profile(side)
                profiles[side + "_gripper"]["intrinsics"]["geometry"]["description"] += (
                    " Calibration attachment: a rigid pointed probe terminates at the TCP."
                )
        for device in unknowns:
            extrinsics = profiles[device]["extrinsics"]
            extrinsics.update(
                source="unknown",
                method="unknown",
                calibrated_at=None,
                needs_calibration="Estimate this mount from calibration observations.",
            )
            extrinsics["mount"].update(translation_m="???", rotation_xyzw="???")
        for device, profile in sorted(profiles.items()):
            directory = robot / "devices" / device
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
            write_driver(directory, profile["type"])
        bindings = {
            "schema_version": "roboenv/bindings/1",
            "backend": "isaaclab",
            "embodiment": "yam",
            "devices": {n: {"binding": "isaaclab.yam/1", "entity": n} for n in sorted(profiles)},
        }
        path = Path(bindings_out) / "isaaclab" / (name + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(bindings, sort_keys=False))
        task = Path(catalog) / "tasks/isaaclab" / task_name / "task.yaml"
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(
            yaml.safe_dump(
                {
                    "id": "isaaclab/" + task_name,
                    "robot": "isaaclab/" + name,
                    "instruction": INSTRUCTIONS[task_name],
                    "seed": 0,
                    "budget_mode": "practice",
                    "execution_mode": "realtime",
                    "stop_on_success": False,
                    "backend_kwargs": {
                        "task_name": task_name,
                        "asset_manifest": ".cache/isaaclab/yam/assets.json",
                    },
                },
                sort_keys=False,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog")
    parser.add_argument(
        "--bindings-out", type=Path, default=ROOT / "src/roboenv/devices/bindings/defaults"
    )
    args = parser.parse_args()
    generate(args.catalog, args.bindings_out)
