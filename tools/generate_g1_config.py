#!/usr/bin/env python3
"""Generate a G1 catalog from a verified local USD/policy bundle on CPU."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "runtime/src")]
from roboenv.devices.export import driver_access, write_driver  # noqa: E402
from roboenv_runtime.adapters.isaaclab.g1_assets import digest, load_bundle  # noqa: E402
from roboenv_runtime.adapters.isaaclab.g1_cameras import specs as camera_specs  # noqa: E402
from roboenv_runtime.adapters.isaaclab.g1_profiles import profiles  # noqa: E402

TASK_ROOT = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/"
SOURCE_FILES = [
    TASK_ROOT + "locomanipulation_g1_env_cfg.py",
    TASK_ROOT + "mdp/actions.py",
    TASK_ROOT + "configs/agile_locomotion_observation_cfg.py",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
]
META = {"created_at": "2026-09-13T00:00:00Z", "updated_at": "2026-09-13T00:00:00Z"}


def generate(assets, out, *, facts_out=None, provenance=None, bindings_out=None):
    from g1_kinematics import extract
    from pxr import Usd

    _, paths = load_bundle(assets)
    source = ROOT / "third_party/IsaacLab"
    model = {
        "asset_manifest_sha256": digest(assets),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source_files": {name: digest(source / name) for name in SOURCE_FILES},
        "cameras": camera_specs(),
        "parts": extract(Usd.Stage.Open(paths["robot"])),
        # Conservative practice command caps, not a certified policy envelope.
        "control": {
            "mode": "motion",
            "body_frame": "pelvis",
            "modes": {
                "locomotion": {
                    "velocity": {
                        "frame": "body",
                        "components": [
                            {
                                "axis": axis,
                                "units": "rad/s" if axis == "wz" else "m/s",
                                "bounds": {"min": -cap, "max": cap},
                            }
                            for axis, cap in (("vx", 0.25), ("vy", 0.15), ("wz", 0.3))
                        ],
                    },
                    "posture": {
                        "reference": "ground",
                        "components": [
                            {
                                "axis": "height",
                                "units": "m",
                                "bounds": {"min": 0.65, "max": 0.8},
                                "default": 0.75,
                            }
                        ],
                    },
                }
            },
        },
    }
    robot = out / "robots/isaaclab/isaaclab-g1"
    robot.mkdir(parents=True, exist_ok=True)
    model_text = json.dumps(model, indent=2) + "\n"
    if facts_out is not None:
        facts_out.parent.mkdir(parents=True, exist_ok=True)
        facts_out.write_text(model_text)
    if provenance is not None:
        receipt = {
            "source_revision": model["source_revision"],
            "source_files": model["source_files"],
            "asset_manifest_sha256": model["asset_manifest_sha256"],
            "generator_sha256": digest(Path(__file__)),
            "kinematics_generator_sha256": digest(ROOT / "tools/g1_kinematics.py"),
            "method": "CPU USD joint chains, limits and wrist-to-palm mounts; configured RGB-D rig",
            "gpu_validated": False,
        }
        provenance.parent.mkdir(parents=True, exist_ok=True)
        provenance.write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = {
        "identity": {
            "id": "g1",
            "name": "G1",
            "kind": "humanoid",
            "model": "Unitree G1 TriHand",
            "platform": "sim",
            "source": "introspected",
        },
        "sim": {
            "simulator": "isaaclab",
            "source": "introspected",
        },
        "devices": {"source": "introspected"},
        "safety": {"max_cartesian_velocity": None, "max_joint_velocity": None},
        "meta": META,
    }
    (robot / "workspace.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    bindings = {}
    for name, value in profiles(model).items():
        directory = robot / "devices" / name
        directory.mkdir(parents=True, exist_ok=True)
        profile = {"name": name, "loaded": False, **value, "meta": META}
        if name != "world":
            actions = ["read", "load", "unload", "status"]
            if value["type"] != "rgbd":
                actions += ["write", "stop"]
            if value["type"] == "joint":
                actions += ["ik", "fk"]
            profile["access"] = driver_access(value["type"], name, actions, "http://127.0.0.1:8123")
            write_driver(directory, value["type"])
        (directory / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
        bindings[name] = {"binding": "isaaclab.device/1", "entity": name}
        if name.endswith("hand"):
            (directory / "postures.json").write_text(
                json.dumps(postures(model["parts"][name], name), indent=2) + "\n"
            )
            (directory / "driver-resources.json").write_text(
                json.dumps(
                    {
                        "schema": "roboenv/driver-resources/1",
                        "files": ["driver.py", "postures.json"],
                    }
                )
                + "\n"
            )
    if bindings_out is not None:
        binding = bindings_out / "isaaclab/isaaclab-g1.yaml"
        binding.parent.mkdir(parents=True, exist_ok=True)
        binding.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "roboenv/bindings/1",
                    "backend": "isaaclab",
                    "embodiment": "g1",
                    "devices": bindings,
                },
                sort_keys=False,
            )
        )
    task = out / "tasks/isaaclab/g1-pick-place"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "isaaclab/g1-pick-place",
                "backend": "isaaclab",
                "backend_task": "isaaclab.g1-pick-place",
                "robot": "isaaclab/isaaclab-g1",
                "seed": 0,
                "instruction": "Pick up and place the steering wheel using the G1 humanoid.",
                "budget_mode": "official",
                "backend_kwargs": {
                    "task_name": "g1-pick-place",
                    "asset_manifest": portable(assets),
                },
            },
            sort_keys=False,
        )
    )
    return robot


def portable(path):
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)


def postures(part, name):
    # Seed postures follow the G1 TriHand controller mapping. They are bounded
    # joint configurations, not validated grasps for a particular object.
    side = name.removesuffix("_hand")
    sign = -1 if side == "left" else 1
    opened = {j["name"]: 0.0 for j in part["joints"]}
    result = {"open": opened}
    for label, index, middle in (("power", 1.0, 1.0), ("pinch", 1.0, 0.0)):
        values = {
            **opened,
            f"{side}_hand_thumb_0_joint": (0.5 * index - 0.5 * middle)
            * (-1 if side == "right" else 1)
            * sign,
            f"{side}_hand_thumb_1_joint": -0.4 * max(index, middle) * sign,
            f"{side}_hand_thumb_2_joint": -0.7 * max(index, middle) * sign,
        }
        for finger, closure in (("index", index), ("middle", middle)):
            for joint in (0, 1):
                values[f"{side}_hand_{finger}_{joint}_joint"] = closure * sign
        if set(values) != set(opened):
            raise ValueError("TriHand mapping differs from asset joint names")
        for joint in part["joints"]:
            if not joint["limits"]["min"] <= values[joint["name"]] <= joint["limits"]["max"]:
                raise ValueError(
                    f"Seed posture {label} exceeds {joint['name']}; "
                    "supply an asset-matched grasp mapping"
                )
        result[label] = values
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, required=True, help="Verified local roboenv/g1-assets/1 manifest"
    )
    parser.add_argument("--out", type=Path, required=True, help="Generated catalog directory")
    parser.add_argument("--facts-out", type=Path, help="Optional runtime model output")
    parser.add_argument("--provenance", type=Path, help="Optional CPU generation receipt")
    parser.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    options = parser.parse_args()
    print(
        generate(
            options.assets,
            options.out,
            facts_out=options.facts_out,
            provenance=options.provenance,
            bindings_out=options.bindings_out,
        )
    )
