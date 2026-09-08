#!/usr/bin/env python3
"""Extract the pinned Go2 USD catalog on CPU (requires usd-core and PyYAML).

Default output is catalog/. --out writes a separate catalog; neither mode boots
Kit or steps physics. Upstream initialization is read as literal AST data.
"""

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))
sys.path.insert(0, str(ROOT / "src"))
from roboenv.provisioning.isaaclab_assets import verify_assets  # noqa: E402
from roboenv_runtime.adapters.isaaclab.rig import (  # noqa: E402
    APERTURE,
    CAMERA_POSITION,
    CAMERA_ROTATION,
    CAMERA_SIZE,
    FEET,
    FOCAL_LENGTH,
    JOINTS,
)

META = {"created_at": "2026-09-07T00:00:00Z", "updated_at": "2026-09-07T00:00:00Z"}
CONVENTION = (
    "child-in-parent: pose of this device's frame expressed in the parent frame "
    "(p_parent = R·p_child + t); quaternion [x,y,z,w]; camera frames are OpenCV "
    "optical (+X right, +Y down, +Z forward)"
)


def extract(project):
    import re

    root = verify_assets(project)
    stage = Usd.Stage.Open(str(root / "go2.usd"))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Expected metric Go2 USD")
    joints = {
        p.GetName(): UsdPhysics.RevoluteJoint(p)
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.RevoluteJoint)
    }
    source = (
        project / "third_party/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
    )
    tree = ast.parse(source.read_text())
    cfg = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "UNITREE_GO2_CFG" for t in n.targets)
    )
    init = next(k.value for k in cfg.keywords if k.arg == "init_state")
    positions = ast.literal_eval(next(k.value for k in init.keywords if k.arg == "joint_pos"))
    home = [
        next(value for pattern, value in positions.items() if re.fullmatch(pattern, name))
        for name in JOINTS
    ]
    limits = [
        [
            math.radians(joints[name].GetLowerLimitAttr().Get()),
            math.radians(joints[name].GetUpperLimitAttr().Get()),
        ]
        for name in JOINTS
    ]
    imu = stage.GetPrimAtPath("/go2_description/base/imu")
    imu_position = list(UsdGeom.Xformable(imu).GetLocalTransformation().ExtractTranslation())
    posture = [
        q + (0.1 if name.endswith("thigh_joint") else -0.2 if name.endswith("calf_joint") else 0)
        for name, q in zip(JOINTS, home, strict=True)
    ]
    return {
        "limits": limits,
        "home": home,
        "posture": posture,
        "imu_position": imu_position,
    }, source


def generate(out, model):
    robot = out / "robots/isaaclab-go2"
    profiles = {}
    joints = [
        {"name": name, "type": "revolute", "limits": {"min": lo, "max": hi}}
        for name, (lo, hi) in zip(JOINTS, model["limits"], strict=True)
    ]
    profiles["body"] = {
        "type": "body",
        "intrinsics": {
            "source": "introspected",
            "mobility": "mobile",
            "drive": "legged",
            "joints": joints,
            "home_joints": model["home"],
            "actions": [
                {"joint": j["name"], "units": "rad", "bounds": dict(j["limits"])} for j in joints
            ],
        },
        "extrinsics": {"source": "introspected", "frames": list(FEET)},
    }

    def mount(pos=(0, 0, 0), rot=(0, 0, 0, 1), frame=None):
        return {
            "source": "introspected",
            "mount": {
                "parent": "body",
                **({"parent_frame": frame} if frame else {}),
                "pose": {"translation": list(pos), "rotation": list(rot)},
                "convention": CONVENTION,
            },
        }

    focal = CAMERA_SIZE * FOCAL_LENGTH / APERTURE
    profiles["camera_front"] = {
        "type": "rgbd",
        "intrinsics": {
            "source": "introspected",
            "value": {
                "width": CAMERA_SIZE,
                "height": CAMERA_SIZE,
                "fx": focal,
                "fy": focal,
                "cx": CAMERA_SIZE / 2,
                "cy": CAMERA_SIZE / 2,
            },
            "depth": {"scale": 0.001, "aligned": True},
        },
        "extrinsics": mount(CAMERA_POSITION, CAMERA_ROTATION),
    }
    profiles["imu"] = {
        "type": "imu",
        "intrinsics": {
            "source": "introspected",
            "notes": (
                "Native synthetic IMU: sensor-frame angular_velocity in rad/s and "
                "linear_acceleration in m/s^2; proper acceleration includes +g at rest. No "
                "world position or linear velocity."
            ),
        },
        "extrinsics": mount(model["imu_position"]),
    }
    for foot in FEET:
        profiles[f"contact_{foot.lower()}"] = {
            "type": "other",
            "intrinsics": {
                "source": "introspected",
                "notes": (
                    "Native synthetic contact sensor; force is the net contact force in newtons "
                    "expressed in the foot frame. No torque measurement."
                ),
            },
            "extrinsics": mount(frame=foot),
        }
    profiles["world"] = {
        "type": "world",
        "intrinsics": {
            "source": "default",
            "bounds": {"min": [None] * 3, "max": [None] * 3},
            "surfaces": [],
        },
    }
    bindings = {}
    driver = (ROOT / "src/roboenv/assets/device-client.py").read_bytes()
    for name, profile in profiles.items():
        profile.update(name=name, meta=META)
        if name != "world":
            actions = ["read", "load", "unload", "status"] + (
                ["write", "stop"] if name == "body" else []
            )
            profile["access"] = {
                "source": "user",
                "kind": "url",
                "url": "http://127.0.0.1:8123",
                "driver": {
                    "entry": "driver.py", "interpreter": "python3",
                    "persistent": False, "actions": actions,
                },
                **({"control_mode": "joint_position"} if name == "body" else {}),
            }
        directory = robot / "devices" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
        if name != "world":
            (directory / "driver.py").write_bytes(driver)
        bindings[name] = {
            "binding": "isaaclab.device/1",
            "entity": name,
            "profile": f"devices/{name}/profile.yaml",
        }
        if name.startswith("contact_"):
            bindings[name]["entity"] = profile["extrinsics"]["mount"]["parent_frame"]
    manifest = {
        "identity": {
            "source": "introspected",
            "id": "isaaclab-go2",
            "name": "Unitree Go2 (Isaac Lab)",
            "platform": "sim",
            "kind": "quadruped",
            "model": "Unitree Go2",
        },
        "sim": {"source": "introspected", "simulator": "isaaclab"},
        "devices": {"source": "introspected"},
        "safety": {
            "source": "default",
            "max_cartesian_velocity": None,
            "max_joint_velocity": None,
            "keep_out_zones": [],
            "notes": (
                "Absolute joint targets through native DC motor control; no walking or "
                "balance policy. Free base. Reads and stop do not step. Camera and contact "
                "devices are synthetic sensors, not hardware sensor replicas."
            ),
        },
        "meta": META,
    }
    (robot / "workspace.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (out / "bindings").mkdir(parents=True, exist_ok=True)
    (out / "bindings/isaaclab-go2.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "roboenv/bindings/1",
                "backend": "isaaclab",
                "embodiment": "go2",
                "devices": bindings,
            },
            sort_keys=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "catalog")
    parser.add_argument("--facts-out", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    model, source = extract(ROOT)
    generate(args.out, model)
    if args.facts_out:
        args.facts_out.write_text(json.dumps(model, indent=2) + "\n")
    if args.provenance:
        receipt = {
            "upstream": json.loads((ROOT / "upstreams.lock.json").read_text())["sources"][
                "isaaclab_native"
            ]["revision"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rig_sha256": hashlib.sha256(
                (ROOT / "runtime/src/roboenv_runtime/adapters/isaaclab/rig.py").read_bytes()
            ).hexdigest(),
            "assets": json.loads((ROOT / "src/roboenv/assets/isaaclab-go2.json").read_text()),
            "method": (
                "CPU USD joint limits and rigid IMU mount; literal upstream initialization; "
                "explicit synthetic camera mount"
            ),
            "gpu_validated": False,
        }
        args.provenance.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
