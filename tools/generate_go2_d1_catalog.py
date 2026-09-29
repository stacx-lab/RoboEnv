"""Generate the portable Go2–D1 practice rig from a prepared, task-free URDF."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILE_TIMESTAMP = "2026-09-23T00:00:00Z"
GO2_ARM_READING = """def kinex_reading(state):
    frames = state.get("frame_poses", {})
    return {
        **state,
        "frame_poses": {
            name: [*value["translation_m"], *value["rotation_xyzw"]]
            if isinstance(value, dict)
            else value
            for name, value in frames.items()
        },
    }
"""
sys.path.insert(0, str(ROOT / "runtime/src"))
sys.path.insert(0, str(ROOT / "src"))
from roboenv.devices.export import driver_access, write_driver  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import RobotModel  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.profiles import profiles  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import TASKS  # noqa: E402


def adapt_go2_arm_driver(path):
    """Translate Go2's native pose mappings at the Kinex driver boundary."""
    source = path.read_text()
    replacements = (
        (
            "\n\ndef handle(action, args):",
            "\n\n" + GO2_ARM_READING.rstrip() + "\n\n\ndef handle(action, args):",
        ),
        (
            'joint.reading(data["state"])',
            'joint.reading(kinex_reading(data["state"]))',
        ),
        (
            'joint.reading({**data, "frame_poses": data.get("frame_poses", {})})',
            'joint.reading(kinex_reading(data))',
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("The portable joint driver changed; update the Go2 arm adapter")
        source = source.replace(before, after)
    path.write_text(source)


def generate(assets, out, *, bindings_out=None):
    rig = out / "robots/isaaclab/isaaclab-go2-d1"
    rig.mkdir(parents=True, exist_ok=True)
    values = profiles(
        RobotModel(assets / "robot.urdf"), json.loads((assets / "home.json").read_text())
    )
    workspace = {
        "identity": {
            "id": "go2-d1",
            "name": "Go2 D1",
            "kind": "quadruped",
            "model": "Unitree Go2 + D1-550",
            "platform": "sim",
            "source": "introspected",
        },
        "sim": {"simulator": "isaaclab", "source": "introspected"},
        "devices": {"source": "introspected"},
        "meta": {"created_at": PROFILE_TIMESTAMP, "updated_at": PROFILE_TIMESTAMP},
    }
    (rig / "workspace.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
    for name, value in values.items():
        dest = rig / "devices" / name
        dest.mkdir(parents=True, exist_ok=True)
        descriptor = {
            "name": name,
            **value,
            "meta": {"created_at": PROFILE_TIMESTAMP, "updated_at": PROFILE_TIMESTAMP},
        }
        actions = ["read", "load", "unload", "status"]
        if name == "world":
            descriptor.pop("extrinsics", None)
            actions.append("wait")
        elif not name.startswith("camera_"):
            actions += ["write", "stop"]
        if name == "arm":
            actions += ["fk", "ik"]
        descriptor["access"] = driver_access(
            value["type"], name, actions, "http://127.0.0.1:8123"
        )
        write_driver(dest, value["type"])
        if name == "arm":
            adapt_go2_arm_driver(dest / "driver.py")
        (dest / "profile.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    if bindings_out is not None:
        bindings = bindings_out / "isaaclab/isaaclab-go2-d1.yaml"
        bindings.parent.mkdir(parents=True, exist_ok=True)
        bindings.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "roboenv/bindings/1",
                    "backend": "isaaclab",
                    "embodiment": "go2_d1",
                    "devices": {n: {"binding": "isaaclab.go2-d1/1", "entity": n} for n in values},
                },
                sort_keys=False,
            )
        )
    for name in TASKS:
        task = out / f"tasks/isaaclab/{name}/task.yaml"
        task.parent.mkdir(parents=True, exist_ok=True)
        # Task YAML owns instructions and task settings; regeneration preserves both.
        source = ROOT / f"catalog/tasks/isaaclab/{name}/task.yaml"
        task.write_text(source.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / ".cache/isaaclab/go2-d1")
    p.add_argument("--out", type=Path, default=ROOT / "catalog")
    p.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    a = p.parse_args()
    generate(a.assets, a.out, bindings_out=a.bindings_out)
