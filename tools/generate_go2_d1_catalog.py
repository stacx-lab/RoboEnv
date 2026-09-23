"""Generate the portable Go2–D1 practice rig from a prepared, task-free URDF."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))
sys.path.insert(0, str(ROOT / "src"))
from roboenv.devices.export import driver_access, write_driver  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import RobotModel  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.profiles import profiles  # noqa: E402
from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import (  # noqa: E402
    CONTROL,
    INSTRUCTIONS,
    TASKS,
)


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
    }
    (rig / "workspace.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
    for name, value in values.items():
        dest = rig / "devices" / name
        dest.mkdir(parents=True, exist_ok=True)
        descriptor = {"name": name, **value}
        if name == "world":
            descriptor.pop("extrinsics", None)
        else:
            actions = ["read", "load", "unload", "status"]
            if not name.startswith("camera_"):
                actions += ["write", "stop"]
            if name == "arm":
                actions += ["fk", "ik"]
            descriptor["access"] = driver_access(
                value["type"], name, actions, "http://127.0.0.1:8123"
            )
            write_driver(dest, value["type"])
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
        task.write_text(
            yaml.safe_dump(
                {
                    "id": f"isaaclab/{name}",
                    "backend": "isaaclab",
                    "backend_task": f"isaaclab.{name}",
                    "robot": "isaaclab/isaaclab-go2-d1",
                    "instruction": INSTRUCTIONS[name],
                    "seed": 0,
                    "budget_mode": "practice",
                    "action_abstraction": "devices",
                    "backend_kwargs": {
                        "task_name": name,
                        "asset_manifest": ".cache/isaaclab/go2-d1/assets.json",
                        "realtime": False,
                        "command_timeout_s": 0.5,
                        "max_steps": 1500 if name == CONTROL else 6000,
                    },
                },
                sort_keys=False,
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT / ".cache/isaaclab/go2-d1")
    p.add_argument("--out", type=Path, default=ROOT / "catalog")
    p.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    a = p.parse_args()
    generate(a.assets, a.out, bindings_out=a.bindings_out)
