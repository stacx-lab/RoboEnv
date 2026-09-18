"""Copy a resolved device catalog into a new current-format Kinex workspace."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from roboenv.devices.export import driver_resources, shared_driver_resources
from roboenv.tasks.resolve import resolve_task


def prepare(task, destination, *, project=None, catalog=None, realtime=None):
    resolved = resolve_task(task, project=project, catalog=catalog, realtime=realtime)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError("Select a new workspace; existing files are preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".workspace-") as temporary:
        staging = Path(temporary)
        config = staging / ".kinex"
        config.mkdir()
        shutil.copy2(resolved.robot / "workspace.yaml", config / "workspace.yaml")
        for spec in resolved.bindings.devices:
            device = config / "devices" / spec.name
            device.mkdir(parents=True)
            profile = spec.resolved_profile(resolved.robot)
            (device / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
            if profile["type"] != "world":
                source = resolved.robot / Path(spec.profile).parent
                for support in shared_driver_resources(source.parent):
                    target = device.parent / support
                    content = (source.parent / support).read_bytes()
                    if target.exists() and target.read_bytes() != content:
                        raise ValueError("Selected devices require different shared transports")
                    if not target.exists():
                        target.write_bytes(content)
                for support in driver_resources(source):
                    target = device / support
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / support, target)
        (staging / "instruction.md").write_text(resolved.task.instruction + "\n")
        (staging / "AGENTS.md").write_text(
            "# Robot workspace\n\nRead instruction.md for the current episode's public task.\n\n"
            "Use the declared device interfaces and camera observations. JointDevice writes "
            "accept absolute joint positions; acceptance does not imply arrival. "
            "Stop holds a device. The environment owner controls the episode lifecycle.\n"
        )
        # Publish only after every selected profile and driver has been copied.
        shutil.copytree(staging, destination)
    return {
        "workspace": str(destination),
        "task": resolved.task.id,
        "robot": resolved.task.robot,
        "devices": [s.name for s in resolved.bindings.devices],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    clock = parser.add_mutually_exclusive_group()
    clock.add_argument("--realtime", dest="realtime", action="store_true")
    clock.add_argument("--sync", dest="realtime", action="store_false")
    parser.set_defaults(realtime=None)
    args = parser.parse_args()
    print(json.dumps(prepare(args.task, args.out, catalog=args.catalog, realtime=args.realtime)))
