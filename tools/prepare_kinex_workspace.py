"""Copy a resolved device catalog into a new current-format Kinex workspace."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from roboenv.tasks.resolve import resolve_task


def prepare(task, destination, *, project=None, catalog=None):
    resolved = resolve_task(task, project=project, catalog=catalog)
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
                for support in source.glob("*.py"):
                    if support.is_symlink():
                        raise ValueError("Device support files must not be symlinks")
                    shutil.copy2(support, device / support.name)
        (staging / "instruction.md").write_text(resolved.task.instruction + "\n")
        (staging / "AGENTS.md").write_text(
            "# Robot workspace\n\n" + resolved.task.instruction + "\n\n"
            "Use the declared device interfaces and camera observations. Manipulator writes "
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
    args = parser.parse_args()
    print(json.dumps(prepare(args.task, args.out, catalog=args.catalog)))
