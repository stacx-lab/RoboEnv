"""Explicit public G1 service and exported Kinex controller integration check."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import roboenv
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = options.out.resolve() / "workspace"
    # Refuse to overwrite a workspace that may contain user edits.
    shutil.copytree(options.catalog / "robots/isaaclab/isaaclab-g1", workspace / ".kinex")
    shutil.copytree(root / "integrations/kinex/g1-motion", workspace / ".kinex/tool/g1-motion")
    for file in (workspace / ".kinex/devices").glob("*/profile.yaml"):
        profile = yaml.safe_load(file.read_text())
        if "access" in profile:
            profile["access"]["driver"]["interpreter"] = sys.executable
            file.write_text(yaml.safe_dump(profile, sort_keys=False))
    with roboenv.launch(
        task="isaaclab/g1-pick-place",
        catalog=options.catalog,
        interpreter=options.interpreter,
        project=root,
        timeout=180,
    ) as run:
        with roboenv.connect(run.endpoint) as env:
            stamp = env.device("base").read()["sample_time_s"]
            cameras = {}
            for name in ("camera_torso", "camera_scene"):
                state = env.device(name).read()
                assert state["sample_time_s"] == state["depth_sample_time_s"] == stamp
                assert (state["width"], state["height"]) == (512, 512)
                for key, suffix in (("file", "png"), ("depth_file", "pgm")):
                    source = Path(state[key])
                    magic = b"\x89PNG" if suffix == "png" else b"P5\n"
                    assert source.read_bytes().startswith(magic)
                    destination = options.out / f"{name}.{suffix}"
                    shutil.copyfile(source, destination)
                    state[key] = str(destination)
                cameras[name] = state
            assert env.device("base").read()["sample_time_s"] == stamp
            (options.out / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
        subprocess.run(
            [
                "bun",
                str(root / "tests/g1_kinex_smoke.ts"),
                str(options.kinex.resolve()),
                str(workspace),
                str(options.out.resolve() / "kinex.json"),
            ],
            env={**os.environ, "ROBOENV_ENDPOINT": run.endpoint},
            check=True,
            timeout=150,
        )
    print("G1 public service and Kinex loop passed", flush=True)


if __name__ == "__main__":
    main()
