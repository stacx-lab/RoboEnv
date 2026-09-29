"""Packaged Kinex validates every exported profile, including invalid ones."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from helpers import kinex_sources, load_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task,count", [("simple/g1-xmove-pick", 8), ("libero/bowl-onto-plate", 5)])
def test_packaged_rig_validation_does_not_silently_drop_invalid_profiles(tmp_path, task, count):
    kinex, _ = kinex_sources()
    node = shutil.which(os.environ.get("NODE", "node"))
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("Requires built Kinex and Node >=22.19")
    workspace = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare(task, workspace, project=ROOT)

    def validate():
        return subprocess.run(
            [node, str(ROOT / "tests/native_rigs_kinex.mjs"), str(kinex), str(workspace)],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "KINEX_CONFIG_DIR": str(tmp_path / "config")},
        )

    checked = validate()
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["results"][0]["profiles"] == count
    arm = "left_arm" if task.startswith("simple") else "arm"
    profile = workspace / ".kinex/devices" / arm / "profile.yaml"
    value = yaml.safe_load(profile.read_text())
    value["type"] = "invalid-device-type"
    profile.write_text(yaml.safe_dump(value))
    assert validate().returncode != 0
