"""Isaac Lab discovery and frame conventions without simulator imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.geometry import relative_pose, rotation

ROOT = Path(__file__).resolve().parents[1]


def test_frame_and_camera_conventions():
    assert np.allclose(rotation((-0.5, 0.5, -0.5, 0.5)), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    parent = [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    child = [1, 3, 3, *parent[3:]]
    assert np.allclose(relative_pose(parent, child), [1, 0, 0, 0, 0, 0, 1])


def test_isaaclab_discovery_does_not_import_simulator():
    code = """
import sys
from roboenv_runtime.core.adapter import BUILTIN_ADAPTERS
from roboenv_runtime.adapters.isaaclab import IsaacLabAdapter
from roboenv.tasks.catalog import _isaaclab
from pathlib import Path
assert 'isaaclab' in BUILTIN_ADAPTERS
assert IsaacLabAdapter.embodiment == 'g1'
assert {task.id for task, _ in _isaaclab(Path('.'), [])} == {'isaaclab/g1-pick-place'}
assert 'isaaclab' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'runtime/src'}"},
    )


def test_removed_go2_joint_demo_is_not_discoverable():
    with pytest.raises(ValueError):
        resolve_task("isaaclab/go2-posture", project=ROOT)
    assert not (ROOT / "catalog/robots/isaaclab/isaaclab-go2/workspace.yaml").exists()


@pytest.mark.parametrize("realtime", [False, True])
def test_g1_clock_selection_updates_episode_profiles_only(realtime, tmp_path):
    import yaml
    from helpers import load_tool

    profile = ROOT / "catalog/robots/isaaclab/isaaclab-g1/devices/base/profile.yaml"
    original = profile.read_bytes()
    resolved = resolve_task("isaaclab/g1-pick-place", project=ROOT, realtime=realtime)
    assert resolved.task.backend_kwargs["realtime"] is realtime
    base = next(spec for spec in resolved.bindings.devices if spec.name == "base")
    execution = base.resolved_profile(resolved.robot)["intrinsics"]["control"]["execution"]
    assert execution == (
        {
            "mode": "realtime",
            "command_timeout_s": resolved.task.backend_kwargs.get("command_timeout_s", 0.5),
        }
        if realtime
        else {"mode": "stepped", "interval_s": 0.02}
    )
    assert profile.read_bytes() == original
    destination = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare(
        "isaaclab/g1-pick-place", destination, project=ROOT, realtime=realtime
    )
    copied = yaml.safe_load((destination / ".kinex/devices/base/profile.yaml").read_text())
    assert copied["intrinsics"]["control"]["execution"] == execution


@pytest.mark.parametrize("timeout", [None, 0.2])
def test_g1_task_timeout_reaches_resolved_profiles(tmp_path, timeout):
    from dataclasses import asdict

    import yaml

    task = asdict(resolve_task("isaaclab/g1-pick-place", project=ROOT).task)
    task["backend_kwargs"].update(realtime=True, command_timeout_s=timeout)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    resolved = resolve_task(path, project=ROOT)
    base = next(spec for spec in resolved.bindings.devices if spec.name == "base")
    assert base.resolved_profile(resolved.robot)["intrinsics"]["control"]["execution"] == {
        "mode": "realtime",
        "command_timeout_s": timeout,
    }
