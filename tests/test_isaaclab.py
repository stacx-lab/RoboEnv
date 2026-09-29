"""Isaac Lab discovery and frame conventions without simulator imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_g1_resolution_leaves_profiles_alone(tmp_path):
    import yaml
    from helpers import load_tool

    profile = ROOT / "catalog/robots/isaaclab/isaaclab-g1/devices/base/profile.yaml"
    original = profile.read_bytes()
    resolved = resolve_task("isaaclab/g1-pick-place", project=ROOT)
    assert "realtime" not in resolved.task.backend_kwargs
    base = next(spec for spec in resolved.bindings.devices if spec.name == "base")
    # Nothing about timing is declared any more: a write reports the device time
    # it spent, so resolving an episode leaves the control block untouched.
    assert "execution" not in base.resolved_profile(resolved.robot)["intrinsics"]["control"]
    assert profile.read_bytes() == original
    destination = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare(
        "isaaclab/g1-pick-place", destination, project=ROOT
    )
    copied = yaml.safe_load((destination / ".kinex/devices/base/profile.yaml").read_text())
    assert "execution" not in copied["intrinsics"]["control"]
