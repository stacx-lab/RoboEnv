"""Kinematics runs in exported drivers without a simulator or numerical dependencies."""

import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices"
KINEX_PYTHON = ROOT.parent / "kinex/python"
sys.path.insert(0, str(KINEX_PYTHON))
Kinematics = pytest.importorskip("kinex.kinematics").Kinematics


@pytest.mark.parametrize("name", ["arm", "torso"])
def test_exported_driver_fk_and_seeded_ik_are_offline(name):
    script = '''import json, runpy, sys
from pathlib import Path
import yaml
path = Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
profile = yaml.safe_load(path.with_name('profile.yaml').read_text())
handle = runpy.run_path(str(path))['handle']
def no_network(*args, **kwargs):
    raise AssertionError('Kinematics contacted the simulator')
handle.__globals__['remote'] = no_network
joints = profile['intrinsics']['kinematics']['model']['joints']
home = [profile['intrinsics']['home'][j['name']] for j in joints]
result = handle('fk', {'joints': home})
assert result['ok'], result
solved = handle('ik', {'pose': result['pose'], 'seed': home})
assert solved['ok'], solved
assert all(abs(a-b) < 1e-10 for a,b in zip(solved['joints'], home))
assert not {'numpy', 'mujoco', 'robosuite', 'roboenv', 'roboenv_runtime'} & sys.modules.keys()
print(json.dumps(result))
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROBOT / name / "driver.py")],
                            capture_output=True, text=True, timeout=10,
                            env={**os.environ, "PYTHONPATH": str(KINEX_PYTHON)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"]


@pytest.mark.parametrize("name", ["arm", "torso"])
def test_implicit_seed_reads_once_and_projects_only_the_local_copy(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROBOT / name))
    profile = yaml.safe_load((ROBOT / name / "profile.yaml").read_text())
    kinematics = profile["intrinsics"]["kinematics"]
    solver = Kinematics(kinematics["model"], kinematics["ik"])
    reading = [profile["intrinsics"]["home"][j["name"]] for j in solver.joints]
    reading[0] = solver.upper[0] + 0.001
    target = solver.fk(solver.project(reading))["pose"]
    handle = runpy.run_path(str(ROBOT / name / "driver.py"))["handle"]
    calls = []

    def remote(action, args):
        calls.append((action, args))
        return {"positions": reading}

    handle.__globals__["remote"] = remote
    result = handle("ik", {"pose": target})
    assert result["ok"], result
    assert calls == [("read", {})]
    assert reading[0] > solver.upper[0]
    assert result["joints"] == solver.project(reading)
    assert not handle("ik", {"pose": target, "seed": reading})["ok"]
    assert calls == [("read", {})]


def test_joint_anchor_and_reference_are_applied_before_flange_offset():
    solver = Kinematics({
        "joints": [{"type": "revolute", "limits": {"min": -3, "max": 3}}],
        "chain": [
            {"origin": [1, 0, 0, 0, 0, 0, 1], "joint": 0, "axis": [0, 0, 1],
             "anchor": [0.1, 0, 0], "reference": 0.2},
            {"origin": [0.3, 0, 0, 0, 0, 0, 1]},
        ],
    }, dict(max_iterations=200, damping=0.01,
            position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3))
    target = solver.fk([0.2 + math.pi/2])["pose"]
    assert target[:3] == pytest.approx([1.1, 0.2, 0], abs=1e-12)
    result = solver.ik(target, seed=[0.2])
    assert result["joints"] == pytest.approx([0.2 + math.pi/2], abs=1e-3)
    for joints in ([float("nan")], [True], [], [5]):
        with pytest.raises(ValueError):
            solver.fk(joints)
    with pytest.raises(ValueError, match="unit quaternion"):
        solver.ik([0]*7, seed=[0.2])
    with pytest.raises(ValueError, match="No IK solution"):
        solver.ik([100, 100, 100, 0, 0, 0, 1], seed=[0.2])


@pytest.mark.parametrize("name", ["arm", "torso"])
def test_manipulator_rejects_bad_commands_and_unacknowledged_stop(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROBOT / name))
    handle = runpy.run_path(str(ROBOT / name / "driver.py"))["handle"]
    calls = []

    def remote(action, args):
        calls.append((action, args))
        return {"stopped": True}

    handle.__globals__["remote"] = remote
    for action, args in (
        ("read", {"extra": True}), ("stop", {"extra": True}),
        ("write", {"target": [True]}), ("write", {"target": [float("nan")]}),
        ("write", {"target": [], "speedFraction": 0.5}),
    ):
        assert not handle(action, args)["ok"]
    assert calls == []
    assert not handle("stop", {})["ok"]
    assert calls == [("stop", {})]
