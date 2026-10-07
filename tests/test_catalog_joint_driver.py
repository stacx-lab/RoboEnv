"""Exported joint IK retains explicit/default seeds and rejects removed controls."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from helpers import kinex_python

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "catalog/robots/libero/libero-panda-kitchen/devices/arm"


@pytest.fixture
def arm(monkeypatch):
    library = kinex_python()
    if not (library / "kinex/joints.py").is_file():
        pytest.skip("Requires the Kinex Python library used by exported drivers")
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.syspath_prepend(str(ARM))
    spec = importlib.util.spec_from_file_location("catalog_joint", ARM / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    joint = module.model()
    measured = [(j["limits"]["min"] + j["limits"]["max"]) / 2 for j in joint.joints]
    remote = Mock(return_value={"positions": measured})
    monkeypatch.setattr(module, "remote", remote)
    return module, joint, measured, remote


def test_explicit_and_default_seed_reach_local_solver(arm, monkeypatch):
    module, joint, measured, remote = arm
    target = joint.fk(measured)["pose"]
    solve = Mock(wraps=joint.solver.ik)
    monkeypatch.setattr(joint.solver, "ik", solve)
    result = module.handle("ik", {"pose": target, "seed": measured})
    assert result["ok"], result
    assert result["joints"] == pytest.approx(measured)
    assert solve.call_args.args[1] == measured
    assert solve.call_args.kwargs == {}
    remote.assert_not_called()
    assert "budget" not in joint.options
    assert module.handle("ik", {"pose": target})["ok"]
    remote.assert_called_once_with("read", {})
    assert solve.call_args.args[1] == measured
    assert solve.call_args.kwargs == {}


def test_invalid_ik_controls_fail_without_transport(arm):
    module, joint, measured, remote = arm
    target = joint.fk(measured)["pose"]
    for extra in ({"budget": None}, {"budget": {"max_attempts": True}},
                  {"budget": {"attempt_time_s": 0}}, {"search": {}},
                  {"reference_joints": []}, {"seed": [0]},
                  {"joint_bounds": []}):
        result = module.handle("ik", {"pose": target, "seed": measured, **extra})
        assert result["ok"] is False, result
    remote.assert_not_called()
