"""Go2's exported adapter uses measured state and preserves environment timing."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "catalog/robots/isaaclab-go2/devices/body"


@pytest.fixture
def legged(monkeypatch):
    library = ROOT.parent / "kinex/python"
    if not (library / "kinex/legged.py").is_file():
        pytest.skip("Requires Kinex's Python legged contract")
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.syspath_prepend(str(BODY))
    spec = importlib.util.spec_from_file_location("catalog_legged", BODY / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = yaml.safe_load((BODY / "profile.yaml").read_text())
    positions = [profile["intrinsics"]["home"][j["name"]] for j in profile["intrinsics"]["joints"]]
    calls = []

    def remote(action, args):
        calls.append((action, args))
        if action == "read":
            return {
                "positions": positions, "velocities": [0] * 12, "efforts": [], "pose": None,
                "body_velocities": [], "faults": [], "sample_time_s": .02,
                "frame_poses": {name: [0, 0, -.3, 0, 0, 0, 1]
                                for name in profile["extrinsics"]["frames"]},
            }
        if action == "stop":
            return {"stopped": True}
        return {"accepted": True}

    monkeypatch.setattr(module, "remote", remote)
    return module, positions, calls


def test_legged_target_order_readback_and_unload_stop(legged):
    module, target, calls = legged
    assert module.handle("write", {"target": target}) == {"ok": True, "accepted": True}
    assert calls == [("write", {"target": target})]
    read = module.handle("read", {})
    assert read["ok"] and read["positions"] == target
    assert read["pose"] is None and read["body_velocities"] == read["efforts"] == []
    assert read["sample_time_s"] == .02
    calls.clear()
    assert module.handle("unload", {})["ok"]
    assert calls == [("stop", {}), ("unload", {})]


def test_legged_rejects_invalid_targets_before_transport(legged):
    module, target, calls = legged
    for value in ([0] * 11, [True] * 12, [float("nan")] * 12, [99] * 12):
        assert not module.handle("write", {"target": value})["ok"]
    assert not module.handle("write", {"target": target, "speedFraction": .1})["ok"]
    assert not module.handle("ik", {})["ok"]
    assert not module.handle("read", {"reset": True})["ok"]
    assert calls == []


def test_legged_uncertain_write_is_not_retried(legged, monkeypatch):
    module, target, calls = legged

    def fail(action, args):
        calls.append(action)
        raise module.WireError("response lost", "outcome_unknown", "request-1")

    monkeypatch.setattr(module, "remote", fail)
    assert module.handle("write", {"target": target}) == {
        "ok": False, "error": "response lost", "code": "outcome_unknown",
        "request_id": "request-1",
    }
    assert calls == ["write"]


def test_legged_unload_does_not_hide_failed_stop(legged, monkeypatch):
    module, _, calls = legged

    def reject(action, args):
        calls.append(action)
        return {"stopped": False}

    monkeypatch.setattr(module, "remote", reject)
    assert not module.handle("unload", {})["ok"]
    assert calls == ["stop"]
