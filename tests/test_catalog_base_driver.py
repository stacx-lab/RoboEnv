"""The exported base adapter validates commands and preserves measured state."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "catalog/robots/robocasa-panda-omron/devices/base"


@pytest.fixture
def base(monkeypatch):
    library = ROOT.parent / "kinex/python"
    if not (library / "kinex/base.py").is_file():
        pytest.skip("Requires Kinex's Python library for exported drivers")
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.syspath_prepend(str(BASE))
    spec = importlib.util.spec_from_file_location("catalog_base", BASE / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def remote(action, args):
        calls.append((action, args))
        if action == "read":
            return {
                "pose": {"translation_m": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1]},
                "velocities": [0.1, 0.2, 0.3], "faults": [], "sample_time_s": 1.25,
            }
        if action == "stop":
            return {"stopped": True}
        return {"accepted": True}

    monkeypatch.setattr(module, "remote", remote)
    return module, calls


def test_base_keeps_normalized_commands_and_measured_physical_readings(base):
    module, calls = base
    assert module.handle("write", {"target": [0.3, 0, -0.1]})["accepted"]
    assert calls == [("write", {"target": [0.3, 0, -0.1]})]
    measured = module.handle("read", {})
    assert measured["ok"] and measured["velocities"] == [0.1, 0.2, 0.3]
    assert measured["pose"]["translation_m"] == [1, 2, 3]
    assert measured["sample_time_s"] == 1.25
    assert module.handle("stop", {}) == {"ok": True, "stopped": True}
    calls.clear()
    assert module.handle("unload", {})["ok"]
    assert calls == [("stop", {}), ("unload", {})]


def test_base_rejects_bad_commands_before_transport(base):
    module, calls = base
    for args in ({"target": [2, 0, 0]}, {"target": [0]}, {"target": [True, 0, 0]},
                 {"target": [float("nan"), 0, 0]},
                 {"target": [0, 0, 0], "speedFraction": 0.1}):
        assert not module.handle("write", args)["ok"]
    assert not module.handle("ik", {})["ok"]
    assert not module.handle("stop", {"target": []})["ok"]
    assert calls == []


def test_base_rejects_malformed_measurements_and_failed_stop(base, monkeypatch):
    module, _ = base
    monkeypatch.setattr(module, "remote", lambda *_: {"velocities": [0, 0, 0], "faults": []})
    assert not module.handle("read", {})["ok"]
    assert not module.handle("stop", {})["ok"]


def test_catalog_base_is_minimal_and_adapter_matches_export_asset():
    profile = yaml.safe_load((BASE / "profile.yaml").read_text())
    assert profile["type"] == "base"
    assert set(profile["intrinsics"]) == {
        "source", "method", "calibrated_at", "mobility", "control", "feedback",
    }
    assert "mount" not in profile["extrinsics"]
    assert profile["access"]["connection"]["device"] == "base"
    asset = ROOT / "src/roboenv/assets/base-client.py"
    assert (BASE / "driver.py").read_bytes() == asset.read_bytes()
