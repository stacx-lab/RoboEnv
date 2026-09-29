"""The exported base adapter validates commands and preserves measured state."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from helpers import kinex_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices/base"


@pytest.fixture
def base(monkeypatch):
    library = kinex_python()
    if not (library / "kinex/velocity.py").is_file():
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
                "velocities": [0.1, 0.2, 0.3],
                "faults": [],
                "sample_time_s": 1.25,
            }
        if action == "stop":
            return {"stopped": True}
        return {"accepted": True}

    monkeypatch.setattr(module, "remote", remote)
    return module, calls


def test_base_keeps_normalized_commands_and_measured_physical_readings(base):
    module, calls = base
    command = {"mode": "drive", "velocity": {"vx": 0.3, "vy": 0, "wz": -0.1},
               "duration_s": 0.5}
    assert module.handle("write", command)["accepted"]
    assert calls == [("write", command)]
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
    for args in (
        {"mode": "unsupported", "velocity": {"vx": 0, "vy": 0, "wz": 0}},
        {"mode": "drive", "velocity": {"vx": 0, "vy": 0}},
        {"mode": "drive", "velocity": {"vx": 0, "vy": 0, "wz": 0}, "posture": {"pitch": True}},
        {
            "mode": "drive",
            "velocity": {"vx": 0, "vy": 0, "wz": 0},
            "posture": {"pitch": float("nan")},
        },
        {"mode": "drive", "velocity": {"vx": 0, "vy": 0, "wz": 0}, "posture": {"typo": 0}},
        {"target": [2, 0, 0]},
        {"target": [0]},
        {"target": [True, 0, 0]},
        {"target": [float("nan"), 0, 0]},
        {"target": [0, 0, 0], "speedFraction": 0.1},
    ):
        assert not module.handle("write", {**args, "duration_s": 0.5})["ok"]
    assert not module.handle("ik", {})["ok"]
    assert not module.handle("stop", {"target": []})["ok"]
    assert calls == []


def test_wheels_reject_unsupported_posture_before_transport(base):
    module, calls = base
    velocity = {"vx": 0.1, "vy": 0, "wz": 0}
    assert not module.handle(
        "write",
        {
            "mode": "drive",
            "velocity": velocity,
            "duration_s": 0.5,
            "posture": {
                "height": 0.7,
                "roll": 0.1,
                "pitch": -0.2,
                "yaw": 0.3,
            },
        },
    )["ok"]
    assert calls == []


def test_stop_and_unload_do_not_require_control_metadata(base, monkeypatch):
    module, calls = base
    monkeypatch.setattr(module, "settings", lambda: {"intrinsics": {"control": "???"}})
    assert module.handle("stop", {})["stopped"]
    assert module.handle("unload", {})["ok"]
    assert calls == [("stop", {}), ("stop", {}), ("unload", {})]


def test_base_rejects_malformed_measurements_and_failed_stop(base, monkeypatch):
    module, _ = base
    monkeypatch.setattr(module, "remote", lambda *_: {"velocities": [0, 0, 0], "faults": []})
    assert not module.handle("read", {})["ok"]
    assert not module.handle("stop", {})["ok"]


def test_catalog_base_is_minimal_and_adapter_matches_export_asset():
    profile = yaml.safe_load((BASE / "profile.yaml").read_text())
    assert profile["type"] == "velocity"
    assert set(profile["intrinsics"]) == {
        "source",
        "method",
        "calibrated_at",
        "control",
        "feedback",
        "stop",
    }
    assert "mount" not in profile["extrinsics"]
    assert profile["access"]["connection"]["device"] == "base"
    asset = ROOT / "src/roboenv/assets/velocity-client.py"
    assert (BASE / "driver.py").read_bytes() == asset.read_bytes()


def test_duration_is_required_and_native_limits_remain_native(base):
    module, calls = base
    args = {"mode": "drive", "velocity": {"vx": 2, "vy": 0, "wz": 0}}
    assert not module.handle("write", args)["ok"]
    for duration in (0, -1, True, float("nan")):
        assert not module.handle("write", {**args, "duration_s": duration})["ok"]
    assert not calls
    assert module.handle("write", {**args, "duration_s": 0.1})["ok"]
    assert calls == [("write", {**args, "duration_s": 0.1})]
