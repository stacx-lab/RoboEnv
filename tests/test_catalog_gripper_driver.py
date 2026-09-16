"""Exported gripper uses typed controls and keeps command and feedback dimensions separate."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GRIPPER = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices/gripper"


@pytest.fixture
def gripper(monkeypatch):
    library = ROOT.parent / "kinex/python"
    if not (library / "kinex/scalar_control.py").is_file():
        pytest.skip("Requires the Kinex Python library used by exported drivers")
    monkeypatch.syspath_prepend(str(library))
    monkeypatch.syspath_prepend(str(GRIPPER))
    spec = importlib.util.spec_from_file_location("catalog_gripper", GRIPPER / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def remote(action, args):
        calls.append((action, args))
        if action == "read":
            return {
                "positions": [0.02, -0.02],
                "velocities": [0, 0],
                "faults": [],
                "sample_time_s": 1.25,
            }
        if action == "stop":
            return {"stopped": True, "behavior": "hold", "affected_devices": ["gripper"]}
        return {"accepted": True}

    monkeypatch.setattr(module, "remote", remote)
    return module, calls


def test_gripper_has_one_signed_command_and_two_measured_channels(gripper):
    module, calls = gripper
    assert module.handle("write", {"target": [0.3]}) == {"ok": True, "accepted": True}
    assert calls == [("write", {"target": [1.0]})]
    result = module.handle("read", {})
    assert result["ok"], result
    assert result["positions"] == [0.02, -0.02]
    assert result["sample_time_s"] == 1.25 and result["aperture_m"] == pytest.approx(0.039)
    assert module.handle("stop", {}) == {
        "ok": True,
        "stopped": True,
        "behavior": "hold",
        "affected_devices": ["gripper"],
    }


def test_bad_targets_and_unsupported_speed_fail_before_transport(gripper):
    module, calls = gripper
    for args in (
        {"target": [2]},
        {"target": [0, 0]},
        {"target": [True]},
        {"target": [float("nan")]},
        {"target": [1], "speedFraction": 0.1},
    ):
        assert module.handle("write", args)["ok"] is False
    assert module.handle("ik", {})["ok"] is False
    assert calls == []


def test_feedback_and_stop_mismatch_are_rejected(gripper, monkeypatch):
    module, _ = gripper
    monkeypatch.setattr(
        module,
        "remote",
        lambda *_: {
            "positions": [0],
            "velocities": [0],
            "faults": [],
            "sample_time_s": 1,
        },
    )
    assert module.handle("read", {})["ok"] is False
    monkeypatch.setattr(module, "remote", lambda *_: {"stopped": True})
    assert module.handle("stop", {})["ok"] is False


def test_catalog_gripper_preserves_tcp_and_omits_ratings():
    profile = yaml.safe_load((GRIPPER / "profile.yaml").read_text())
    assert profile["extrinsics"]["mount"]["translation_m"] == [0, 0, 0.097]
    assert profile["extrinsics"]["mount"]["parent_frame"] == "flange"
    assert "grip_force_n" not in profile["intrinsics"]
    assert "mass_kg" not in profile["intrinsics"]
    assert profile["intrinsics"]["feedback"]["sensors"] == []
    assert profile["intrinsics"]["control"]["increment_per_interval"] == 0.2
