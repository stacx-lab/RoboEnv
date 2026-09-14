"""Go2's exported adapter uses measured state and preserves environment timing."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "catalog/robots/isaaclab/isaaclab-go2/devices/body"


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
    joints = profile["intrinsics"]["kinematics"]["model"]["joints"]
    positions = [profile["intrinsics"]["home"][j["name"]] for j in joints]
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


def test_legged_local_fk_ik_roundtrip_and_no_transport_with_seed(legged):
    module, home, calls = legged
    target = [q + (.04 if i % 3 == 1 else -.08 if i % 3 == 2 else 0)
              for i, q in enumerate(home)]
    fk = module.handle("fk", {"joints": target})
    assert fk["ok"] and len(fk["pose"]) == 4
    pose = [{**entry, "orientation": None} for entry in fk["pose"]]
    solved = module.handle("ik", {"pose": pose, "seed": home})
    assert solved["ok"], solved
    actual = module.handle("fk", {"joints": solved["joints"]})["pose"]
    import numpy as np
    assert np.allclose([p["position"] for p in actual], [p["position"] for p in pose], atol=.0001)
    assert calls == []
    assert module.handle("ik", {"pose": pose})["ok"]
    assert calls == [("read", {})]


def test_legged_ik_rejects_missing_feet_and_impossible_target_without_write(legged):
    module, home, calls = legged
    pose = module.handle("fk", {"joints": home})["pose"]
    assert not module.handle("ik", {"pose": pose[:3], "seed": home})["ok"]
    impossible = [{"position": [9, 9, 9], "orientation": None} for _ in range(4)]
    assert not module.handle("ik", {"pose": impossible, "seed": home})["ok"]
    assert calls == []


def test_legged_dynamic_export_advertises_local_kinematics(tmp_path):
    from roboenv.devices.export import export_devices

    profile = yaml.safe_load((BODY / "profile.yaml").read_text())
    profile["actions"] = {action: {"supported": True}
                          for action in ("read", "write", "stop", "load", "unload", "status")}
    export_devices([profile], tmp_path, {"endpoint": "http://127.0.0.1:8123"})
    exported = yaml.safe_load((tmp_path / "body/profile.yaml").read_text())
    assert {"fk", "ik"} <= set(exported["access"]["driver"]["actions"])
    assert (tmp_path / "body/driver.py").read_bytes() == (BODY / "driver.py").read_bytes()


def test_go2_nonzero_fk_matches_analytic_leg_geometry(legged):
    """Independent Go2 geometry: abduction X, then two pitch Y joints."""
    import math

    import numpy as np

    module, home, _ = legged
    rng = np.random.default_rng(7)
    for _ in range(8):
        joints = (np.asarray(home) + rng.uniform(-.15, .15, 12)).tolist()
        poses = module.handle("fk", {"joints": joints})["pose"]
        for i, pose in enumerate(poses):
            hip, thigh, calf = joints[i*3:i*3+3]
            side = 1 if i % 2 == 0 else -1
            front = 1 if i < 2 else -1
            # Pinned USD offsets and link lengths, in metres.
            x = -.213*(math.sin(thigh) + math.sin(thigh + calf))
            y = side*.0955
            z = -.213*(math.cos(thigh) + math.cos(thigh + calf))
            expected = [front*.1934 + x,
                        side*.0465 + math.cos(hip)*y - math.sin(hip)*z,
                        math.sin(hip)*y + math.cos(hip)*z]
            assert np.allclose(pose["position"], expected, atol=1e-6)
