"""Portable sensor bundles share transport and never apply calibration twice."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["imu", "force_torque"])
def test_sensor_driver_is_read_only_and_returns_raw_measurements(name, monkeypatch):
    directory = ROOT / "src/roboenv/assets"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("catalog_sensor", directory / "sensor-client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.transport, "settings", lambda: {"type": name})
    calls = []
    raw = (
        {"linear_acceleration": {"x": 0, "y": 0, "z": 9.81}, "orientation": None}
        if name == "imu"
        else {"force": {"x": 1, "y": 2, "z": 3}, "torque": None}
    )

    def remote(action, args):
        calls.append(action)
        return {**raw, "timestamp": 0.04}

    monkeypatch.setattr(module.transport, "remote", remote)
    for action, args in [("write", {}), ("stop", {}), ("read", {"reset": True}), ("read", [])]:
        assert not module.handle(action, args)["ok"]
    assert calls == []
    assert module.handle("read", {}) == {"ok": True, **raw, "timestamp": 0.04}
    assert calls == ["read"]
