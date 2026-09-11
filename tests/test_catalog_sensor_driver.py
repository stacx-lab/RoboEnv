"""Portable sensor bundles share transport and never apply calibration twice."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["imu", "contact_fl_foot", "contact_fr_foot",
                                  "contact_rl_foot", "contact_rr_foot"])
def test_sensor_driver_is_read_only_and_returns_raw_measurements(name, monkeypatch):
    directory = ROOT / "catalog/robots/isaaclab/isaaclab-go2/devices" / name
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("catalog_sensor", directory / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    raw = ({"linear_acceleration": {"x": 0, "y": 0, "z": 9.81}, "orientation": None}
           if name == "imu" else {"force": {"x": 1, "y": 2, "z": 3}, "torque": None})

    def remote(action, args):
        calls.append(action)
        return {**raw, "sample_time_s": .04}

    monkeypatch.setattr(module.transport, "remote", remote)
    for action, args in [("write", {}), ("stop", {}), ("read", {"reset": True}), ("read", [])]:
        assert not module.handle(action, args)["ok"]
    assert calls == []
    assert module.handle("read", {}) == {"ok": True, **raw, "sample_time_s": .04}
    assert calls == ["read"]
    assert (directory / "_roboenv.py").read_bytes() == (
        ROOT / "src/roboenv/assets/_roboenv.py"
    ).read_bytes()
