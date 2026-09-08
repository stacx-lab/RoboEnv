"""The migrated catalog camera adapts RoboEnv transport to the Complex contract."""

import importlib.util
import json
import struct
import zlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "catalog/robots/robocasa-panda-omron/devices/camera_left"


@pytest.fixture
def camera(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CAMERA))
    spec = importlib.util.spec_from_file_location("catalog_camera", CAMERA / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "driver.py")
    module.transport.directory = tmp_path
    profile = yaml.safe_load((CAMERA / "profile.yaml").read_text())
    for stream in ("color", "depth"):
        profile["intrinsics"][stream].update(width=2, height=1)
    profile["access"]["connection"].update(url="http://test-camera:8123", camera="remote-camera")
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "connection.json").write_text(
        json.dumps(
            {
                "endpoint": "http://test-camera:8123",
                "session": "test",
                "generation": 1,
            }
        )
    )

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"\0\xff\0\0\0\xff\0")) + chunk(b"IEND", b"")
    pgm = b"P5\n2 1\n65535\n" + struct.pack(">HH", 1000, 0)
    calls = []

    def exchange(endpoint, method, path="/", body=None, destination=None):
        calls.append((endpoint, method, path))
        if destination:
            destination.write_bytes(pgm if destination.suffix == ".pgm" else png)
            return None
        request = json.loads(body)
        assert request["args"]["device"] == "remote-camera"
        response = {
            key: request[key] for key in ("protocol", "session", "generation", "request_id")
        }
        data = {
            "file": "/server/color.png",
            "depth_file": "/server/depth.pgm",
            "width": 2,
            "height": 1,
            "depth_width": 2, "depth_height": 1,
            "sample_time_s": 1.25, "depth_sample_time_s": 1.25,
        }
        return (json.dumps({**response, "ok": True, "result": {"data": data}}) + "\n").encode()

    monkeypatch.setattr(module.transport, "exchange", exchange)
    return module, calls


def test_catalog_camera_uses_profile_target_and_returns_calibrated_files(camera):
    module, calls = camera
    reading = module.handle("read", {})
    assert reading["ok"], reading
    assert (
        reading["width"],
        reading["height"],
        reading["depth_width"],
        reading["depth_height"],
    ) == (2, 1, 2, 1)
    assert reading["sample_time_s"] == reading["depth_sample_time_s"] == 1.25
    assert reading["received_time_s"] > 0
    assert "depth_scale" not in reading
    assert Path(reading["file"]).is_file()
    assert Path(reading["file"]).parent != module.transport.directory
    assert all(endpoint == "http://test-camera:8123" for endpoint, _, _ in calls)
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["depth_dimensions", "dimensions", "registration"])
def test_catalog_camera_rejects_profile_mismatch(camera, change):
    module, _ = camera
    profile = module.settings()
    if change == "depth_dimensions":
        profile["intrinsics"]["depth"]["width"] = 4
    elif change == "dimensions":
        profile["intrinsics"]["color"]["width"] = 4
    else:
        profile["intrinsics"]["depth"]["registration"] = "native"
    Path(module.__file__).with_name("profile.yaml").write_text(yaml.safe_dump(profile))
    assert module.handle("read", {})["ok"] is False


def test_catalog_camera_rejects_actuator_commands_before_transport(camera):
    module, calls = camera
    for action in ("write", "stop", "ik", "fk"):
        assert module.handle(action, {})["ok"] is False
    assert not module.handle("read", {"extra": True})["ok"]
    assert calls == []


def test_catalog_camera_rejects_mismatched_response(camera, monkeypatch):
    module, _ = camera
    monkeypatch.setattr(module.transport, "exchange", lambda *args, **kwargs: b'{"ok":true}\n')
    reading = module.handle("read", {})
    assert reading["ok"] is False and reading["code"] == "outcome_unknown"


@pytest.mark.parametrize("timestamp", [True, float("nan"), None])
def test_catalog_camera_rejects_invalid_acquisition_time(camera, monkeypatch, timestamp):
    module, _ = camera
    remote = module.remote

    def read(action, args):
        return {**remote(action, args), "sample_time_s": timestamp}

    monkeypatch.setattr(module, "remote", read)
    assert not module.handle("read", {})["ok"]
