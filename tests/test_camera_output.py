"""Client output modes reuse server artifacts without changing wire read arguments."""

import base64
import importlib.util
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from roboenv.assets._camera import arrays, camera_read
from roboenv.assets._roboenv import Transport
from roboenv.devices.client import Environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def camera_server(tmp_path):
    color = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.array([[0, 1], [256, 65535]], dtype=np.uint16)
    stream = io.BytesIO()
    Image.fromarray(color).save(stream, format="PNG")
    images = {
        "rgb.png": stream.getvalue(),
        "depth.pgm": b"P5\n2 2\n65535\n" + depth.astype(">u2").tobytes(),
    }
    data = {"file": "/server/rgb.png", "depth_file": "/server/depth.pgm", "timestamp": 2.5}
    profile = {
        "name": "camera",
        "type": "rgbd",
        "actions": {"read": {"supported": True}},
        "intrinsics": {
            "color": {"width": 2, "height": 2},
            "depth": {
                "width": 2,
                "height": 2,
                "invalid_value": 0,
                "registration": "color",
                "quantity": "z",
                "scale_m_per_unit": 0.001,
                "valid_range_m": {"min": 0.001, "max": 65.535},
            },
        },
    }
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send(images[Path(self.path).name])

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            assert request["args"] == {"device": "camera", "action": "read", "args": {}}
            self.send(
                (
                    json.dumps(
                        {
                            **{
                                key: request[key]
                                for key in ("protocol", "session", "generation", "request_id")
                            },
                            "ok": True,
                            "result": {"data": data},
                        }
                    )
                    + "\n"
                ).encode()
            )

        def send(self, body):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    wire = Transport(tmp_path / "driver.py", device="camera")
    wire.config = {"endpoint": endpoint, "session": "test", "generation": 0}
    wire.profile = profile
    try:
        yield SimpleNamespace(
            wire=wire,
            images=images,
            data=data,
            profile=profile,
            color=color,
            depth=depth,
            calls=calls,
            endpoint=endpoint,
        )
    finally:
        wire.close_all()
        server.shutdown()
        thread.join()
        server.server_close()


def test_driver_converts_received_images_and_keeps_server_read_unchanged(
    camera_server, monkeypatch
):
    fixture = camera_server
    monkeypatch.syspath_prepend(str(ROOT / "src/roboenv/assets"))
    spec = importlib.util.spec_from_file_location(
        "camera_client", ROOT / "src/roboenv/assets/camera-client.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.remote = fixture.wire.remote
    driver.settings = lambda: fixture.profile
    import roboenv.assets._roboenv as transport_module

    original_mkdtemp = transport_module.tempfile.mkdtemp
    monkeypatch.setattr(
        transport_module.tempfile,
        "mkdtemp",
        lambda **_: pytest.fail("content read created a local image directory"),
    )
    for mode in ("base64", "array"):
        result = driver.handle("read", {"output": mode})
        assert result.pop("ok"), result
        assert set(result) == {"color", "depth", "timestamp"}
        assert base64.b64decode(result["color"]) == fixture.images["rgb.png"]
        assert base64.b64decode(result["depth"]) == fixture.images["depth.pgm"]
        decoded = arrays(result, fixture.profile)
        np.testing.assert_array_equal(decoded["color"], fixture.color)
        np.testing.assert_array_equal(decoded["depth"], fixture.depth)
        assert decoded["depth"].dtype == np.uint16 and decoded["color"].dtype == np.uint8
        assert decoded["color"].flags.owndata and decoded["depth"].flags.writeable
    count = len(fixture.calls)
    assert not driver.handle("read", {"output": "invalid"})["ok"]
    assert len(fixture.calls) == count
    monkeypatch.setattr(transport_module.tempfile, "mkdtemp", original_mkdtemp)
    files = driver.handle("read", {})
    assert files["ok"] and files["timestamp"] == 2.5
    assert Path(files["file"]).read_bytes() == fixture.images["rgb.png"]
    for key in ("file", "depth_file"):
        Path(files[key]).unlink()
    Path(files["file"]).parent.rmdir()


def test_sdk_array_base64_and_default_file_outputs(camera_server):
    fixture = camera_server
    requests = []

    def call(action, args):
        requests.append((action, args))
        return {"data": fixture.data}

    transport = SimpleNamespace(
        endpoint=fixture.endpoint,
        session="test",
        generation=0,
        describe=lambda: {"task": {"id": "test", "instruction": "test"}},
        devices=lambda: {"devices": [fixture.profile]},
        device=lambda _: SimpleNamespace(call=call),
    )
    with Environment(transport) as env:
        camera = env.device("camera")
        sample = camera.read(output="array")
        assert set(sample) == {"color", "depth", "timestamp"}
        np.testing.assert_array_equal(sample["color"], fixture.color)
        np.testing.assert_array_equal(sample["depth"], fixture.depth)
        sample["color"].fill(0)
        assert fixture.color.any()
        encoded = camera.read(output="base64")
        assert base64.b64decode(encoded["color"]) == fixture.images["rgb.png"]
        assert env._artifacts is None
        files = camera.read()
        assert Path(files["file"]).read_bytes() == fixture.images["rgb.png"]
        assert all(args == {} for _, args in requests)
        assert "output" not in fixture.data
    assert not Path(files["file"]).exists()


def test_local_content_preserves_server_files(camera_server, tmp_path):
    fixture = camera_server
    color = tmp_path / "color.png"
    color.write_bytes(fixture.images["rgb.png"])
    env = Environment(
        SimpleNamespace(
            endpoint=None,
            describe=lambda: {"task": {"id": "test", "instruction": "test"}},
            devices=lambda: {"devices": []},
        )
    )
    assert env._download(str(color), content=True) == fixture.images["rgb.png"]
    env.close()
    assert color.is_file()


def test_malformed_content_and_profile_mismatches_fail(camera_server):
    fixture = camera_server
    data = {
        "timestamp": 1,
        "color": base64.b64encode(fixture.images["rgb.png"]).decode(),
        "depth": base64.b64encode(fixture.images["depth.pgm"]).decode(),
    }
    assert camera_read(data, fixture.profile) == data
    for patch in (
        {"color": "!"},
        {"depth": base64.b64encode(b"P5\n2 2\n65535\n").decode()},
        {"timestamp": float("nan")},
    ):
        with pytest.raises(ValueError):
            camera_read({**data, **patch}, fixture.profile)
    wrong = {
        **fixture.profile,
        "intrinsics": {**fixture.profile["intrinsics"], "color": {"width": 3, "height": 2}},
    }
    with pytest.raises(ValueError, match="resolution"):
        camera_read(data, wrong)


def test_oversized_remote_content_fails_without_local_artifacts(camera_server, monkeypatch):
    fixture = camera_server
    import roboenv.assets._roboenv as transport_module

    monkeypatch.setattr(transport_module, "MAX_BYTES", 1024)
    fixture.images["rgb.png"] = b"x" * 1025
    with pytest.raises(ValueError, match="output='file'"):
        fixture.wire.remote("read", {}, image_output="base64")
