"""Copied typed drivers keep routing and uncertain outcomes across transport failures."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transport(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portable_transport", ROOT / "src/roboenv/assets/_roboenv.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wire = module.Transport(tmp_path / "renamed-directory/driver.py")
    monkeypatch.setattr(wire, "settings", lambda: {
        "type": "manipulator",
        "access": {"connection": {"url": "http://test:8123", "device": "remote_arm"}},
    })
    for key in ("ROBOENV_ENDPOINT", "ROBOENV_SOCKET", "ROBOENV_SESSION", "ROBOENV_GENERATION"):
        monkeypatch.delenv(key, raising=False)
    return module, wire


@pytest.mark.parametrize("failure", ["disconnect", "request_id", "generation"])
def test_mutating_request_is_never_retried_after_uncertain_response(transport, failure):
    module, wire = transport
    requests = []

    def exchange(endpoint, method, path="/", body=None):
        identity = {"protocol": "roboenv/1", "session": "current", "generation": 7}
        if method == "GET":
            return json.dumps(identity).encode()
        request = json.loads(body)
        requests.append(request)
        assert request["args"]["device"] == "remote_arm"
        assert all(request[key] == value for key, value in identity.items())
        if failure == "disconnect":
            raise ConnectionResetError("response lost")
        response = {**identity, "request_id": request["request_id"], "ok": True,
                    "result": {"data": {"accepted": True}}}
        response[failure] = "different" if failure == "request_id" else 8
        return (json.dumps(response) + "\n").encode()

    wire.exchange = exchange
    with pytest.raises(module.WireError) as error:
        wire.remote("write", {"target": [0.1]})
    assert error.value.code == "outcome_unknown"
    assert len(requests) == 1
    assert error.value.request_id == requests[0]["request_id"]
