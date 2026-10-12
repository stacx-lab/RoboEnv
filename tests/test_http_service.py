"""HTTP and Unix requests share identity, budgets, deduplication and uncertainty."""

import http.client
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest
import roboenv
from roboenv.client import Client, RemoteError, UncertainResult


@pytest.fixture
def http_client(device_service):
    unix, supervisor, _ = device_service
    deadline = time.monotonic() + 5
    while supervisor.http_endpoint is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return Client(supervisor.http_endpoint), unix, supervisor


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_identity_discovery_is_read_only_and_checks_endpoint(http_client):
    client, unix, _ = http_client
    url = urlsplit(client.endpoint)
    for path, headers, status in (
        ("/", {}, 200),
        ("/wrong", {}, 404),
        ("/", {"Origin": "https://untrusted.example"}, 403),
        ("/", {"Host": "untrusted.example"}, 403),
    ):
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
        try:
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            assert response.status == status
            if status == 200:
                identity = json.loads(response.read())
                assert identity["session"] == unix.session
                assert identity["generation"] == unix.generation
        finally:
            conn.close()
    assert unix.request("observe")["budget"]["used"] == 0


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_standalone_cli_and_sdk_receive_native_goal_without_benchmark(http_client, tmp_path):
    client, unix, _ = http_client
    result = subprocess.run(
        [sys.executable, "-m", "roboenv.cli.main", "describe", "--endpoint", client.endpoint],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    task = json.loads(result.stdout)["task"]
    assert task["id"] == "fakeshell/device-toy"
    # The fixture catalog says only "Push the puck."; native supplies the target.
    assert task["instruction"] == task["native_instruction"] == "Push the puck to x >= 0.3."
    assert task["context"] == {}
    with roboenv.connect(client.endpoint) as env:
        assert env.task.id == task["id"]
        assert env.task.instruction == task["instruction"]
        assert env.refresh_task().instruction == task["instruction"]
        assert unix.request("observe")["budget"]["used"] == 0
        env.device("controller").write(target=[0.3])
    assert unix.request("score.peek")["completed"]


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_transport_switch_does_not_repeat_a_motion(http_client):
    http, unix, _ = http_client
    args = {"target": [0.3]}
    first = http.device("controller").call("write", args, request_id="both-transports")
    assert unix.device("controller").call("write", args, request_id="both-transports") == first
    assert unix.request("observe")["budget"]["used"] == 50
    with pytest.raises(RemoteError, match="unsupported_op"):
        http.request("observe")
    with pytest.raises(RemoteError, match="request_id_conflict"):
        unix.device("controller").call(
            "write", {"target": [0.1]}, request_id="both-transports"
        )


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_http_status_remains_available_during_motion(http_client):
    http, _, _ = http_client
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            http.device("controller").call,
            "write",
            {"target": [0.3], "sleep_s": 0.8},
            request_id="slow-http",
        )
        deadline = time.monotonic() + 5
        while http.operation("slow-http")["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert http.request("health", timeout=0.2)["worker"]["alive"]
        future.result(timeout=5)


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_http_rejects_wrong_path_and_owner_operations(http_client):
    http, unix, _ = http_client
    with pytest.raises(ValueError):
        Client(http.endpoint + "/wrong-run").devices()
    owner = Client(http.endpoint)
    owner.allowed_ops = Client.allowed_ops | {"shutdown"}
    with pytest.raises(RemoteError):
        owner.request("shutdown")
    assert unix.request("health")["worker"]["alive"]
    assert unix.request("observe")["budget"]["used"] == 0


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_public_metadata_never_reuses_privileged_owner_results(http_client):
    http, owner, supervisor = http_client
    owner.request("observe", request_id="private-state")
    owner.request("operations.get", {"request_id": "private-state"}, request_id="owner-lookup")
    with pytest.raises(RemoteError, match="unsupported_op"):
        http.request("operations.get", {"request_id": "private-state"}, request_id="owner-lookup")
    assert http.operation("owner-lookup")["state"] == "unknown"
    private_health = owner.request("health", request_id="shared-health")
    assert "seed" in private_health
    public_health = http.request("health", request_id="shared-health")
    assert "seed" not in public_health
    assert set(public_health["worker"]) == {"state", "alive"}
    assert supervisor.config.run_dir.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_http_rejects_bad_length_and_browser_requests(http_client):
    client, unix, _ = http_client
    url = urlsplit(client.endpoint)
    for headers, body, status in (
        ({"Content-Length": "999999999"}, b"", 413),
        ({"Origin": "https://untrusted.example", "Content-Type": "application/json"}, b"{}", 403),
        ({"Content-Type": "application/json"}, b"{broken", 200),
    ):
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
        try:
            conn.request("POST", url.path, body=body, headers=headers)
            response = conn.getresponse()
            assert response.status == status
            if status == 200:
                assert json.loads(response.read())["ok"] is False
        finally:
            conn.close()
    assert unix.request("observe")["budget"]["used"] == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:0",
        "http://" + "user:secret@localhost:8000",
        "http://localhost:8000/wrong/id",
        "http://localhost:8000?x=1",
    ],
)
def test_endpoint_validation_precedes_network_access(endpoint):
    with pytest.raises(ValueError):
        Client(endpoint)


def test_lost_http_reply_preserves_request_id_without_resending(monkeypatch):
    attempts = []

    class LostReply:
        def __init__(self, *_args, **_kwargs):
            pass

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            attempts.append("sent")

        def getresponse(self):
            raise http.client.RemoteDisconnected("reply lost after send")

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", LostReply)
    with pytest.raises(UncertainResult) as failure:
        Client("http://127.0.0.1:8123", session="test").device("arm").write(
            actions=[[0.3]], duration_s=0.1, request_id="uncertain-motion"
        )
    assert attempts == ["sent"]
    assert failure.value.request_id == "uncertain-motion"
