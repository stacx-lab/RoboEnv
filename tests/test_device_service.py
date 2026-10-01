"""Device-contract conformance using real sockets and fresh CPU worker processes."""

import importlib.resources
import json
import os
import runpy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from roboenv.client import Client, RemoteError
from roboenv.devices.contract import DEVICE_CONTRACT
from roboenv.devices.export import write_driver
from roboenv_runtime.runtime.protocol import Request, Response
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig
from service_fixtures import invoke

ROOT = Path(__file__).resolve().parents[1]


def test_discover_control_and_observe_through_real_worker(device_service):
    client, _, _ = device_service
    assert client.describe()["device_contract"] == DEVICE_CONTRACT
    devices = client.devices()["devices"]
    assert [d["name"] for d in devices] == ["controller"]
    descriptor = client.device("controller").describe()
    assert descriptor["actions"]["write"]["supported"]
    assert not descriptor["actions"]["stop"]["supported"]
    assert descriptor["execution"]["interruptible"] is False
    result = invoke(client)
    assert result["budget"]["used"] == 50
    assert [entry for entry in result["calls"] if entry["op"] == "binding.write"] == [
        {"op": "binding.write", "args": {"target": [0.3]}, "ok": True}
    ]
    assert client.request("score.peek")["completed"]
    assert client.device("controller").read()["read_only"]


def test_durable_dedup_survives_cache_eviction_and_conflict(device_service):
    client, _, _ = device_service
    first = invoke(client, request_id="motion-1")
    for _ in range(4):
        client.request("health")
    assert invoke(client, request_id="motion-1") == first
    assert client.request("observe")["budget"]["used"] == 50
    with pytest.raises(RemoteError) as error:
        invoke(client, request_id="motion-1", sleep_s=0)
    assert error.value.code == "request_id_conflict"
    assert client.operation("motion-1")["state"] == "succeeded"
    assert client.operation("not-sent")["state"] == "unknown"


def test_status_and_duplicate_timeout_do_not_wait_for_or_poison_motion(device_service):
    client, _, _ = device_service
    with ThreadPoolExecutor() as pool:
        future = pool.submit(invoke, client, request_id="slow", sleep_s=1.5)
        deadline = time.monotonic() + 5
        while client.operation("slow")["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        started = time.monotonic()
        assert client.operation("slow")["state"] == "running"
        assert time.monotonic() - started < 0.75
        health = subprocess.run(
            [sys.executable, str(ROOT / "src/roboenv/assets/healthcheck.py")],
            env={**os.environ, "ROBOENV_SOCKET": client.socket_path},
            capture_output=True,
            text=True,
            timeout=1,
        )
        assert health.returncode == 0, health.stderr
        with pytest.raises(RemoteError) as error:
            client.device("controller").call(
                "write",
                {"target": [0.3], "sleep_s": 1.5},
                request_id="slow",
                timeout=0.05,
            )
        assert error.value.code == "request_in_progress"
        future.result(timeout=5)
    assert client.operation("slow")["state"] == "succeeded"
    assert client.request("observe")["budget"]["used"] == 50


def test_unsupported_capabilities_never_succeed(device_service):
    client, _, _ = device_service
    for action in ("stop", "ik", "fk", "unload", "policy_run", "reset"):
        with pytest.raises(RemoteError) as error:
            client.device("controller").call(action)
        assert error.value.code == "unsupported_action"
        assert error.value.details["effects"] == "none"
    assert client.request("observe")["budget"]["used"] == 0


def test_read_cannot_dispatch_motion_and_identity_is_checked(device_service):
    client, _, _ = device_service
    with pytest.raises(RemoteError) as error:
        client.device("controller").read(bad_read=True)
    assert error.value.code == "permission_denied"
    for name in ("../controller", "/etc/passwd", "unknown"):
        with pytest.raises(RemoteError, match="unknown_device"):
            client.device(name).read()
    with pytest.raises(RemoteError, match="generation_mismatch"):
        Client(client.socket_path, generation=1).devices()
    assert client.request("observe")["budget"]["used"] == 0


def test_no_local_agent_dependency_cli_and_independent_verifier(device_service, tmp_path):
    client, supervisor, _ = device_service
    program = tmp_path / "solution.py"
    program.write_text(
        'env.device("controller").write(target=[0.3])\n'
        'assert env.device("controller").read()["objects"]["puck"]["position"][0] >= 0.3\n'
    )
    env = {
        **os.environ,
        "ROBOENV_SOCKET": client.socket_path,
        "ROBOENV_RUN_DIR": str(supervisor.config.run_dir),
    }
    for arguments in (
        ["devices"],
        ["device", "controller", "write", '{"target":[0.3]}'],
        ["replay", str(program)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "roboenv.cli.main", *arguments],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)
    from roboenv.verification import attach_budget, validate_journal, validate_score

    client.allowed_ops = client.allowed_ops | {"submit"}
    verdict = client.request("submit", request_id="owner-submit")
    observation = client.request("observe")
    raw = supervisor.journal.path.read_bytes()
    assert validate_score(attach_budget(verdict, observation))
    assert validate_journal(
        raw, session=client.session, submit_id="owner-submit", verdict=verdict
    )["requests"] > 0
    with pytest.raises(ValueError, match="identity changed"):
        validate_journal(raw, session="wrong-session", submit_id="owner-submit", verdict=verdict)
    with pytest.raises(RemoteError, match="already_submitted"):
        invoke(client)
    client.device("controller").read()


def test_thin_driver_needs_only_stdlib_and_socket(device_service, tmp_path):
    client, _, _ = device_service
    driver = tmp_path / "controller/driver.py"
    write_driver(driver.parent, "other")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(driver), "read", "{}"],
        env={**os.environ, "ROBOENV_SOCKET": client.socket_path},
        cwd=tmp_path.parent,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
    assert (tmp_path / "_roboenv.py").is_file()
    assert not (driver.parent / "_roboenv.py").exists()
    assert not list(tmp_path.rglob("__pycache__"))
    (tmp_path / "_roboenv.py").unlink()
    missing = subprocess.run(
        [sys.executable, "-I", "-S", str(driver), "read", "{}"],
        cwd=tmp_path.parent,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert missing.returncode != 0 and "ModuleNotFoundError" in missing.stderr


@pytest.mark.parametrize("override", [None, "environment", "connection"])
def test_portable_driver_uses_profile_url_with_explicit_overrides(tmp_path, monkeypatch, override):
    driver = tmp_path / "driver.py"
    write_driver(driver.parent, "other")
    monkeypatch.syspath_prepend(str(driver.parent))
    (tmp_path / "profile.yaml").write_text(
        "access:\n  connection:\n    kind: url\n    url: http://profile:8124\n"
    )
    for key in ("ROBOENV_ENDPOINT", "ROBOENV_SOCKET", "ROBOENV_SESSION", "ROBOENV_GENERATION"):
        monkeypatch.delenv(key, raising=False)
    expected = "http://profile:8124"
    if override:
        monkeypatch.setenv("ROBOENV_ENDPOINT", "http://environment:8125")
        expected = "http://environment:8125"
    if override == "connection":
        expected = "http://connection:8126"
        (tmp_path / "connection.json").write_text(json.dumps({"endpoint": expected}))
    remote = runpy.run_path(str(driver))["remote"]

    def exchange(endpoint, method, path="/", body=None):
        assert endpoint == expected
        if method == "GET":
            return json.dumps({"protocol": "roboenv/1", "session": "test", "generation": 0})
        request = json.loads(body)
        response = {
            key: request[key] for key in ("protocol", "session", "generation", "request_id")
        }
        response.update(ok=True, result={"data": {"loaded": True}})
        return (json.dumps(response) + "\n").encode()

    remote.__self__.exchange = exchange
    assert remote("status", {}) == {"loaded": True}


def test_partial_device_motion_is_not_certified(device_service):
    client, supervisor, _ = device_service
    with pytest.raises(RemoteError, match="partial motion"):
        invoke(client, fail_after=True)
    assert client.request("observe")["budget"]["used"] == 50
    verifier = runpy.run_path(str(ROOT / "src/roboenv/verification.py"))
    raw = supervisor.journal.path.read_bytes()
    with pytest.raises(ValueError, match="indeterminate"):
        verifier["validate_journal"](raw, session="roboenv", submit_id="none", verdict={})


@pytest.mark.parametrize("journal_delay_s", [0, 0.2])
def test_timeout_fences_request_and_never_claims_emergency_stop(
    device_service, monkeypatch, journal_delay_s
):
    client, supervisor, _ = device_service
    intent = supervisor.journal.intent
    proxy = supervisor._proxy
    dispatched = []

    def delayed_intent(*args, **kwargs):
        intent(*args, **kwargs)
        time.sleep(journal_delay_s)

    def short_worker_deadline(request, deadline):
        dispatched.append(request.request_id)
        return proxy(request, min(deadline, time.monotonic() + 0.1))

    monkeypatch.setattr(supervisor.journal, "intent", delayed_intent)
    # Allow durable admission to finish even on slow CI disks; exercise the real
    # socket timeout at the worker proxy, after the intent has been recorded.
    monkeypatch.setattr(supervisor, "_proxy", short_worker_deadline)
    arguments = {"target": [0.3], "sleep_s": 2}
    for timeout in (5, 10):
        with pytest.raises(RemoteError) as error:
            client.device("controller").call(
                "write", arguments, timeout=timeout, request_id="timeout-motion"
            )
        assert error.value.code == "worker_timeout"
        assert error.value.details["worker_killed"] is True
    assert dispatched == ["timeout-motion"]
    assert client.operation("timeout-motion")["state"] == "indeterminate"
    assert client.request("health")["worker"]["alive"] is False
    events = supervisor.journal.events()
    assert (
        sum(
            e["phase"] == "intent" and e["request"]["request_id"] == "timeout-motion"
            for e in events
        )
        == 1
    )


def test_restart_retrieves_terminal_results_and_fences_unfinished_intents(tmp_path):
    config = SupervisorConfig(
        run_dir=tmp_path,
        task="test",
        seed=0,
        budget_mode="official",
        session="roboenv",
        cache_entries=1,
    )
    before = Supervisor(config)
    raw = {
        "protocol": "roboenv/1",
        "session": "roboenv",
        "generation": 0,
        "request_id": "done",
        "deadline_s": 1,
        "op": "devices.call",
        "args": {"device": "arm", "action": "write", "args": {"target": [0, 0, 0]}},
    }
    done = Request.from_mapping(raw)
    response = Response.success(done, {"recorded": True}).as_dict()
    before.journal.intent(raw)
    before.journal.commit(raw, response)
    pending = Request.from_mapping({**raw, "request_id": "unfinished"})
    before.journal.intent(pending.as_dict())
    restored = Supervisor(config)
    assert restored.handle(done) == response
    assert restored.handle(pending)["error"]["code"] == "indeterminate_previous_attempt"
    assert restored.operation({"request_id": "unfinished"})["state"] == "indeterminate"
    conflict = Request.from_mapping({**raw, "args": {**raw["args"], "action": "read"}})
    assert restored.handle(conflict)["error"]["code"] == "request_id_conflict"


def test_call_results_match_published_schema(device_service):
    client, _, _ = device_service
    schema = json.loads(
        importlib.resources.files("roboenv")
        .joinpath("specs/device-service.schema.json")
        .read_text()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(client.device("controller").describe())
    validator.validate(client.device("controller").read())
    validator.validate(invoke(client))


@pytest.mark.parametrize(
    "args",
    [
        [],
        "invalid",
        {"device": "controller", "action": "read"},
        {"device": "controller", "action": "read", "args": []},
    ],
)
def test_malformed_device_requests_cannot_move_world(device_service, args):
    client, _, _ = device_service
    with pytest.raises((ValueError, RemoteError)):
        client.request("devices.call", args)
    assert client.request("observe")["budget"]["used"] == 0


@pytest.mark.parametrize("failure", ["lost", "mismatched", "missing_data"])
def test_portable_driver_never_retries_an_uncertain_write(tmp_path, failure, monkeypatch):
    device = tmp_path / "arm"
    device.mkdir()
    driver = device / "driver.py"
    write_driver(driver.parent, "other")
    monkeypatch.syspath_prepend(str(driver.parent))
    (device / "connection.json").write_text(
        json.dumps(
            {
                "endpoint": "http://unused",
                "session": "test",
                "generation": 0,
            }
        )
    )
    handle = runpy.run_path(str(driver))["handle"]
    requests = []

    def exchange(endpoint, method, body):
        request = json.loads(body)
        requests.append(request)
        if failure == "lost":
            raise ConnectionResetError("response lost after acceptance")
        response = {
            key: request[key] for key in ("protocol", "session", "generation", "request_id")
        }
        response.update(ok=True, result={"data": {"accepted": True}})
        if failure == "mismatched":
            response["request_id"] = "another-request"
        if failure == "missing_data":
            response["result"] = {}
        return json.dumps(response).encode() + b"\n"

    handle.__globals__["transport"].exchange = exchange
    result = handle("write", {"joints": [0.1]})
    assert len(requests) == 1
    assert result["ok"] is False
    assert result["code"] == "outcome_unknown"
    assert result["request_id"] == requests[0]["request_id"]
