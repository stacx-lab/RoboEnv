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
import yaml
from jsonschema import Draft202012Validator
from roboenv.client import Client, RemoteError
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.contract import DEVICE_CONTRACT
from roboenv.devices.export import write_driver
from roboenv.harbor.export import export_task
from roboenv.tasks.model import DEVICE_BACKENDS
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.protocol import Request, Response
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError
from service_fixtures import invoke, legacy_registry

ROOT = Path(__file__).resolve().parents[1]


def test_discover_control_and_observe_through_real_worker(device_service):
    client, _, _ = device_service
    assert client.describe()["device_contract"] == DEVICE_CONTRACT
    devices = client.devices()["devices"]
    assert [d["name"] for d in devices] == ["controller"]
    descriptor = client.device("controller").describe()
    assert descriptor["actions"]["invoke"]["supported"]
    assert not descriptor["actions"]["stop"]["supported"]
    assert descriptor["execution"]["interruptible"] is False
    result = invoke(client)
    assert result["budget"]["used"] == 50
    assert [entry for entry in result["calls"] if entry["op"] == "call"] == [
        {"op": "call", "args": {"name": "push", "kwargs": {"dx": 0.3}}, "ok": True}
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
                "invoke",
                {"tool_name": "push", "kwargs": {"dx": 0.3}, "sleep_s": 1.5},
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
    program.write_text('push(0.3)\nassert get_object_pose(name="puck")[0][0] >= 0.3\n')
    env = {
        **os.environ,
        "ROBOENV_SOCKET": client.socket_path,
        "ROBOENV_RUN_DIR": str(supervisor.config.run_dir),
        "ROBOENV_VERIFIER_OUT": str(tmp_path / "verdict"),
    }
    for arguments in (
        ["devices"],
        ["device", "controller", "invoke", '{"tool_name":"push","kwargs":{"dx":0.3}}'],
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
    assets = importlib.resources.files("roboenv").joinpath("assets")
    verifier = tmp_path / "verify.py"
    verifier.write_text(assets.joinpath("verify.py").read_text())
    (tmp_path / "task.json").write_text(
        json.dumps({"task_id": "device-toy", "seed": 0, "session": "harbor"})
    )
    result = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "verdict/reward.json").read_text()) == {"reward": 1.0}
    with pytest.raises(RemoteError, match="already_submitted"):
        invoke(client)
    client.device("controller").read()


def test_thin_driver_needs_only_stdlib_and_socket(device_service, tmp_path):
    client, _, _ = device_service
    driver = tmp_path / "controller/driver.py"
    write_driver(driver.parent, "other")
    result = subprocess.run(
        [sys.executable, "-S", str(driver), "read", "{}"],
        env={**os.environ, "ROBOENV_SOCKET": client.socket_path},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.parametrize("override", [None, "environment", "connection"])
def test_portable_driver_uses_profile_url_with_explicit_overrides(tmp_path, monkeypatch, override):
    driver = tmp_path / "driver.py"
    write_driver(driver.parent, "other")
    monkeypatch.syspath_prepend(str(driver.parent))
    (tmp_path / "profile.yaml").write_text("access:\n  connection:\n    kind: url\n    url: http://profile:8124\n")
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


def test_all_catalogs_load_without_contacting_simulators():
    from roboenv_runtime.bindings import default_registry

    def forbidden(*args):
        raise AssertionError("Catalog loading must not perform I/O")

    schema = json.loads(
        importlib.resources.files("roboenv")
        .joinpath("specs/device-service.schema.json")
        .read_text()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for workspace in sorted((ROOT / "catalog/robots").rglob("workspace.yaml")):
        from roboenv.devices.bindings.spec import binding_path

        robot = workspace.parent
        raw = yaml.safe_load(binding_path(robot).read_text())
        plan = BindingPlan.load(robot, backend=raw["backend"])
        if raw["backend"] in DEVICE_BACKENDS:
            # Offline definitions validate without a world; native bindings require
            # real model/controller facts, covered by backend conformance tests.
            assert all(spec.definition(robot) for spec in plan.devices)
            continue
        context = BindingContext(raw["backend"], raw["embodiment"], object(), forbidden, {})
        service = default_registry().assemble(plan, robot, context)
        validator.validate(service.inventory())
        for device in service.inventory()["devices"]:
            assert not device["actions"].get("stop", {}).get("supported", False)
            assert "driver" not in device


def test_server_reads_cannot_bypass_submission_or_return_fake_poses(tmp_path, monkeypatch):
    monkeypatch.setattr("roboenv_runtime.bindings.assembly.default_registry", legacy_registry)

    class World(FakeShell):
        name, embodiment = "robodojo", "arx_x5"

        def tools(self):
            return {
                **super().tools(),
                "get_gripper_pose": lambda: [0, 0, 0, 1, 0, 0, 0],
                "get_gripper_state": lambda: 0,
                "get_step_budget": lambda: self.budget().as_dict(),
                "goto_pose": lambda *args: None,
            }

    worker = SimWorker(
        World(),
        task_id="test",
        seed=0,
        run_dir=tmp_path,
        device_root=ROOT / "tests/fixtures/legacy-native-catalogs/robodojo-arx-x5",
    )
    try:
        result = worker.dispatch("devices.call", {"device": "arm", "action": "read", "args": {}})
        assert "flange_pose" not in result["data"]
        assert "native_tool_pose" in result["data"]
        worker.dispatch("submit", {})
        with pytest.raises(WorkerRPCError, match="frozen"):
            worker.dispatch("devices.call", {"device": "arm", "action": "write", "args": {}})
    finally:
        worker.close()


def test_native_export_has_ready_server_and_thin_drivers(tmp_path):
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "fakeshell",
                "image": "sha256:" + "1" * 64,
                "legacy_tree_required": False,
            }
        )
    )
    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"fixture only")
    out = tmp_path / "export"
    export_task(
        ROOT / "tests/fixtures/native-toy/task.yaml",
        ROOT / "tests/fixtures/native-toy/catalog",
        out,
        wheel,
        deployment_path=deployment,
    )
    environment = out / "environment"
    compose = yaml.safe_load((environment / "docker-compose.yaml").read_text())
    assert compose["services"]["main"]["depends_on"]["sim"]["condition"] == "service_healthy"
    assert compose["services"]["sim"]["environment"]["ROBOENV_SERVER_ROBOT"] == "/opt/serve/robot"
    assert compose["services"]["sim"]["environment"]["ROBOENV_SOCKET"] == "/run/roboenv/sim.sock"
    driver = (environment / "definition/robot/devices/controller/driver.py").read_text()
    assert "from roboenv" not in driver and "roboenv_runtime" not in driver
    assert (environment / "sim/robot/devices/controller/driver.py").is_file()


def test_finalizer_classifies_device_mutations():
    from roboenv_runtime.runtime.finalize import _is_physical_mutation

    for action, expected in (("invoke", True), ("read", False), ("write", True)):
        request = Request.from_mapping(
            {
                "protocol": "roboenv/1",
                "session": "test",
                "generation": 0,
                "request_id": "x",
                "deadline_s": 1,
                "op": "devices.call",
                "args": {"device": "arm", "action": action, "args": {}},
            }
        )
        assert _is_physical_mutation(request) is expected


def test_partial_device_motion_is_not_certified(device_service):
    client, supervisor, _ = device_service
    with pytest.raises(RemoteError, match="partial motion"):
        invoke(client, fail_after=True)
    assert client.request("observe")["budget"]["used"] == 50
    verifier = runpy.run_path(str(ROOT / "src/roboenv/assets/verify.py"))
    raw = supervisor.journal.path.read_bytes()
    with pytest.raises(ValueError, match="indeterminate"):
        verifier["validate_journal"](
            raw, session="harbor", submit_id="none", verdict={}, api=client.api()
        )


def test_timeout_fences_request_and_never_claims_emergency_stop(device_service):
    client, supervisor, _ = device_service
    arguments = {"tool_name": "push", "kwargs": {"dx": 0.3}, "sleep_s": 2}
    for timeout in (0.1, 5):
        with pytest.raises(RemoteError) as error:
            client.device("controller").call(
                "invoke", arguments, timeout=timeout, request_id="timeout-motion"
            )
        assert error.value.code == "worker_timeout"
        assert error.value.details["worker_killed"] is True
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
        session="harbor",
        cache_entries=1,
    )
    before = Supervisor(config)
    raw = {
        "protocol": "roboenv/1",
        "session": "harbor",
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


def test_camera_artifact_uses_shared_path_and_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("roboenv_runtime.bindings.assembly.default_registry", legacy_registry)
    import numpy as np

    class CameraWorld(FakeShell):
        name, embodiment = "robodojo", "arx_x5"

        def capture(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    worker = SimWorker(
        CameraWorld(),
        task_id="test",
        seed=0,
        run_dir=tmp_path,
        device_root=ROOT / "tests/fixtures/legacy-native-catalogs/robodojo-arx-x5",
    )
    try:
        result = worker.dispatch(
            "devices.call",
            {"device": "camera", "action": "read", "args": {"file_name": "frame.png"}},
        )
        path = Path(result["data"]["file"])
        assert path == tmp_path / "captures/frame.png"
        assert path.read_bytes().startswith(b"\x89PNG")
        with pytest.raises(WorkerRPCError, match="relative"):
            worker.dispatch(
                "devices.call",
                {"device": "camera", "action": "read", "args": {"file_name": "../../escape.png"}},
            )
    finally:
        worker.close()


def test_call_results_match_published_schema(device_service):
    client, _, _ = device_service
    schema = json.loads(
        importlib.resources.files("roboenv")
        .joinpath("specs/device-service.schema.json")
        .read_text()
    )
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
    (device / "connection.json").write_text(json.dumps({
        "endpoint": "http://unused", "session": "test", "generation": 0,
    }))
    handle = runpy.run_path(str(driver))["handle"]
    requests = []

    def exchange(endpoint, method, body):
        request = json.loads(body)
        requests.append(request)
        if failure == "lost":
            raise ConnectionResetError("response lost after acceptance")
        response = {key: request[key] for key in
                    ("protocol", "session", "generation", "request_id")}
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
