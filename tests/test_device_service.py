"""Device-contract conformance using real sockets and fresh CPU worker processes."""

import importlib.resources
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from roboenv.bindings.registry import BindingContext
from roboenv.bindings.spec import BindingPlan
from roboenv.client import Client, RemoteError
from roboenv.contract import DEVICE_CONTRACT
from roboenv.harbor import export_task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))

from roboenv_runtime.adapters.fakeshell import FakeShell  # noqa: E402
from roboenv_runtime.runtime.protocol import Request, Response, socket_address  # noqa: E402
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig  # noqa: E402
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError  # noqa: E402


@pytest.fixture
def device_service(tmp_path, monkeypatch, request):
    robot = tmp_path / "robot"
    shutil.copytree(ROOT / "tests/fixtures/native-toy/catalog/robots/toy-puck", robot)
    # This fixture deliberately tests the legacy macro binding's error paths.
    if getattr(request, "param", None) != "native":
        (robot / "bindings.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "roboenv/bindings/1",
                    "backend": "fakeshell",
                    "devices": {"controller": {"binding": "catalog/1", "entity": "controller"}},
                }
            )
        )
    driver = robot / "devices/controller/driver.py"
    driver.write_text(
        "import time\nfrom roboenv.client import call, rpc\n"
        "def handle(action, args):\n"
        "    if args.get('bad_read'): return call('push', dx=0.1)\n"
        "    if action == 'invoke':\n"
        "        result = call(args['tool_name'], **args.get('kwargs', {}))\n"
        "        time.sleep(args.get('sleep_s', 0))\n"
        "        if args.get('fail_after'): raise RuntimeError('partial motion')\n"
        "        return {'result': result}\n"
        "    return rpc('observe')\n"
    )
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "device-toy.yaml").write_text(
        "id: device-toy\nadapter: fakeshell\ntext: Push the puck.\n"
    )
    monkeypatch.setenv("ROBOENV_SERVER_ROBOT", str(robot))
    monkeypatch.delenv("ROBOENV_SERVER_MANIFEST", raising=False)
    monkeypatch.delenv("ROBOENV_SOCKET_ROOT", raising=False)
    monkeypatch.setenv("ROBOENV_TASKS", str(tasks))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(ROOT / "runtime/src"), str(ROOT / "src")]),
    )
    config = SupervisorConfig(
        run_dir=tmp_path / "run",
        task="device-toy",
        seed=0,
        budget_mode="official",
        session="harbor",
        cache_entries=2,
        terminate_grace_s=0.2,
    )
    supervisor = Supervisor(config)
    supervisor.start_worker()
    supervisor.open_server()
    thread = threading.Thread(target=supervisor.serve_forever, daemon=True)
    thread.start()
    client = Client(socket_address(config.public_socket))
    try:
        client.request("devices.list", timeout=10)
        yield client, supervisor, robot
    finally:
        supervisor._schedule_server_shutdown()
        thread.join(timeout=5)
        supervisor.close()


def invoke(client, *, request_id=None, **extra):
    return client.device("controller").call(
        "invoke", {"tool_name": "push", "kwargs": {"dx": 0.3}, **extra}, request_id=request_id
    )


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


@pytest.mark.parametrize("action", ["stop", "ik", "fk", "unload", "policy_run", "reset"])
def test_unsupported_capabilities_never_succeed(device_service, action):
    client, _, _ = device_service
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
    env = {
        **os.environ,
        "ROBOENV_SOCKET": client.socket_path,
        "ROBOENV_RUN_DIR": str(supervisor.config.run_dir),
        "ROBOENV_VERIFIER_OUT": str(tmp_path / "verdict"),
    }
    for arguments in (
        ["devices"],
        ["device", "controller", "invoke", '{"tool_name":"push","kwargs":{"dx":0.3}}'],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "roboenv.cli", *arguments],
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


def test_thin_driver_needs_only_sdk_and_socket(device_service, tmp_path):
    client, _, _ = device_service
    driver = tmp_path / "controller/driver.py"
    driver.parent.mkdir()
    driver.write_text(
        importlib.resources.files("roboenv").joinpath("assets/device-client.py").read_text()
    )
    result = subprocess.run(
        [sys.executable, str(driver), "read", "{}"],
        env={**os.environ, "ROBOENV_SOCKET": client.socket_path},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


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
    for robot in sorted((ROOT / "catalog/robots").iterdir()):
        raw = yaml.safe_load((robot / "bindings.yaml").read_text())
        plan = BindingPlan.load(robot, backend=raw["backend"])
        context = BindingContext(raw["backend"], raw["embodiment"], object(), forbidden, {})
        service = default_registry().assemble(plan, robot, context)
        validator.validate(service.inventory())
        for device in service.inventory()["devices"]:
            assert not device["actions"].get("stop", {}).get("supported", False)
            assert "driver" not in device


def test_server_reads_cannot_bypass_submission_or_return_fake_poses(tmp_path):
    # The production LIBERO driver uses pure tcp(), which must be advertised read-only.
    class World(FakeShell):
        name, embodiment = "libero", "panda"  # CPU stub of the legacy public LIBERO surface

        def tools(self):
            return {
                **super().tools(),
                "get_hand_pose": lambda: [0, 0, 0, 1, 0, 0, 0],
                "tcp": lambda: [0, 0, 0],
                "get_step_budget": lambda: self.budget().as_dict(),
                "goto_pose": lambda *args: None,
            }

    worker = SimWorker(
        World(),
        task_id="test",
        seed=0,
        run_dir=tmp_path,
        device_root=ROOT / "catalog/robots/libero-panda",
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
        None,
        wheel,
        deployment_path=deployment,
    )
    environment = out / "environment"
    compose = yaml.safe_load((environment / "docker-compose.yaml").read_text())
    assert compose["services"]["main"]["depends_on"]["sim"]["condition"] == "service_healthy"
    assert compose["services"]["sim"]["environment"]["ROBOENV_SERVER_ROBOT"] == "/opt/serve/robot"
    assert compose["services"]["sim"]["environment"]["ROBOENV_SOCKET"] == "/run/roboenv/sim.sock"
    driver = (environment / "definition/robot/devices/controller/driver.py").read_text()
    assert "Client()" in driver and "roboenv_runtime" not in driver
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


def test_camera_artifact_uses_shared_path_and_rejects_traversal(tmp_path):
    import numpy as np

    class CameraWorld(FakeShell):
        name, embodiment = "libero", "panda"  # CPU camera stub, not a GPU replay

        def capture(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    worker = SimWorker(
        CameraWorld(),
        task_id="test",
        seed=0,
        run_dir=tmp_path,
        device_root=ROOT / "catalog/robots/libero-panda",
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
