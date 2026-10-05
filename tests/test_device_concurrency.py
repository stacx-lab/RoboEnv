"""Full supervisor/worker device concurrency, including native owner-thread scheduling."""

import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from helpers import kinex_sources, load_script
from roboenv.client import Client, RemoteError
from roboenv.devices.export import driver_access, write_driver
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def execution_mode():
    return "stepped"


@pytest.fixture(params=[False, True], ids=["thread-safe", "owner-thread"])
def concurrent_service(tmp_path, monkeypatch, request, execution_mode):
    robot = tmp_path / "robot"
    for name in ("left", "right", "world"):
        directory = robot / "devices" / name
        directory.mkdir(parents=True)
        (directory / "profile.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "type": "world" if name == "world" else "joint",
                    "intrinsics": {
                        "model": {
                            "joints": [
                                {
                                    "name": "axis",
                                    "type": "revolute",
                                    "limits": {"min": -2, "max": 2},
                                }
                            ]
                        },
                        "control": {"mode": "position"},
                        "feedback": {"velocities": True},
                        "stop": {"behavior": "hold", "affected_devices": [name]},
                    },
                }
            )
        )
    (robot / "bindings.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "roboenv/bindings/1",
                "backend": "fakeshell",
                "devices": {
                    name: {"binding": "test.concurrent/1", "entity": name}
                    for name in ("left", "right", "world")
                },
            }
        )
    )
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "concurrent.yaml").write_text(
        "id: fakeshell/concurrent\n"
        "robot: toy-puck\ninstruction: Test.\n"
    )
    monkeypatch.setenv("ROBOENV_TASKS", str(tasks))
    monkeypatch.setenv("ROBOENV_SERVER_ROBOT", str(robot))
    monkeypatch.delenv("ROBOENV_SERVER_MANIFEST", raising=False)
    monkeypatch.setenv("TEST_OWNER_THREAD", "1" if request.param else "0")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src", "tests"))
    )
    supervisor = Supervisor(
        SupervisorConfig(
            run_dir=tmp_path / "run",
            task="fakeshell/concurrent",
            seed=0,
            budget_mode="practice",
            session="concurrent",
            http_port=0,
            worker_module="concurrent_device_worker",
            worker_extra_args=("--execution-mode", execution_mode),
            terminate_grace_s=0.2,
        )
    )
    supervisor.start_worker()
    supervisor.open_server()
    thread = threading.Thread(target=supervisor.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while supervisor.http_endpoint is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        client = Client(endpoint=supervisor.http_endpoint)
        client.devices()
        yield client, supervisor, request.param
    finally:
        supervisor._schedule_server_shutdown()
        thread.join(5)
        supervisor.close()


def wait_commands(client, name, count):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.device(name).read()["data"]
        if len(state["commands"]) == count:
            return state
        time.sleep(0.005)
    pytest.fail("write did not start")


def test_independent_writes_and_concurrent_waits_share_time(concurrent_service):
    client, _, owner = concurrent_service
    left = client.device("left").write(actions=[[0.4]], duration_s=0.8)["data"]
    right = client.device("right").write(actions=[[0.7]], duration_s=0.2)["data"]
    assert "elapsed_s" not in left and "elapsed_s" not in right
    assert client.device("world").read()["data"]["time_s"] == 0
    with ThreadPoolExecutor() as pool:
        wait = pool.submit(client.device("world").wait, 0.8)
        samples = [client.device("left").read()["data"] for _ in range(8)]
        assert all(sample["owner_only"] for sample in samples) if owner else True
        assert wait.result(timeout=3)["data"]["waited_s"] == pytest.approx(0.8)
    assert client.device("left").read()["data"]["positions"] == [0.4]
    assert client.device("right").read()["data"]["positions"] == [0.7]


def test_stop_clears_only_its_scope(concurrent_service):
    client, _, _ = concurrent_service
    client.device("left").write(actions=[[0.3]], duration_s=2)
    client.device("left").write(actions=[[0.9]], duration_s=1)
    client.device("right").write(actions=[[0.8]], duration_s=1)
    client.device("left").stop()
    client.device("world").wait(0.2)
    assert client.device("left").read()["data"]["positions"] == [0]
    assert client.device("right").read()["data"]["positions"] == [0.8]


def test_duplicate_submission_executes_once(concurrent_service):
    client, _, _ = concurrent_service
    first = client.device("left").write(actions=[[0.2]], duration_s=0.3, request_id="once")
    assert client.device("left").write(actions=[[0.2]], duration_s=0.3, request_id="once") == first
    assert len(client.device("left").read()["data"]["commands"]) == 1


@pytest.mark.parametrize("concurrent_service", [True], indirect=True, ids=["owner-thread"])
def test_kinex_recording_through_portable_drivers(concurrent_service, tmp_path):
    client, supervisor, _ = concurrent_service
    kinex, _ = kinex_sources()
    node = shutil.which(os.environ.get("NODE", "node"))
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("Set KINEX_ROOT to a built Kinex checkout and use Node >=22.19")
    workspace = tmp_path / "workspace"
    for name in ("left", "right", "world"):
        directory = workspace / ".kinex/devices" / name
        write_driver(directory, "world" if name == "world" else "other")
        (directory / "connection.json").write_text(
            json.dumps(
                {
                    "endpoint": supervisor.http_endpoint,
                    "session": "concurrent",
                    "generation": 0,
                    "device": name,
                }
            )
        )
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests/kinex_concurrency.mjs"),
            str(kinex),
            str(workspace),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not client.device("left").read()["data"]["faults"]


def test_read_timeout_does_not_kill_background_motion(concurrent_service, monkeypatch):
    client, supervisor, _ = concurrent_service
    original = supervisor._proxy

    def proxy(request, deadline):
        if request.op == "devices.call" and request.args["args"].get("slow"):
            raise TimeoutError("slow observation transport")
        return original(request, deadline)

    monkeypatch.setattr(supervisor, "_proxy", proxy)
    client.device("left").write(actions=[[0.2]], duration_s=0.2)
    with pytest.raises(RemoteError) as error:
        client.device("right").read(slow=True)
    assert error.value.details["worker_killed"] is False
    client.device("world").wait(0.2)
    assert client.device("left").read()["data"]["positions"] == [0.2]


def test_invalid_lifecycle_does_not_interrupt_motion(concurrent_service):
    client, _, _ = concurrent_service
    client.device("left").write(actions=[[0.2]], duration_s=0.2)
    with pytest.raises(RemoteError):
        client.request("simulation.stop", {"invalid": True})
    client.device("world").wait(0.2)
    assert client.device("left").read()["data"]["positions"] == [0.2]


@pytest.mark.parametrize("concurrent_service", [True], indirect=True, ids=["owner-thread"])
def test_kinex_buffers_and_upper_layer_planning(concurrent_service, tmp_path):
    client, supervisor, _ = concurrent_service
    kinex, _ = kinex_sources()
    node = shutil.which(os.environ.get("NODE", "node"))
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("requires built Kinex and Node >=22.19")
    workspace = tmp_path / "buffer-workspace"
    for name in ("left", "right", "world"):
        directory = workspace / ".kinex/devices" / name
        write_driver(directory, "world" if name == "world" else "joint")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests/kinex_buffers.mjs"),
            str(kinex),
            str(workspace),
            sys.executable,
            supervisor.http_endpoint,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    left = client.device("left").read()["data"]["commands"][:3]
    right = client.device("right").read()["data"]["commands"][:3]
    assert len(left) == len(right) == 3
    for commands in (left, right):
        start = commands[0]["frame"]
        assert [entry["frame"] - start for entry in commands] == [0, 17, 34]


def test_portable_cleanup_only_cancels_its_own_request(concurrent_service, tmp_path):
    from test_persistent_driver import transport_module

    client, supervisor, _ = concurrent_service
    wire_module = transport_module()
    wire = wire_module.Transport(tmp_path / "driver.py", device="left")
    wire.config = {"endpoint": supervisor.http_endpoint}
    try:
        wire.remote("write", {"actions": [[0.3]], "duration_s": 1})
        client.device("left").write(actions=[[0.7]], duration_s=1)
        wire.cancel_motions()
        client.device("world").wait(0.2)
        assert client.device("left").read()["data"]["positions"] == [0.7]
    finally:
        wire.close_all()


def test_portable_world_read_reports_episode_budget(concurrent_service, tmp_path):
    _, supervisor, _ = concurrent_service
    devices = tmp_path / "devices"
    for name, kind, actions in (
        ("world", "world", ["read", "wait", "status"]),
        ("left", "joint", ["read"]),
    ):
        directory = devices / name
        write_driver(directory, kind)
        (directory / "profile.yaml").write_text(yaml.safe_dump({
            "name": name, "type": kind,
            "access": driver_access(kind, name, actions, supervisor.http_endpoint),
        }))
        (directory / "connection.json").write_text(json.dumps({
            "endpoint": supervisor.http_endpoint, "session": "concurrent", "generation": 0,
        }))
    proc = subprocess.Popen(
        [sys.executable, str(devices / "world/driver.py"), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    replies = {}
    try:
        for tag, action, args in (
            ("before", "read", {}),
            ("wait", "wait", {"duration_s": 0.2}),
            ("status", "status", {}),
            ("after", "read", {}),
        ):
            proc.stdin.write(json.dumps({"id": tag, "action": action, "args": args}) + "\n")
            proc.stdin.flush()
            assert select.select([proc.stdout], [], [], 5)[0], f"No reply for {tag}"
            replies[tag] = json.loads(proc.stdout.readline())
            assert replies[tag]["id"] == tag and replies[tag]["ok"], replies[tag]
        proc.stdin.close()
        assert proc.wait(timeout=5) == 0, proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
    assert replies["before"]["budget"] == {"used_s": 0.0, "max_s": None}
    assert replies["after"]["budget"] == {"used_s": pytest.approx(0.2), "max_s": None}
    assert "budget" not in replies["wait"]
    assert replies["wait"]["waited_s"] == pytest.approx(0.2)
    assert replies["status"] == {"id": "status", "ok": True, "loaded": True}
    wire = load_script("src/roboenv/assets/_roboenv.py").Transport(devices / "left/driver.py")
    try:
        assert "budget" not in wire.remote("read", {})
    finally:
        wire.close_all()


@pytest.mark.parametrize("execution_mode", ["realtime"])
def test_realtime_worker_advances_while_idle(concurrent_service):
    client, _, owner = concurrent_service
    first = client.device("left").read()["data"]["timestamp"]
    time.sleep(0.1)
    later = client.device("left").read()["data"]
    assert later["timestamp"] > first
    clock = client.device("world").read()["data"]
    assert clock["budget"] == {"used_s": pytest.approx(clock["time_s"]), "max_s": None}
    assert not owner or later["owner_only"]
    result = client.device("left").write(actions=[[0.3]], duration_s=0.1)["data"]
    assert result["execution_id"]
    assert "elapsed_s" not in result
    time.sleep(0.06)
    assert client.device("left").read()["data"]["positions"] == [0.3]
