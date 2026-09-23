"""Full supervisor/worker device concurrency, including native owner-thread scheduling."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from helpers import kinex_sources
from roboenv.client import Client, RemoteError
from roboenv.devices.export import write_driver
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def execution_mode():
    return "stepped"


@pytest.fixture(params=[False, True], ids=["thread-safe", "owner-thread"])
def concurrent_service(tmp_path, monkeypatch, request, execution_mode):
    robot = tmp_path / "robot"
    for name in ("left", "right"):
        directory = robot / "devices" / name
        directory.mkdir(parents=True)
        (directory / "profile.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "type": "joint",
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
                    for name in ("left", "right")
                },
            }
        )
    )
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "concurrent.yaml").write_text("id: concurrent\nadapter: fakeshell\ntext: Test.\n")
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
            task="concurrent",
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


def test_independent_writes_overlap_and_reads_progress(concurrent_service):
    client, _, owner = concurrent_service
    with ThreadPoolExecutor() as pool:
        left = pool.submit(client.device("left").write, [[0.4]], freq_hz=1.25)
        first = wait_commands(client, "left", 1)
        right = pool.submit(client.device("right").write, [[0.7]], freq_hz=5.0)
        wait_commands(client, "right", 1)
        samples = [client.device("left").read()["data"] for _ in range(8)]
        assert not left.done()
        assert all(sample["owner_only"] for sample in samples) if owner else True
        assert max(sample["sample_time_s"] for sample in samples) >= first["sample_time_s"]
        right_result = right.result(timeout=2)["data"]
        assert not left.done()
        assert 0.2 <= right_result["elapsed_s"] <= 0.22
        left_result = left.result(timeout=2)["data"]
        assert 0.8 <= left_result["elapsed_s"] <= 0.82
        assert right_result["state"]["sample_time_s"] < left_result["state"]["sample_time_s"]


def test_stop_cancels_active_and_queued_only_in_its_scope(concurrent_service):
    client, _, _ = concurrent_service
    with ThreadPoolExecutor() as pool:
        active = pool.submit(client.device("left").write, [[0.3]], freq_hz=0.5)
        wait_commands(client, "left", 1)
        queued = pool.submit(
            client.device("left").write, [[0.9]], freq_hz=10.0, request_id="queued"
        )
        peer = pool.submit(client.device("right").write, [[0.8]], freq_hz=2.5)
        wait_commands(client, "right", 1)
        deadline = time.monotonic() + 2
        while client.operation("queued")["state"] != "running":
            assert time.monotonic() < deadline
        assert client.device("left").stop()["data"]["stopped"]
        for future in (active, queued):
            with pytest.raises(RemoteError, match="interrupted"):
                future.result(timeout=1)
        assert len(client.device("left").read()["data"]["commands"]) == 2
        assert peer.result(timeout=2)["data"]["completed_actions"] == 1
        assert client.device("left").write([[0.1]], freq_hz=25.0)["data"]["completed_actions"] == 1


def test_concurrent_duplicate_id_executes_once(concurrent_service):
    client, _, _ = concurrent_service
    with ThreadPoolExecutor() as pool:
        original = pool.submit(
            client.device("left").write, [[0.2]], freq_hz=3.3333333333333335, request_id="once"
        )
        wait_commands(client, "left", 1)
        with pytest.raises(RemoteError) as error:
            client.device("left").write([[0.2]], freq_hz=3.3333333333333335, request_id="once")
        assert error.value.code == "request_in_progress"
        result = original.result(timeout=2)
        assert (
            client.device("left").write([[0.2]], freq_hz=3.3333333333333335, request_id="once")
            == result
        )
        assert len(client.device("left").read()["data"]["commands"]) == 1


def test_kinex_recording_through_portable_drivers(concurrent_service, tmp_path):
    client, supervisor, _ = concurrent_service
    kinex, _ = kinex_sources()
    node = shutil.which("node")
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("Set KINEX_ROOT to a built Kinex checkout and use Node >=22.19")
    workspace = tmp_path / "workspace"
    for name in ("left", "right"):
        directory = workspace / ".kinex/devices" / name
        write_driver(directory, "other")
        (directory / "connection.json").write_text(
            json.dumps(
                {
                    "endpoint": supervisor.http_endpoint,
                    "session": "concurrent",
                    "generation": 0,
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


def test_read_timeout_does_not_kill_an_independent_write(concurrent_service, monkeypatch):
    client, supervisor, _ = concurrent_service
    original = supervisor._proxy

    def proxy(request, deadline):
        if request.op == "devices.call" and request.args["args"].get("slow"):
            raise TimeoutError("slow observation transport")
        return original(request, deadline)

    monkeypatch.setattr(supervisor, "_proxy", proxy)
    with ThreadPoolExecutor() as pool:
        write = pool.submit(client.device("left").write, [[0.2]], freq_hz=5.0)
        wait_commands(client, "left", 1)
        with pytest.raises(RemoteError) as error:
            client.device("right").read(slow=True)
        assert error.value.details["worker_killed"] is False
        assert write.result(timeout=2)["data"]["completed_actions"] == 1


def test_invalid_stop_and_lifecycle_requests_do_not_interrupt_motion(concurrent_service):
    client, _, _ = concurrent_service
    with ThreadPoolExecutor() as pool:
        write = pool.submit(client.device("left").write, [[0.2]], freq_hz=5.0)
        wait_commands(client, "left", 1)
        with pytest.raises(RemoteError):
            client.request("simulation.stop", {"invalid": True})
        assert write.result(timeout=2)["data"]["completed_actions"] == 1


def test_kinex_buffers_and_upper_layer_planning(concurrent_service, tmp_path):
    client, supervisor, _ = concurrent_service
    kinex, _ = kinex_sources()
    node = shutil.which("node")
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("requires built Kinex and Node >=22.19")
    workspace = tmp_path / "buffer-workspace"
    for name in ("left", "right"):
        directory = workspace / ".kinex/devices" / name
        write_driver(directory, "joint")
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
        assert [entry["frame"] - start for entry in commands] == [0, 17, 33]


def test_portable_cleanup_only_cancels_its_own_request(concurrent_service, tmp_path):
    from test_persistent_driver import transport_module

    client, supervisor, _ = concurrent_service
    wire_module = transport_module()
    wire = wire_module.Transport(tmp_path / "driver.py", device="left")
    wire.config = {"endpoint": supervisor.http_endpoint}
    try:
        with ThreadPoolExecutor() as pool:
            old = pool.submit(wire.remote, "write", {"actions": [[0.3]], "freq_hz": 0.5})
            wait_commands(client, "left", 1)
            new = pool.submit(client.device("left").write, [[0.7]], freq_hz=2)
            wait_commands(client, "left", 2)
            with pytest.raises(wire_module.WireError):
                old.result(timeout=2)
            wire.cancel_motions()
            assert new.result(timeout=2)["data"]["completed_actions"] == 1
            assert client.device("left").read()["data"]["positions"] == [0.7]
    finally:
        wire.close_all()


@pytest.mark.parametrize("execution_mode", ["realtime"])
def test_realtime_worker_advances_while_idle(concurrent_service):
    client, _, owner = concurrent_service
    first = client.device("left").read()["data"]["sample_time_s"]
    time.sleep(0.1)
    later = client.device("left").read()["data"]
    assert later["sample_time_s"] > first
    assert not owner or later["owner_only"]
    result = client.device("left").write([[0.3]], freq_hz=10)["data"]
    assert result["reference_state"] == "target"
    assert result["elapsed_s"] == 0.1
    time.sleep(0.06)
    assert client.device("left").read()["data"]["positions"] == [0.3]
