"""Fresh real-worker fixtures shared by wire and binding conformance tests."""

import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from roboenv.client import Client
from roboenv_runtime.runtime.protocol import socket_address
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def running_service(tmp_path, monkeypatch, *, variant=None):
    """A fresh CPU worker behind a real supervisor socket, ready to list its devices.

    ``variant`` selects the packaged native binding (``"native"``) or a public HTTP
    endpoint (``"http"``); the default injects a test-only fallible binding. Environment
    variables set through ``monkeypatch`` beforehand reach the supervisor.
    """
    robot = tmp_path / "robot"
    shutil.copytree(ROOT / "tests/fixtures/native-toy/catalog/robots/toy-puck", robot)
    # A test-only binding injects partial effects, delay, and read violations.
    if variant != "native":
        (robot / "bindings.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "roboenv/bindings/1",
                    "backend": "fakeshell",
                    "devices": {"controller": {"binding": "test.device/1", "entity": "controller"}},
                }
            )
        )
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "device-toy.yaml").write_text(
        "id: device-toy\nbackend: fakeshell\n"
        "backend_task: device-toy\nrobot: toy-puck\ninstruction: Push the puck.\n"
    )
    monkeypatch.setenv("ROBOENV_SERVER_ROBOT", str(robot))
    monkeypatch.delenv("ROBOENV_SERVER_MANIFEST", raising=False)
    monkeypatch.delenv("ROBOENV_SOCKET_ROOT", raising=False)
    monkeypatch.setenv("ROBOENV_TASKS", str(tasks))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(ROOT / "runtime/src"), str(ROOT / "src"), str(ROOT / "tests")]),
    )
    config = SupervisorConfig(
        run_dir=tmp_path / "run",
        task="device-toy",
        seed=0,
        budget_mode="official",
        session="local",
        cache_entries=2,
        terminate_grace_s=0.2,
        worker_module="device_test_worker",
        http_port=0 if variant == "http" else None,
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


@pytest.fixture
def device_service(tmp_path, monkeypatch, request):
    variant = getattr(request, "param", None)
    with running_service(tmp_path, monkeypatch, variant=variant) as service:
        yield service


def invoke(client, *, request_id=None, **extra):
    return client.device("controller").call(
        "write", {"target": [0.3], **extra}, request_id=request_id
    )
