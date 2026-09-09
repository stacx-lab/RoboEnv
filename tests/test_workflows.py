"""User workflows exercise real CPU worlds and the same task resolver as export."""

import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import roboenv
import yaml
from roboenv.environment import connect, launch, stop
from roboenv.tasks.model import Task
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"


def launch_toy(**kwargs):
    return launch(FIXTURE / "task.yaml", project=ROOT, catalog=FIXTURE / "catalog", **kwargs)


def test_nested_simulation_and_flat_task_resolve_identically():
    flat = asdict(Task.load(FIXTURE / "task.yaml"))
    nested = dict(flat)
    nested["simulation"] = {
        key: nested.pop(key) for key in ("backend", "robot", "backend_kwargs", "device_bindings")
    }
    assert Task.from_dict(nested) == Task.from_dict(flat)
    nested["backend"] = "fakeshell"
    with pytest.raises(ValueError, match="mix"):
        Task.from_dict(nested)


def test_task_id_resolution_and_missing_task():
    resolved = resolve_task("libero/mug-into-microwave", project=ROOT)
    assert resolved.runtime_config()["adapter"] == "libero"
    assert resolved.task.backend_kwargs == {
        "suite_name": "libero_10", "task_id": 9, "init_state_index": 0, "realtime": True,
    }
    with pytest.raises(ValueError, match="tasks list"):
        resolve_task("not-a-task", project=ROOT)


def test_launch_connect_driver_and_budget(monkeypatch):
    monkeypatch.delenv("ROBOENV_SOCKET", raising=False)
    with launch_toy(timeout=20) as env:
        client = roboenv.connect(env.record["id"], project=ROOT)
        endpoint_client = roboenv.connect(env.record["endpoint"])
        endpoint_client.devices()
        assert endpoint_client.task.id == "native-toy-push"
        assert env.record["endpoint"].count("/") == 2
        assert client.task.id == "native-toy-push"
        assert env.devices()["devices"][0]["name"] == "controller"
        before = client.device("controller").read()
        assert before["result"]["budget"]["used"] == 0
        result = client.device("controller").write([0.3])
        assert result["position"][0] == pytest.approx(0.3)
        assert env.device("controller").read()["budget"]["used"] > 0
        device = Path(env.record["devices"]) / "controller"
        profile = yaml.safe_load((device / "profile.yaml").read_text())
        assert "stop" not in profile["access"]["driver"]["actions"]
        driver = subprocess.run(
            [sys.executable, str(device / "driver.py"), "read", "{}"],
            cwd=device,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert driver.returncode == 0, driver.stderr
        assert json.loads(driver.stdout)["ok"] is True
        record_path = env.path
    assert env.process.poll() == 0
    assert json.loads(record_path.read_text())["state"] == "stopped"


def test_multiple_runs_require_selection(monkeypatch):
    monkeypatch.delenv("ROBOENV_SOCKET", raising=False)
    with launch_toy(timeout=20) as first, launch_toy(timeout=20) as second:
        with pytest.raises(ValueError, match="select --endpoint"):
            connect(project=ROOT)
        assert first.client.session != second.client.session
        assert connect(first.endpoint).task.id == "native-toy-push"
        assert first.record["endpoint"] != second.record["endpoint"]
        first.device("controller").write([0.3])
        assert second.device("controller").read()["budget"]["used"] == 0
        stop(first.record["endpoint"], project=ROOT)
        first.process.wait(timeout=10)
        assert stop(first.path)["state"] == "stopped"
        assert second.client.request("health")["device_service"]["ready"]


def test_launch_validation_precedes_process_creation(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(ValueError, match="executable"):
        launch_toy(interpreter="/missing/interpreter")
    with pytest.raises(ValueError, match="positive finite"):
        launch_toy(timeout=float("nan"))
    with pytest.raises(ValueError, match="port"):
        launch_toy(port=-1)


def test_occupied_port_fails_before_a_world_is_started():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="supervisor exited"):
            launch_toy(port=listener.getsockname()[1], timeout=5)


def test_reused_port_accepts_a_fresh_connection():
    from roboenv.client import RemoteError

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with launch_toy(port=port, timeout=20) as first:
        old_endpoint = first.record["endpoint"]
        assert old_endpoint == f"http://127.0.0.1:{port}"
        old_client = connect(old_endpoint)
        old_client.devices()
    with launch_toy(port=port, timeout=20) as second:
        assert second.record["endpoint"] == old_endpoint
        with pytest.raises(RemoteError):
            old_client.device("controller").write([0.3])
        assert second.device("controller").read()["budget"]["used"] == 0
        assert connect(old_endpoint).device("controller").write([0.3])["position"][0] > 0


def test_startup_failure_preserves_logs(tmp_path):
    interpreter = tmp_path / "broken-python"
    interpreter.write_text("#!/bin/sh\nexit 1\n")
    interpreter.chmod(0o700)
    with pytest.raises(RuntimeError, match="supervisor.log"):
        launch_toy(interpreter=interpreter, timeout=5)


def test_worker_exit_fails_launch_without_waiting_for_deadline(monkeypatch):
    from roboenv.client import Client

    request = Client.request

    def worker_exited(self, op, *args, **kwargs):
        result = request(self, op, *args, **kwargs)
        if op == "health":
            result["worker"]["state"] = "exited"
        return result

    monkeypatch.setattr(Client, "request", worker_exited)
    with pytest.raises(RuntimeError, match="Simulator worker failed during startup"):
        launch_toy(timeout=5)


def test_cli_detach_status_and_stop():
    command = [sys.executable, "-m", "roboenv.cli.main"]
    env = {k: v for k, v in os.environ.items() if k != "ROBOENV_SOCKET"}
    started = subprocess.run(
        [
            *command,
            "launch",
            str(FIXTURE / "task.yaml"),
            "--catalog",
            str(FIXTURE / "catalog"),
            "--project",
            str(ROOT),
            "--detach",
            "--json",
            "--timeout",
            "20",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert started.returncode == 0, started.stdout + started.stderr
    assert "Simulator logs:" in started.stderr
    assert "worker ready:" in started.stderr
    record = json.loads(started.stdout)
    try:
        result = subprocess.run(
            [*command, "devices", "--endpoint", record["endpoint"]],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout
        assert json.loads(result.stdout)["devices"][0]["name"] == "controller"
        result = subprocess.run(
            [*command, "status", record["endpoint"]],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["device_service"]["ready"]
        result = subprocess.run(
            [*command, "stop", record["endpoint"], "--project", str(ROOT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        stop(record["id"], project=ROOT)


@pytest.mark.parametrize("prefix", [None, "new-environment"])
def test_prepare_uses_managed_default_or_explicit_prefix(monkeypatch, prefix):
    from roboenv.cli.main import main
    from roboenv.provisioning import setup

    calls = []
    monkeypatch.setattr(setup, "prepare", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(setup, "doctor", lambda *args, **kwargs: {"install_preflight_passed": True})
    command = ["prepare", "libero"]
    if prefix is not None:
        command.extend(["--prefix", prefix])
    assert main(command) == 0
    assert len(calls) == 1
    assert callable(calls[0].pop("progress"))
    assert calls == [{"project": None, "install_to": Path(prefix) if prefix else None}]
    calls.clear()
    assert main(["doctor", "libero"]) == 0
    assert not calls


def test_prepare_reports_install_failure(monkeypatch, capsys):
    from roboenv.cli.main import main
    from roboenv.provisioning import setup

    def fail(*args, **kwargs):
        raise ValueError("installation blocked")

    monkeypatch.setattr(setup, "prepare", fail)
    assert main(["prepare", "libero"]) == 1
    assert "installation blocked" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["style", "mode", "max_turns", "options"])
def test_runtime_rejects_agent_configuration(field):
    from roboenv_runtime.tasks.schema import TaskSpec

    with pytest.raises(ValueError, match="non-environment"):
        TaskSpec.from_dict({"id": "test", "adapter": "fakeshell", field: "agent setting"})


def test_simple_export_uses_the_shared_resolver(tmp_path):
    from roboenv.harbor.workflow import export

    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"test-only wheel; never installed")
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "fakeshell",
                "image": "sha256:" + "a" * 64,
                "legacy_tree_required": False,
                "mounts": [],
            }
        )
    )
    out = tmp_path / "task"
    export(
        FIXTURE / "task.yaml",
        project=ROOT,
        catalog=FIXTURE / "catalog",
        out=out,
        deployment=deployment,
        wheel=wheel,
    )
    generated = yaml.safe_load(
        (out / "environment/sim/tasks/fakeshell.native-push.yaml").read_text()
    )
    assert (
        generated
        == resolve_task(FIXTURE / "task.yaml", catalog=FIXTURE / "catalog").runtime_config()
    )
    assert not (out / "solution").exists()


def test_cli_rejects_removed_start_command(capsys):
    from roboenv.cli.main import main

    with pytest.raises(SystemExit) as error:
        main(["start", "example-task"])
    assert error.value.code == 2
    assert "invalid choice: 'start'" in capsys.readouterr().err


def test_launch_never_changes_catalog():
    def snapshot():
        return {
            str(p.relative_to(FIXTURE / "catalog")): p.read_bytes()
            for p in (FIXTURE / "catalog").rglob("*")
            if p.is_file()
        }

    before = snapshot()
    with launch_toy(timeout=20):
        assert snapshot() == before
    assert snapshot() == before


def test_grouped_catalog_ids_and_listing(capsys):
    from roboenv.cli.main import main
    from roboenv.tasks.catalog import load_catalog

    catalog = load_catalog(ROOT / "catalog/tasks")
    assert len(catalog) == 15
    for identifier, (task, path) in catalog.items():
        assert path.relative_to(ROOT / "catalog/tasks").as_posix() == identifier + "/task.yaml"
        assert identifier.split("/")[0] == task.backend
        assert resolve_task(identifier, project=ROOT).task == task
        if task.backend == "robocasa":
            assert task.backend_kwargs["realtime"] is True
    assert main(["tasks", "list", "--simulator", "robocasa", "--project", str(ROOT)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ids"] == ["robocasa/arrange-vegetables", "robocasa/pick-place-microwave"]
    with pytest.raises(ValueError, match="tasks list"):
        resolve_task("robocasa-pick-place-microwave", project=ROOT)


@pytest.mark.parametrize("value", ["robocasa/../escape", "robocasa/..", "libero/task", "robocasa/"])
def test_task_rejects_invalid_namespaced_ids(value):
    task = asdict(resolve_task("robocasa/pick-place-microwave", project=ROOT).task)
    task["id"] = value
    with pytest.raises(ValueError, match="task id"):
        Task.from_dict(task)


@pytest.mark.parametrize("realtime", [None, True, False])
def test_clock_override_reaches_runtime_without_catalog_edits(realtime):
    from roboenv_runtime.tasks.schema import TaskSpec

    original = resolve_task("robocasa/pick-place-microwave", project=ROOT)
    before = original.path.read_bytes()
    resolved = resolve_task(original.path, project=ROOT, realtime=realtime)
    expected = realtime if realtime is not None else True
    assert resolved.task.backend_kwargs["realtime"] is expected
    assert resolved.runtime_config()["adapter_kwargs"]["realtime"] is expected
    assert TaskSpec.from_dict(asdict(resolved.task)).adapter_kwargs["realtime"] is expected
    assert original.path.read_bytes() == before
    assert original.task.backend_kwargs["realtime"] is True


def test_clock_validation_precedes_launch_effects(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(ValueError, match="Clock selection is not supported"):
        launch("robotwin/handover-mic", project=ROOT, realtime=True)
    with pytest.raises(ValueError, match="boolean"):
        launch("robocasa/pick-place-microwave", project=ROOT, realtime="false")
    task = asdict(resolve_task("robocasa/pick-place-microwave", project=ROOT).task)
    task["backend_kwargs"]["realtime"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        Task.from_dict(task)


@pytest.mark.parametrize(
    "flags, expected", [([], None), (["--realtime"], True), (["--sync"], False)]
)
def test_launch_cli_clock_selection(monkeypatch, flags, expected):
    from roboenv.cli.main import main

    class Selected(Exception):
        pass

    calls = []

    def selected(task, **kwargs):
        calls.append((task, kwargs["realtime"]))
        raise Selected("clock reached launch")

    monkeypatch.setattr("roboenv.environment.launch", selected)
    assert main(["launch", "robocasa/pick-place-microwave", *flags]) == 1
    assert calls == [("robocasa/pick-place-microwave", expected)]
    with pytest.raises(SystemExit) as error:
        main(["launch", "robocasa/pick-place-microwave", "--realtime", "--sync"])
    assert error.value.code == 2
