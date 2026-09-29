"""User workflows exercise real CPU worlds and the same task resolver as export."""

import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import roboenv
import yaml
from roboenv.environment import connect, launch, status, stop
from roboenv.tasks.model import Task
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"


def launch_toy(**kwargs):
    return launch(FIXTURE / "task.yaml", project=ROOT, catalog=FIXTURE / "catalog", **kwargs)


def test_task_id_resolution_and_missing_task():
    resolved = resolve_task("libero/mug-into-microwave", project=ROOT)
    assert asdict(resolved.task)["backend"] == "libero"
    assert resolved.task.backend_kwargs == {
        "suite_name": "libero_10",
        "task_id": 9,
        "init_state_index": 0,
    }
    with pytest.raises(ValueError, match="tasks list"):
        resolve_task("not-a-task", project=ROOT)


def test_grouped_robot_resolves_owner_bundle_bindings_and_launches(tmp_path):
    catalog = tmp_path / "catalog"
    reference = "fakeshell/toy-puck"
    robot = catalog / "robots" / reference
    shutil.copytree(FIXTURE / "catalog/robots/toy-puck", robot)
    task = asdict(Task.load(FIXTURE / "task.yaml"))
    task["robot"] = reference
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    resolved = resolve_task(path, catalog=catalog)
    assert resolved.robot == robot
    assert [spec.name for spec in resolved.bindings.devices] == ["controller"]
    with launch(path, project=ROOT, catalog=catalog, timeout=20) as env:
        client = roboenv.connect(env.endpoint)
        assert client.device("controller").write([0.3])["position"][0] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "reference",
    [
        "../toy-puck",
        "libero/../toy-puck",
        "libero/./toy-puck",
        "/libero/toy-puck",
        "libero//toy-puck",
        "libero/",
        "libero\\toy-puck",
    ],
)
def test_robot_references_reject_unsafe_paths(reference):
    task = asdict(Task.load(FIXTURE / "task.yaml"))
    task["robot"] = reference
    with pytest.raises(ValueError, match="invalid task robot"):
        Task.from_dict(task)


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
        first.device("controller").call("write", {"target": [0.3]})
        assert second.device("controller").read()["budget"]["used"] == 0
        stop(first.record["endpoint"], project=ROOT)
        first.process.wait(timeout=10)
        assert stop(first.path)["state"] == "stopped"
        assert second.client.request("health")["device_service"]["ready"]


@pytest.mark.parametrize("selector", ["--port", "--endpoint"])
def test_cli_lifecycle_selects_only_requested_service(monkeypatch, tmp_path, selector):
    monkeypatch.delenv("ROBOENV_ENDPOINT", raising=False)
    monkeypatch.delenv("ROBOENV_SOCKET", raising=False)

    def target(run):
        return (
            selector,
            str(urlsplit(run.endpoint).port) if selector == "--port" else run.endpoint,
        )

    def cli(*args, code=0):
        result = subprocess.run(
            [sys.executable, "-m", "roboenv.cli.main", *args, "--json"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == code, result.stdout + result.stderr
        return json.loads(result.stdout) if code == 0 else result.stdout + result.stderr

    with (
        launch_toy(timeout=20, allow_stop=True) as first,
        launch_toy(timeout=20, allow_stop=True) as second,
    ):
        assert first.endpoint != second.endpoint
        assert "select --endpoint" in cli("status", "--project", str(ROOT), code=1)
        assert "--endpoint" in cli("stop", code=1)
        monkeypatch.setenv("ROBOENV_ENDPOINT", second.endpoint)
        assert cli("status")["session"] == second.record["session"]
        for run in (first, second):
            result = cli("status", *target(run))
            assert result["endpoint"] == run.endpoint
            assert result["simulation"]["session"] == run.record["session"]
            assert "success" not in result["simulation"]
        assert cli("status", first.endpoint)["session"] == first.record["session"]
        for command in ("status", "stop"):
            cli(command, first.endpoint, "--endpoint", second.endpoint, code=2)
            cli(command, first.endpoint, "--port", "8123", code=2)
            cli(command, "--port", "8123", "--endpoint", second.endpoint, code=2)
            cli(command, "--endpoint", "", code=1)
            cli(command, "--endpoint", "not-a-url", code=1)
            for port in ("0", "-1", "65536"):
                cli(command, "--port", port, code=1)
        assert second.client.request("health")["worker"]["alive"]
        # Explicit HTTP selection must work even without a local project or run manifest.
        stopped = cli("stop", *target(first), "--project", str(tmp_path))
        assert stopped["session"] == first.record["session"]
        assert stopped["service_state"] == "stopped"
        assert type(stopped["success"]) is bool
        assert cli("stop", *target(first)) == stopped
        assert cli("status", *target(first))["simulation"]["terminal"]
        assert cli("status")["worker"]["alive"]
        assert second.device("controller").read()["budget"]["used"] == 0
        assert connect(second.endpoint).device("controller").write([0.3])["position"][0] > 0
        assert cli("stop")["session"] == second.record["session"]


@pytest.mark.parametrize("operation", [status, stop])
def test_lifecycle_rejects_conflicting_or_empty_endpoint(operation, monkeypatch):
    monkeypatch.setenv("ROBOENV_ENDPOINT", "http://127.0.0.1:8123")
    with pytest.raises(ValueError, match="Choose a run, endpoint, or port"):
        operation("some-run", endpoint="http://127.0.0.1:8124")
    with pytest.raises(ValueError, match="Choose a run, endpoint, or port"):
        operation(endpoint="http://127.0.0.1:8124", port=8123)
    with pytest.raises(ValueError, match="Choose a run, endpoint, or port"):
        operation("some-run", port=8123)
    with pytest.raises(ValueError, match="endpoint must"):
        operation(endpoint="")
    for port in (0, -1, 65536, True, "8123"):
        with pytest.raises(ValueError, match="port must"):
            operation(port=port)


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
        == asdict(resolve_task(FIXTURE / "task.yaml", catalog=FIXTURE / "catalog").task)
    )
    assert not (out / "solution").exists()


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
            assert "realtime" not in task.backend_kwargs
    assert main(["tasks", "list", "--simulator", "robocasa", "--project", str(ROOT)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ids"] == ["robocasa/arrange-vegetables", "robocasa/pick-place-microwave"]
    assert result["execution_modes"] == {key: "realtime" for key in result["ids"]}
    with pytest.raises(ValueError, match="tasks list"):
        resolve_task("robocasa-pick-place-microwave", project=ROOT)


@pytest.mark.parametrize("value", ["robocasa/../escape", "robocasa/..", "libero/task", "robocasa/"])
def test_task_rejects_invalid_namespaced_ids(value):
    task = asdict(resolve_task("robocasa/pick-place-microwave", project=ROOT).task)
    task["id"] = value
    with pytest.raises(ValueError, match="task id"):
        Task.from_dict(task)


def test_clock_setting_must_be_at_task_level():
    task = asdict(resolve_task("robocasa/pick-place-microwave", project=ROOT).task)
    task["backend_kwargs"]["realtime"] = True
    with pytest.raises(ValueError, match="realtime is not supported"):
        Task.from_dict(task)


@pytest.mark.parametrize("mode", [None, True, "synchronous", "fast", {}, []])
def test_invalid_task_execution_modes_are_rejected(mode):
    from roboenv.tasks.model import Task

    task = asdict(Task.load(FIXTURE / "task.yaml"))
    task["execution_mode"] = mode
    with pytest.raises(ValueError, match="execution_mode"):
        Task.from_dict(task)


@pytest.mark.parametrize("override, expected", [(None, "realtime"), ("stepped", "stepped")])
def test_launch_uses_catalog_mode_unless_overridden(tmp_path, monkeypatch, override, expected):
    task = asdict(Task.load(FIXTURE / "task.yaml"))
    task["execution_mode"] = "realtime"
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    seen = []

    def capture(command, **kwargs):
        seen.append((command, kwargs["env"]))
        raise OSError("stop before starting a simulator")

    monkeypatch.setattr("roboenv.environment.subprocess.Popen", capture)
    with pytest.raises(OSError, match="stop before starting"):
        launch(
            path,
            project=ROOT,
            catalog=FIXTURE / "catalog",
            execution_mode=override,
            run_dir=tmp_path / "runs",
        )
    command, env = seen[0]
    index = command.index("--worker-arg=--execution-mode")
    assert command[index + 1] == "--worker-arg=" + expected
    definition = json.loads(Path(env["ROBOENV_SERVER_MANIFEST"]).read_text())
    assert definition["task"]["execution_mode"] == expected
    config = yaml.safe_load((Path(env["ROBOENV_TASKS"]) / "fakeshell.native-push.yaml").read_text())
    assert config["execution_mode"] == expected
    shutil.rmtree(env["ROBOENV_SOCKET_ROOT"])
