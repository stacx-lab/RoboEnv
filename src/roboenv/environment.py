"""Local environment ownership and connection discovery.

Launch delegates world ownership to the existing runtime supervisor. This
module imports no simulator and never executes an agent program.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml
from roboenv.client import Client
from roboenv.project import project_root
from roboenv.tasks.resolve import resolve_task
from roboenv.validation import json_loads


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _client(record):
    return Client(
        record.get("endpoint") or record["socket"],
        session=record["session"],
        generation=record["generation"],
    )


def _records(project):
    return project_root(project) / ".cache/roboenv/runs"


def _record_path(run, project):
    if str(run).startswith(("http://", "https://")):
        endpoint = str(run).rstrip("/")
        matches = []
        for candidate in _records(project).glob("*/run.json"):
            record = json_loads(candidate.read_text())
            if record.get("endpoint") == endpoint and record.get("state") == "ready":
                if Path(record["socket"]).exists():
                    matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f"Expected one running local environment at {endpoint}")
        return matches[0]
    path = Path(run)
    if path.is_dir():
        path /= "run.json"
    if not path.is_file():
        if Path(run).name != str(run):
            raise ValueError("Select a run ID or an existing run.json")
        path = _records(project) / str(run) / "run.json"
    return path


def _record(run, project):
    value = json_loads(_record_path(run, project).read_text())
    if value.get("schema") != "roboenv/local-run/1":
        raise ValueError("Unsupported local run manifest")
    return value


def _connect(
    run=None, *, endpoint=None, socket_path=None, session=None, generation=None, project=None
):
    """Attach without acquiring ownership. Explicit endpoint, run, then workspace."""
    if sum(value is not None for value in (run, endpoint, socket_path)) > 1:
        raise ValueError("Choose a run, endpoint, or socket")
    if run is not None and str(run).startswith(("http://", "https://")):
        endpoint, run = str(run), None
    if run is not None:
        return _client(_record(run, project))
    if (
        endpoint
        or socket_path
        or os.environ.get("ROBOENV_ENDPOINT")
        or os.environ.get("ROBOENV_SOCKET")
    ):
        return Client(socket_path, endpoint=endpoint, session=session, generation=generation)
    try:
        root = _records(project)
    except ValueError:
        return Client(session=session, generation=generation)
    live = []
    for path in sorted(root.glob("*/run.json")):
        record = _record(path, project)
        if record.get("state") != "ready" or not Path(record["socket"]).exists():
            continue
        candidate = _client(record)
        try:
            health = candidate.request("health", timeout=0.2)
        except (OSError, RuntimeError):
            continue
        if health.get("device_service", {}).get("ready"):
            live.append(candidate)
    if len(live) != 1:
        raise ValueError(
            f"Found {len(live)} ready environments; "
            "start one with 'roboenv launch TASK' or select --endpoint URL"
        )
    return live[0]


class _OwnerClient(Client):
    allowed_ops = Client.allowed_ops | {"shutdown"}


def stop(run, *, project=None):
    """Ask the identified supervisor to stop; never signal a recorded PID."""
    path = _record_path(run, project)
    record = _record(path, project)
    if record.get("state") == "stopped":
        return {"state": "stopped", "id": record["id"]}
    owner = _OwnerClient(
        record["socket"], session=record["session"], generation=record["generation"]
    )
    owner.request("shutdown", timeout=15)
    deadline = time.monotonic() + 15
    while Path(record["socket"]).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Shutdown requested but supervisor has not closed its socket")
        time.sleep(0.05)
    record["state"] = "stopped"
    _save(path, record)
    return {"state": "stopped", "id": record["id"]}


def connect(*args, **kwargs):
    from roboenv.devices.client import Environment

    return Environment(_connect(*args, **kwargs))


def status(run=None, *, project=None):
    if str(run).startswith(("http://", "https://")):
        return _connect(run).request("health", timeout=2)
    if run is not None:
        record = _record(run, project)
        if record.get("state") in {"stopped", "failed"}:
            return {"state": record["state"], "id": record["id"]}
    client = _connect(run, project=project)
    return client.request("health", timeout=2)


class RunningEnvironment:
    """Owner handle. Context exit stops only the process launched by this handle."""

    def __init__(self, process, record, path):
        self.process, self.record, self.path = process, record, path
        self.client = _client(record)

    @property
    def endpoint(self):
        return self.record["endpoint"]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def devices(self):
        return self.client.devices()

    def device(self, name):
        return self.client.device(name)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Supervisor did not stop; inspect the preserved run logs"
                ) from exc
        self.record["state"] = "stopped"
        _save(self.path, self.record)


def launch(
    task,
    *,
    project=None,
    catalog=None,
    interpreter=None,
    timeout=900,
    host="127.0.0.1",
    port=0,
    realtime=None,
):
    """Start a task in its isolated interpreter; return after device readiness.

    realtime=None preserves the task clock; True/False override it for RoboCasa.
    Backend assets must already be prepared. No installation or download occurs.
    All run definitions and logs are retained under the selected project's cache.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    import ipaddress

    ipaddress.IPv4Address(host)
    root = project_root(project)
    resolved = resolve_task(task, project=root, catalog=catalog, realtime=realtime)
    spec = resolved.task
    if interpreter is None:
        if spec.backend == "fakeshell":
            interpreter = sys.executable
        else:
            registration = root / ".cache/roboenv/installations" / f"{spec.backend}.json"
            if not registration.is_file():
                raise ValueError(
                    f"Simulator is not prepared; run 'roboenv prepare {spec.backend}' "
                    "or pass --interpreter"
                )
            interpreter = json_loads(registration.read_text())["python"]
    interpreter = Path(interpreter).absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("Simulator interpreter must be an existing executable")
    if any(path.is_symlink() for path in resolved.robot.rglob("*")):
        raise ValueError("Robot definition must not contain symlinks")
    if spec.backend != "fakeshell":
        from roboenv.provisioning.sources import status as source_status

        if not source_status(root, [spec.backend])["ready"]:
            raise ValueError(f"Sources are not ready; run 'roboenv doctor {spec.backend}'")
    run_id = uuid.uuid4().hex[:16]
    directory = _records(root) / run_id
    directory.mkdir(parents=True, mode=0o700)
    sockets = Path(tempfile.mkdtemp(prefix="roboenv-socket-", dir="/tmp"))
    robot = directory / "server/robot"
    shutil.copytree(resolved.robot, robot)
    (robot / "bindings.yaml").write_text(
        yaml.safe_dump(resolved.bindings.as_dict(), sort_keys=False)
    )
    tasks = directory / "server/tasks"
    tasks.mkdir()
    (tasks / f"{spec.backend_task}.yaml").write_text(
        yaml.safe_dump(resolved.runtime_config(), sort_keys=False)
    )
    manifest = directory / "environment.json"
    _save(
        manifest,
        {
            "task": asdict(spec),
            "observation_access": "sensors-only"
            if spec.backend == "robocasa"
            else "benchmark-ground-truth",
        },
    )
    record = {
        "schema": "roboenv/local-run/1",
        "id": run_id,
        "task": spec.id,
        "socket": str(sockets / "sim.sock"),
        "session": run_id,
        "generation": 0,
        "state": "starting",
        "directory": str(directory),
        "devices": str(directory / "devices"),
    }
    record_path = directory / "run.json"
    _save(record_path, record)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("ROBOENV_") or key in {"PYTHONPATH", "PYTHONHOME"}:
            env.pop(key)
    env.update(
        ROBOENV_ROOT=str(root),
        ROBOENV_TASKS=str(tasks),
        ROBOENV_SERVER_ROBOT=str(robot),
        ROBOENV_SERVER_MANIFEST=str(manifest),
        ROBOENV_SOCKET=record["socket"],
        ROBOENV_SOCKET_ROOT=str(sockets),
        ROBOENV_SESSION=run_id,
        ROBOENV_GENERATION="0",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="egl",
    )
    command = [
        str(interpreter),
        "-m",
        "roboenv_runtime.runtime.supervisor",
        "--task",
        spec.backend_task,
        "--seed",
        str(spec.seed),
        "--budget-mode",
        spec.budget_mode,
        "--session",
        run_id,
        "--run-dir",
        str(directory / "evidence"),
        "--socket",
        record["socket"],
        "--worker-socket",
        str(sockets / "worker.sock"),
        "--worker-startup-timeout",
        str(timeout),
        "--port",
        str(port),
        "--host",
        host,
    ]
    try:
        with (directory / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError:
        record["state"] = "failed"
        _save(record_path, record)
        raise
    running = RunningEnvironment(process, record, record_path)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Simulator supervisor exited during startup")
            if Path(record["socket"]).exists():
                health = running.client.request("health", timeout=1)
                if health.get("worker", {}).get("state") == "ready":
                    inventory = running.devices()
                    record["endpoint"] = health["endpoint"]
                    from roboenv.devices.export import export_devices

                    export_devices(
                        inventory["devices"],
                        directory / "devices",
                        {key: record[key] for key in ("endpoint", "session", "generation")},
                    )
                    record["state"] = "ready"
                    _save(record_path, record)
                    running.client = _client(record)
                    return running
                if health.get("worker", {}).get("state") in {"failed", "terminated", "timed_out"}:
                    raise RuntimeError("Simulator worker failed during startup")
            time.sleep(0.05)
        raise TimeoutError("Simulator did not become ready before the startup deadline")
    except BaseException as exc:
        running.close()
        record["state"] = "failed"
        _save(record_path, record)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(
            f"{exc}; inspect {directory / 'supervisor.log'} and evidence/logs"
        ) from exc
