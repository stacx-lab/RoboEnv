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
import threading
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from roboenv.client import Client
from roboenv.devices.status import status_feedback
from roboenv.project import project_root
from roboenv.provisioning.interpreter import resolve_interpreter
from roboenv.tasks.resolve import resolve_task
from roboenv.validation import json_loads


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _client(record, *, owner=False):
    # The owner reaches its run through the Unix socket; the public HTTP service refuses owner
    # operations such as observe and score.peek.
    return Client(
        record["socket"] if owner else record.get("endpoint") or record["socket"],
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


def load_run_record(run, project=None):
    """Resolve a run ID, run directory, run.json path or endpoint to its launch record.

    Returns the record's path and content (schema ``roboenv/local-run/1``); the launch
    manifest ``environment.json`` sits beside that path.
    """
    path = _record_path(run, project)
    value = json_loads(path.read_text())
    if value.get("schema") != "roboenv/local-run/1":
        raise ValueError("Unsupported local run manifest")
    return path, value


def _record(run, project):
    return load_run_record(run, project)[1]


def _connect(
    run=None,
    *,
    endpoint=None,
    socket_path=None,
    session=None,
    generation=None,
    project=None,
    owner=False,
):
    """Attach without acquiring ownership. Explicit endpoint, run, then workspace.

    owner selects a local run's own socket, for owner operations; an endpoint cannot offer them.
    """
    if sum(value is not None for value in (run, endpoint, socket_path)) > 1:
        raise ValueError("Choose a run, endpoint, or socket")
    if run is not None and str(run).startswith(("http://", "https://")):
        endpoint, run = str(run), None
    if owner and endpoint:
        raise ValueError("Owner operations need a local run, not an endpoint: select it with --run")
    if run is not None:
        return _client(_record(run, project), owner=owner)
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
        candidate = _client(record, owner=owner)
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


def _lifecycle_endpoint(run, endpoint, port):
    if sum(value is not None for value in (run, endpoint, port)) > 1:
        raise ValueError("Choose a run, endpoint, or port")
    if port is not None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        return f"http://127.0.0.1:{port}"
    if endpoint is not None and not endpoint:
        raise ValueError("endpoint must be http(s)://HOST:PORT")
    return endpoint


def stop(run=None, *, endpoint=None, port=None, project=None):
    """Stop one selected service; port selects a local HTTP service."""
    endpoint = _lifecycle_endpoint(run, endpoint, port)
    if endpoint is not None:
        return Client(endpoint=endpoint).request("simulation.stop")
    run = run or os.environ.get("ROBOENV_ENDPOINT")
    if run is None:
        raise ValueError("Select a run, use --port PORT / --endpoint URL, or set ROBOENV_ENDPOINT")
    if str(run).startswith(("http://", "https://")) and project is None:
        return Client(endpoint=run).request("simulation.stop")
    path, record = load_run_record(run, project)
    if record.get("state") == "stopped":
        return {"state": "stopped", "id": record["id"], "simulation": record.get("simulation")}
    owner = _OwnerClient(
        record["socket"], session=record["session"], generation=record["generation"]
    )
    record["simulation"] = owner.request("simulation.stop", timeout=240)
    _save(path, record)
    owner.request("shutdown", timeout=15)
    deadline = time.monotonic() + 15
    while Path(record["socket"]).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Shutdown requested but supervisor has not closed its socket")
        time.sleep(0.05)
    record["state"] = "stopped"
    _save(path, record)
    return {"state": "stopped", "id": record["id"], "simulation": record["simulation"]}


def connect(*args, **kwargs):
    from roboenv.devices.client import Environment

    return Environment(_connect(*args, **kwargs))


def status(run=None, *, endpoint=None, port=None, project=None):
    """Inspect one selected service without exposing its final judgment."""
    endpoint = _lifecycle_endpoint(run, endpoint, port)
    if endpoint is not None:
        # Validate an explicit URL before considering any ambient connection.
        client = Client(endpoint=endpoint)
    else:
        run = run or os.environ.get("ROBOENV_ENDPOINT")
        if run is not None and not str(run).startswith(("http://", "https://")):
            record = _record(run, project)
            if record.get("state") in {"stopped", "failed"}:
                return {
                    "state": record["state"],
                    "id": record["id"],
                    "simulation": status_feedback(record.get("simulation")),
                }
        client = _connect(run, project=project)
    return {
        **client.request("health", timeout=2),
        "simulation": client.request("simulation.status", timeout=30),
    }


class RunningEnvironment:
    """Owner handle. Context exit stops only the process launched by this handle."""

    def __init__(self, process, record, path):
        self.process, self.record, self.path = process, record, path
        self.client = _client(record)
        self._log_stop = threading.Event()
        self._log_thread = None

    def stream_logs(self, output):
        from contextlib import ExitStack

        paths = [self.path.parent / "supervisor.log", self.path.parent / "evidence/logs/worker.log"]

        def follow():
            with ExitStack() as stack:
                files = {}
                while True:
                    finished = self._log_stop.wait(0.1) or self.process.poll() is not None
                    for path in paths:
                        if path not in files and path.exists():
                            files[path] = stack.enter_context(path.open(errors="replace"))
                        if path in files:
                            text = files[path].read()
                            if text:
                                output.write(text)
                                output.flush()
                    if finished:
                        break

        print(
            f"Simulator logs: {self.path.parent / 'evidence/logs/worker.log'}",
            file=output,
            flush=True,
        )
        self._log_thread = threading.Thread(target=follow, daemon=True)
        self._log_thread.start()

    def stop_logs(self):
        self._log_stop.set()
        if self._log_thread is not None:
            self._log_thread.join()

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
            try:
                result = stop(self.path)
                self.record["simulation"] = result.get("simulation")
            except (OSError, RuntimeError):
                self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Supervisor did not stop; inspect the preserved run logs"
                ) from exc
        self.record["state"] = "stopped"
        _save(self.path, self.record)
        self.stop_logs()


def launch(
    task,
    *,
    project=None,
    catalog=None,
    interpreter=None,
    timeout=900,
    host="127.0.0.1",
    port=0,
    arm_gravity_compensation=None,
    log_stream=None,
    runtime_root=None,
    run_dir=None,
    allow_stop=False,
    advertise_host=None,
    python_path=(),
    execution_mode=None,
):
    """Start a task in its isolated interpreter; return after device readiness.

    arm_gravity_compensation overrides the SIMPLE task setting; None preserves
    the task value, which defaults to True. Other backends reject this override.
    execution_mode overrides the task setting; None preserves the catalog mode.
    Backend assets must already be prepared. No installation or download occurs.
    All run definitions and logs are retained under the selected project's cache.
    """
    if execution_mode is not None and execution_mode not in ("stepped", "realtime"):
        raise ValueError("execution_mode must be stepped or realtime")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    import ipaddress

    ipaddress.IPv4Address(host)
    if runtime_root is not None and (catalog is None or interpreter is None):
        raise ValueError("Packaged startup requires --catalog and --interpreter")
    if type(allow_stop) is not bool:
        raise ValueError("allow_stop must be a boolean")
    if isinstance(python_path, (str, bytes)):
        raise ValueError("python_path must be a sequence of deployment paths")
    python_path = [str(Path(path).resolve(strict=True)) for path in python_path]
    root = Path(runtime_root).resolve(strict=True) if runtime_root else project_root(project)
    resolved = resolve_task(
        task, project=root, catalog=catalog, arm_gravity_compensation=arm_gravity_compensation
    )
    if execution_mode is not None:
        resolved = replace(resolved, task=replace(resolved.task, execution_mode=execution_mode))
    spec = resolved.task
    execution_mode = spec.execution_mode
    if interpreter is None and spec.backend == "fakeshell":
        interpreter = sys.executable
    interpreter = resolve_interpreter(root, spec.backend, interpreter)
    if any(path.is_symlink() for path in resolved.robot.rglob("*")):
        raise ValueError("Robot definition must not contain symlinks")
    if spec.backend != "fakeshell" and runtime_root is None:
        from roboenv.provisioning.sources import status as source_status

        if not source_status(root, [spec.backend])["ready"]:
            raise ValueError(f"Sources are not ready; run 'roboenv doctor {spec.backend}'")
    run_id = uuid.uuid4().hex[:16]
    runs = (
        Path(run_dir).resolve()
        if run_dir
        else (root / ".cache/roboenv/runs" if runtime_root else _records(root))
    )
    directory = runs / run_id
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
        yaml.safe_dump(asdict(spec), sort_keys=False)
    )
    manifest = directory / "environment.json"
    _save(
        manifest,
        {
            "task": asdict(spec),
            "observation_access": "sensors-only",
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
        PYTHONUNBUFFERED="1",
        MUJOCO_GL="egl",
        ROBOENV_ALLOW_STOP="1" if allow_stop else "0",
    )
    if python_path:
        env["PYTHONPATH"] = os.pathsep.join(python_path)
    command = [
        str(interpreter),
        "-m",
        "roboenv_runtime.runtime.supervisor",
        "--task",
        spec.id,
        "--seed",
        str(spec.seed),
        "--worker-arg=--execution-mode",
        f"--worker-arg={execution_mode}",
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
    if log_stream is not None:
        running.stream_logs(log_stream)
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
                    if advertise_host:
                        from urllib.parse import urlsplit

                        record["endpoint"] = (
                            f"http://{advertise_host}:{urlsplit(health['endpoint']).port}"
                        )
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
                if health.get("worker", {}).get("state") in {
                    "failed",
                    "exited",
                    "terminated",
                    "timed_out",
                    "identity_mismatch",
                }:
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
