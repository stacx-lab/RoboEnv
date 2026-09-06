"""Stateful simulator-process supervisor and serial Unix-socket proxy.

The supervisor owns exactly one ``sim_worker`` process group.  Public health
checks are answered from supervisor state and never depend on the worker.
Every other operation is serialized, journaled, and proxied to the worker's
private Unix socket.  A request deadline kills only the process group created
by this supervisor; no process discovery or broad cleanup is performed.
"""

# Adapted from codex_dojo_alltasks/harness/supervisor.py on 2026-08-26 (refactor P3):
# worker module, socket root and environment variables generalised; otherwise verbatim.

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from roboenv_runtime.runtime.atomic import AtomicJournal, atomic_write_json
from roboenv_runtime.runtime.protocol import (
    ALLOWED_OPS,
    MAX_NDJSON_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    Request,
    RequestCache,
    RequestIdConflict,
    Response,
    encode_ndjson,
    read_ndjson,
    socket_address,
)


def _inside(root: Path, candidate: str | os.PathLike[str]) -> Path:
    root = root.resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {path} escapes run directory {root}") from exc
    return path


def _remove_stale_socket(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket path: {path}")
    path.unlink()


@dataclass
class SupervisorConfig:
    run_dir: Path
    task: str
    seed: int
    budget_mode: str
    session: str
    generation: int = 0
    public_socket: Path | None = None
    worker_socket: Path | None = None
    worker_module: str = "roboenv_runtime.runtime.worker"
    worker_extra_args: tuple[str, ...] = ()
    python_executable: str = sys.executable
    cache_entries: int = 2048
    worker_startup_timeout_s: float = 900.0
    terminate_grace_s: float = 10.0
    max_ndjson_bytes: int = MAX_NDJSON_BYTES

    def __post_init__(self) -> None:
        self.run_dir = self.run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # sockets must live below the run directory (or $ROBOENV_SOCKET_ROOT: Unix socket
        # paths are limited to 107 bytes, so deep run directories may need a short root)
        socket_root = Path(os.environ.get("ROBOENV_SOCKET_ROOT", str(self.run_dir))).resolve(
            strict=True
        )
        self.public_socket = _inside(
            socket_root, self.public_socket or (self.run_dir / "supervisor.sock")
        )
        self.worker_socket = _inside(
            socket_root, self.worker_socket or (self.run_dir / "worker.sock")
        )
        for socket_path in (self.public_socket, self.worker_socket):
            socket_address(socket_path)  # raises when even the short alias is too long
        if self.public_socket == self.worker_socket:
            raise ValueError("public and worker sockets must be distinct")
        if self.budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be 'official' or 'practice'")
        # Reuse protocol validation for externally visible identities.
        Request.from_mapping(
            {
                "protocol": PROTOCOL_VERSION,
                "session": self.session,
                "generation": self.generation,
                "request_id": "config-check",
                "deadline_s": 1,
                "op": "health",
                "args": {},
            }
        )


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = True


class Supervisor:
    """Own a single simulator worker and expose the public protocol socket."""

    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.journal = AtomicJournal(config.run_dir / "journal" / "actions.jsonl")
        self.cache = RequestCache(config.cache_entries)
        self._cache_lock = threading.Lock()
        self._identity_lock = threading.RLock()
        self._known_ids: set[str] = set()
        self._inflight: dict[str, Request] = {}
        self._worker_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._worker: subprocess.Popen[bytes] | None = None
        self._worker_log = None
        self._worker_pgid: int | None = None
        self._worker_state = "not_started"
        self._device_registry_ready = False
        self._worker_failure: str | None = None
        self._worker_started_monotonic: float | None = None
        self._started_monotonic = time.monotonic()
        self._started_wall = time.time()
        self._server: _ThreadingUnixServer | None = None
        self._closing = False
        self._restore_cache()

    # -------------------------------------------------------------- durability

    @property
    def state_path(self) -> Path:
        return self.config.run_dir / "state" / "supervisor.json"

    def _state_payload(self) -> dict[str, Any]:
        worker = self._worker
        return {
            "protocol": PROTOCOL_VERSION,
            "session": self.config.session,
            "generation": self.config.generation,
            "task": self.config.task,
            "seed": self.config.seed,
            "budget_mode": self.config.budget_mode,
            "run_dir": str(self.config.run_dir),
            "public_socket": str(self.config.public_socket),
            "worker_socket": str(self.config.worker_socket),
            "supervisor_pid": os.getpid(),
            "worker_pid": worker.pid if worker is not None else None,
            "worker_pgid": self._worker_pgid,
            "worker_state": self._worker_state,
            "worker_returncode": worker.poll() if worker is not None else None,
            "worker_failure": self._worker_failure,
            "started_unix": self._started_wall,
            "updated_unix": time.time(),
        }

    def _write_state(self) -> None:
        with self._state_lock:
            atomic_write_json(self.state_path, self._state_payload())

    def _restore_cache(self) -> None:
        """Recover terminal responses and fence crash-interrupted request ids."""

        pending: dict[tuple[str, int, str], Request] = {}
        for event in self.journal.events():
            raw_request = event.get("request")
            if not isinstance(raw_request, dict):
                continue
            try:
                request = Request.from_mapping(raw_request)
            except ProtocolError:
                continue
            if (
                request.session != self.config.session
                or request.generation != self.config.generation
            ):
                continue
            key = (request.session, request.generation, request.request_id)
            self._known_ids.add(request.request_id)
            if event.get("phase") == "intent":
                pending[key] = request
                continue
            if event.get("phase") not in {"commit", "error"}:
                continue
            response = event.get("response")
            if isinstance(response, dict):
                with contextlib.suppress(ProtocolError):
                    self.cache.store(request, response)
            pending.pop(key, None)

        # Re-executing a request that reached the worker but did not receive a
        # durable terminal event can duplicate physical actions.  Fence it and
        # require a new, explicit request id after human inspection.
        for request in pending.values():
            response = Response.failure(
                request,
                "indeterminate_previous_attempt",
                "a durable intent exists without commit/error; automatic replay is unsafe",
                details={"journal": str(self.journal.path)},
                retryable=False,
            ).as_dict()
            self.cache.store(request, response)

    # --------------------------------------------------------------- lifecycle

    def _worker_command(self) -> list[str]:
        return [
            self.config.python_executable,
            "-m",
            self.config.worker_module,
            "--socket",
            str(self.config.worker_socket),
            "--run-dir",
            str(self.config.run_dir),
            "--task",
            self.config.task,
            "--seed",
            str(self.config.seed),
            "--budget-mode",
            self.config.budget_mode,
            *self.config.worker_extra_args,
        ]

    def _worker_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        root = self.config.run_dir
        dirs = {
            "HOME": root / "home",
            "TMPDIR": root / "tmp",
            "XDG_CACHE_HOME": root / "cache" / "xdg",
            "XDG_CONFIG_HOME": root / "config" / "xdg",
            "XDG_DATA_HOME": root / "data" / "xdg",
            "OMNI_USER_CONFIG_PATH": root / "config" / "omniverse",
            "OMNI_LOG_ROOT": root / "logs" / "omniverse",
            "OMNI_CACHE_ROOT": root / "cache" / "omniverse",
            "OV_CACHE_ROOT": root / "cache" / "ov",
            "OV_DATA_ROOT": root / "data" / "ov",
            "OV_LOG_ROOT": root / "logs" / "ov",
        }
        for key, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(
            {
                "ROBOENV_RUN_DIR": str(root),
                "ROBOENV_WORKER_SOCKET": str(self.config.worker_socket),
                "ROBOENV_SESSION": self.config.session,
                "ROBOENV_GENERATION": str(self.config.generation),
                # historical names, still read by moved RoboDojo helpers
                "DOJO_RUN_DIR": str(root),
                "DOJO_SESSION": self.config.session,
                "DOJO_GENERATION": str(self.config.generation),
                "ROBODOJO_RUN_ID": (f"{self.config.session}-g{self.config.generation}"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    def start_worker(self) -> None:
        if self._worker is not None:
            raise RuntimeError("worker already started")
        _remove_stale_socket(self.config.worker_socket)
        log_path = self.config.run_dir / "logs" / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_log = log_path.open("ab", buffering=0)
        cmd = self._worker_command()
        self._worker_state = "starting"
        self._worker = subprocess.Popen(
            cmd,
            cwd=self.config.run_dir,
            env=self._worker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=self._worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._worker_started_monotonic = time.monotonic()
        self._worker_pgid = os.getpgid(self._worker.pid)
        if self._worker_pgid != self._worker.pid:
            # This should be impossible with start_new_session=True.  Refuse
            # to retain a process whose group identity is not exact.
            self._worker.terminate()
            self._worker.wait(timeout=5)
            raise RuntimeError("worker did not become leader of its own process group")
        self._write_state()

    def _worker_alive(self) -> bool:
        return self._worker is not None and self._worker.poll() is None

    def _refresh_worker_state(self) -> None:
        if self._worker is None:
            return
        previous_state = self._worker_state
        rc = self._worker.poll()
        if rc is not None and self._worker_state not in {
            "stopped",
            "timed_out",
            "terminated",
        }:
            self._worker_state = "exited"
            self._worker_failure = f"worker exited with return code {rc}"
        elif rc is None and self.config.worker_socket.exists():
            self._worker_state = "ready"
        if self._worker_state != previous_state:
            self._write_state()

    def terminate_worker(self, *, reason: str, timed_out: bool = False) -> None:
        """Terminate only the exact process group created by this object."""

        worker = self._worker
        pgid = self._worker_pgid
        if worker is None or pgid is None:
            return
        if worker.poll() is None:
            try:
                actual_pgid = os.getpgid(worker.pid)
            except ProcessLookupError:
                actual_pgid = None
            if actual_pgid != pgid or pgid != worker.pid:
                self._worker_failure = (
                    "refused kill: recorded worker process-group identity changed"
                )
                self._worker_state = "identity_mismatch"
                self._write_state()
                return
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
            try:
                worker.wait(timeout=self.config.terminate_grace_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    worker.wait(timeout=self.config.terminate_grace_s)
        if timed_out:
            self._worker_state = "timed_out"
            self._worker_failure = reason
        elif self._worker_state != "timed_out":
            self._worker_state = "terminated"
            self._worker_failure = reason
        self._write_state()

    # --------------------------------------------------------------- protocol

    def health(self) -> dict[str, Any]:
        self._refresh_worker_state()
        worker = self._worker
        return {
            "supervisor": "ok" if not self._closing else "closing",
            "protocol": PROTOCOL_VERSION,
            "session": self.config.session,
            "generation": self.config.generation,
            "task": self.config.task,
            "seed": self.config.seed,
            "budget_mode": self.config.budget_mode,
            "uptime_s": round(time.monotonic() - self._started_monotonic, 3),
            "worker": {
                "state": self._worker_state,
                "alive": self._worker_alive(),
                "pid": worker.pid if worker is not None else None,
                "pgid": self._worker_pgid,
                "returncode": worker.poll() if worker is not None else None,
                "socket_ready": self.config.worker_socket.exists(),
                "failure": self._worker_failure,
            },
            "device_service": {
                "contract": "roboenv/devices/1",
                "ready": self._device_registry_ready and self._worker_state == "ready",
            },
        }

    @staticmethod
    def _match_response(request: Request, response: Response) -> None:
        if (
            response.session != request.session
            or response.generation != request.generation
            or response.request_id != request.request_id
        ):
            raise ProtocolError(
                "invalid_response",
                "worker response identity does not match request",
            )

    @staticmethod
    def _normalize_worker_response(request: Request, raw: Mapping[str, Any]) -> Response:
        # Preferred full campaign envelope.
        if {"protocol", "session", "generation", "request_id"} <= set(raw):
            response = Response.from_mapping(raw)
            Supervisor._match_response(request, response)
            return response

        # Transitional worker envelope: {id/request_id, ok, result|error}.
        worker_id = raw.get("request_id", raw.get("id"))
        if worker_id != request.request_id:
            raise ProtocolError("invalid_response", "worker response id does not match request")
        ok = raw.get("ok")
        if ok is True:
            return Response.success(request, raw.get("result"))
        if ok is not False:
            raise ProtocolError("invalid_response", "worker response ok must be boolean")
        error = raw.get("error")
        if not isinstance(error, dict):
            raise ProtocolError("invalid_response", "worker error must be an object")
        code = error.get("code", error.get("type", "worker_error"))
        message = error.get("message", str(error))
        details = {k: v for k, v in error.items() if k not in {"code", "type", "message"}}
        return Response.failure(
            request,
            str(code),
            str(message),
            details=details or None,
            retryable=bool(error.get("retryable", False)),
        )

    def _connect_worker(self, deadline: float) -> socket.socket:
        self._refresh_worker_state()
        worker_started = self._worker_started_monotonic or self._started_monotonic
        # The startup timeout only limits the initial boot. Once the worker
        # has exposed its socket, its lifetime is bounded by each request's
        # own deadline, not by worker_started + startup_timeout. Using the
        # startup deadline forever used to kill healthy long-lived Isaac
        # workers on the first RPC after 900 seconds.
        ready = self._worker_state == "ready" or self.config.worker_socket.exists()
        connect_deadline = (
            deadline
            if ready
            else min(deadline, worker_started + self.config.worker_startup_timeout_s)
        )
        last_error: OSError | None = None
        while time.monotonic() < connect_deadline:
            self._refresh_worker_state()
            if not self._worker_alive():
                raise RuntimeError(self._worker_failure or "worker is not running")
            remaining = connect_deadline - time.monotonic()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(max(0.001, remaining))
            try:
                sock.connect(socket_address(self.config.worker_socket))
                self._worker_state = "ready"
                return sock
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_error = exc
                sock.close()
                time.sleep(min(0.05, max(0.0, remaining)))
            except Exception:
                sock.close()
                raise
        raise TimeoutError(
            f"worker socket was not ready (state={self._worker_state}, last_error={last_error!r})"
        )

    def _proxy(self, request: Request, deadline: float) -> Response:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request deadline expired before worker dispatch")
        sock = self._connect_worker(deadline)
        try:
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.sendall(encode_ndjson(request.as_dict(), max_bytes=self.config.max_ndjson_bytes))
            stream = sock.makefile("rb")
            try:
                raw = read_ndjson(stream, max_bytes=self.config.max_ndjson_bytes)
            finally:
                stream.close()
            return self._normalize_worker_response(request, raw)
        finally:
            sock.close()

    def _cache_lookup(self, request: Request) -> dict[str, Any] | None:
        with self._cache_lock:
            lookup = self.cache.lookup(request)
            active = self._inflight.get(request.request_id)
            known = request.request_id in self._known_ids
        if lookup.hit:
            return lookup.response
        if active is not None:
            if active.semantic_fingerprint() != request.semantic_fingerprint():
                raise RequestIdConflict(request.request_id)
            return None
        # Cache eviction is not permission to execute an old motion again.
        if known:
            previous, response = self._history(request.request_id)
            if previous is not None:
                if previous.semantic_fingerprint() != request.semantic_fingerprint():
                    raise RequestIdConflict(request.request_id)
                return response or Response.failure(
                    request, "indeterminate_previous_attempt",
                    "A durable intent has no terminal result; automatic replay is unsafe",
                ).as_dict()
        return None

    def _history(self, request_id):
        previous, response = None, None
        for event in self.journal.events():
            raw = event.get("request", {})
            if (raw.get("session"), raw.get("generation"), raw.get("request_id")) != (
                self.config.session, self.config.generation, request_id
            ):
                continue
            previous = Request.from_mapping(raw)
            if event.get("phase") in {"commit", "error"}:
                response = event.get("response")
        return previous, response

    def operation(self, args):
        if set(args) != {"request_id"} or not isinstance(args["request_id"], str):
            raise ProtocolError("invalid_args", "operations.get requires a request_id string")
        target = args["request_id"]
        with self._cache_lock:
            active = self._inflight.get(target)
        if active is not None:
            return {"request_id": target, "state": "running", "op": active.op}
        request, response = self._history(target)
        if response is not None:
            code = (response.get("error") or {}).get("code")
            state = "succeeded" if response.get("ok") else "failed"
            effects = (response.get("error") or {}).get("details", {}).get("effects")
            if effects == "unknown" or code in {
                "worker_timeout", "worker_unavailable", "indeterminate_previous_attempt"
            }:
                state = "indeterminate"
            return {"request_id": target, "state": state, "response": response}
        return {"request_id": target, "state": "indeterminate" if request else "unknown"}

    def _cache_store(self, request: Request, response: Mapping[str, Any]) -> None:
        with self._identity_lock, self._cache_lock:
            self.cache.store(request, response)
            self._known_ids.add(request.request_id)
            self._inflight.pop(request.request_id, None)

    def _queue_timeout_response(
        self, request: Request, *, received_monotonic: float
    ) -> dict[str, Any]:
        with self._identity_lock:
            return self._queue_timeout_locked(request, received_monotonic=received_monotonic)

    def _queue_timeout_locked(
        self, request: Request, *, received_monotonic: float
    ) -> dict[str, Any]:
        try:
            cached = self._cache_lookup(request)
        except RequestIdConflict as exc:
            return Response.failure(request, exc.code, exc.message).as_dict()
        if cached is not None:
            return cached
        with self._cache_lock:
            active = self._inflight.get(request.request_id)
        if active is not None:
            # A retry waiting behind its original must not poison the original's
            # deduplication entry or append a second intent for the same action.
            return Response.failure(
                request, "request_in_progress", "Original request is still running",
                retryable=True,
            ).as_dict()
        raw_request = request.as_dict()
        elapsed = round(max(0.0, time.monotonic() - received_monotonic), 6)
        response = Response.failure(
            request,
            "supervisor_queue_timeout",
            "deadline expired while waiting for the serialized worker proxy",
            details={"worker_killed": False},
            retryable=True,
        ).as_dict()
        self.journal.intent(raw_request, supervisor_pid=os.getpid())
        self.journal.error(raw_request, response, elapsed_s=elapsed)
        self._cache_store(request, response)
        return response

    def handle(
        self, request: Request, *, received_monotonic: float | None = None
    ) -> dict[str, Any]:
        received = time.monotonic() if received_monotonic is None else received_monotonic
        request_deadline = received + request.deadline_s
        if request.session != self.config.session:
            return Response.failure(
                request,
                "session_mismatch",
                f"supervisor session is {self.config.session!r}",
            ).as_dict()
        if request.generation != self.config.generation:
            return Response.failure(
                request,
                "generation_mismatch",
                f"supervisor generation is {self.config.generation}",
            ).as_dict()

        try:
            cached = self._cache_lookup(request)
        except RequestIdConflict as exc:
            return Response.failure(request, exc.code, exc.message).as_dict()
        if cached is not None:
            return cached

        if request.op == "health":
            response = Response.success(request, self.health()).as_dict()
            self._cache_store(request, response)
            return response
        if request.op == "operations.get":
            try:
                response = Response.success(request, self.operation(request.args)).as_dict()
            except ProtocolError as exc:
                response = Response.failure(request, exc.code, exc.message).as_dict()
            self._cache_store(request, response)
            return response

        remaining_for_queue = request_deadline - time.monotonic()
        acquired = remaining_for_queue > 0 and self._worker_lock.acquire(
            timeout=remaining_for_queue
        )
        if not acquired:
            return self._queue_timeout_response(request, received_monotonic=received)

        try:
            # Atomically reserve dispatch against a retry's queue-timeout result.
            with self._identity_lock:
                try:
                    cached = self._cache_lookup(request)
                except RequestIdConflict as exc:
                    return Response.failure(request, exc.code, exc.message).as_dict()
                if cached is not None:
                    return cached
                if time.monotonic() >= request_deadline:
                    return self._queue_timeout_response(request, received_monotonic=received)

                raw_request = request.as_dict()
                started = time.monotonic()
                with self._cache_lock:
                    self._inflight[request.request_id] = request
                    self._known_ids.add(request.request_id)
                self.journal.intent(raw_request, supervisor_pid=os.getpid())
            if time.monotonic() >= request_deadline:
                response = Response.failure(
                    request,
                    "request_deadline_expired",
                    "deadline expired before worker dispatch",
                    details={"worker_killed": False},
                    retryable=True,
                ).as_dict()
                self.journal.error(
                    raw_request,
                    response,
                    elapsed_s=round(time.monotonic() - started, 6),
                )
                self._cache_store(request, response)
                return response
            try:
                response_obj = self._proxy(request, request_deadline)
                response = response_obj.as_dict()
                if request.op == "devices.list" and response_obj.ok:
                    result = response_obj.result
                    self._device_registry_ready = (
                        isinstance(result, dict)
                        and result.get("schema_version") == "roboenv/devices/1"
                        and isinstance(result.get("devices"), list)
                        and bool(result["devices"])
                    )
            except (TimeoutError, socket.timeout) as exc:
                worker_was_alive = self._worker_alive()
                self.terminate_worker(reason=str(exc), timed_out=True)
                worker_killed = worker_was_alive and self._worker_state == "timed_out"
                response = Response.failure(
                    request,
                    "worker_timeout",
                    str(exc),
                    details={"worker_killed": worker_killed},
                    retryable=False,
                ).as_dict()
            except (EOFError, OSError, RuntimeError, ProtocolError) as exc:
                self._refresh_worker_state()
                code = exc.code if isinstance(exc, ProtocolError) else "worker_unavailable"
                response = Response.failure(
                    request,
                    code,
                    str(exc),
                    details={"worker_state": self._worker_state},
                    retryable=False,
                ).as_dict()
            elapsed = round(time.monotonic() - started, 6)
            if response["ok"]:
                self.journal.commit(raw_request, response, elapsed_s=elapsed)
            else:
                self.journal.error(raw_request, response, elapsed_s=elapsed)
            self._cache_store(request, response)

            if request.op == "shutdown":
                self.terminate_worker(reason="shutdown requested")
                self._schedule_server_shutdown()
            return response
        finally:
            self._worker_lock.release()

    # --------------------------------------------------------------- server I/O

    def _unparsed_error(self, raw: Any, exc: ProtocolError) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        request_id = raw.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = "invalid-" + uuid.uuid4().hex
        session = raw.get("session")
        if not isinstance(session, str) or not session:
            session = self.config.session
        generation = raw.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            generation = self.config.generation
        return {
            "protocol": PROTOCOL_VERSION,
            "session": session,
            "generation": generation,
            "request_id": request_id,
            "ok": False,
            "error": {**exc.as_error(), "retryable": False},
        }

    def _handler_type(self):
        supervisor = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    try:
                        raw = read_ndjson(self.rfile, max_bytes=supervisor.config.max_ndjson_bytes)
                    except EOFError:
                        return
                    except ProtocolError as exc:
                        response = supervisor._unparsed_error({}, exc)
                    else:
                        received_monotonic = time.monotonic()
                        try:
                            request = Request.from_mapping(raw, allowed_ops=ALLOWED_OPS)
                            response = supervisor.handle(
                                request, received_monotonic=received_monotonic
                            )
                        except ProtocolError as exc:
                            response = supervisor._unparsed_error(raw, exc)
                        except Exception as exc:  # keep the supervisor reachable
                            response = supervisor._unparsed_error(
                                raw,
                                ProtocolError(
                                    "internal_error",
                                    f"supervisor request handler failed: {type(exc).__name__}: {exc}",
                                ),
                            )
                    try:
                        self.wfile.write(
                            encode_ndjson(
                                response,
                                max_bytes=supervisor.config.max_ndjson_bytes,
                            )
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        return Handler

    def open_server(self) -> None:
        if self._server is not None:
            raise RuntimeError("server already open")
        _remove_stale_socket(self.config.public_socket)
        self._server = _ThreadingUnixServer(
            socket_address(self.config.public_socket), self._handler_type()
        )
        os.chmod(self.config.public_socket, 0o600)
        self._write_state()

    def _schedule_server_shutdown(self) -> None:
        server = self._server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()

    def serve_forever(self) -> None:
        if self._worker is None:
            self.start_worker()
        if self._server is None:
            self.open_server()
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.2)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        server, self._server = self._server, None
        if server is not None:
            server.server_close()
        self.terminate_worker(reason="supervisor closing")
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        with contextlib.suppress(FileNotFoundError):
            if self.config.public_socket.is_socket():
                self.config.public_socket.unlink()
        self._write_state()


def _session_default(task: str, seed: int) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in task)
    return f"{clean}-s{seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget-mode", required=True, choices=("official", "practice"))
    parser.add_argument("--session")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--socket", dest="public_socket", type=Path)
    parser.add_argument("--worker-socket", type=Path)
    parser.add_argument("--worker-module", default="roboenv_runtime.runtime.worker")
    parser.add_argument("--worker-arg", action="append", default=[])
    parser.add_argument("--worker-startup-timeout", type=float, default=900.0)
    parser.add_argument("--terminate-grace", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = (
        args.session
        or os.environ.get("ROBOENV_SESSION")
        or os.environ.get("DOJO_SESSION")
        or _session_default(args.task, args.seed)
    )
    config = SupervisorConfig(
        run_dir=args.run_dir,
        task=args.task,
        seed=args.seed,
        budget_mode=args.budget_mode,
        session=session,
        generation=args.generation,
        public_socket=args.public_socket,
        worker_socket=args.worker_socket,
        worker_module=args.worker_module,
        worker_extra_args=tuple(args.worker_arg),
        worker_startup_timeout_s=args.worker_startup_timeout,
        terminate_grace_s=args.terminate_grace,
    )
    supervisor = Supervisor(config)

    def request_shutdown(_signum, _frame) -> None:
        supervisor._schedule_server_shutdown()

    old_term = signal.signal(signal.SIGTERM, request_shutdown)
    old_int = signal.signal(signal.SIGINT, request_shutdown)
    try:
        supervisor.serve_forever()
        return 0
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
