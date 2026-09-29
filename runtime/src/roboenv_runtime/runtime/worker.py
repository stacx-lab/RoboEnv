"""One native world, typed device dispatch, and private owner operations."""

from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import os
import queue
import signal
import socketserver
import stat
import struct
import sys
import threading
import time
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
from roboenv.devices.execution import map_result
from roboenv.devices.status import status_feedback

from roboenv_runtime.core.adapter import EnvAdapter, make_adapter
from roboenv_runtime.runtime.protocol import (
    ALLOWED_OPS,
    MAX_NDJSON_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    Request,
    Response,
    encode_ndjson,
    read_ndjson,
    socket_address,
)
from roboenv_runtime.runtime.scoring import SimpleScorer
from roboenv_runtime.runtime.world import WorldSession

__all__ = ["SimWorker", "WorkerRPCError", "WorkerUnixServer", "handle_mapping", "main"]


log = logging.getLogger(__name__)

WORKER_OPS = ALLOWED_OPS - {"operations.get"}


class WorkerRPCError(RuntimeError):
    """A stable, client-facing worker error."""

    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def jsonable(value: Any) -> Any:
    """Recursively convert numpy and other exotic values to JSON-compatible ones."""
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    as_dict = getattr(value, "as_dict", None) or getattr(value, "to_dict", None)
    if callable(as_dict):
        return jsonable(as_dict())
    return repr(value)


def _inside(root: Path, candidate: str | os.PathLike[str], *, allow_root: bool = True) -> Path:
    root = root.resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    if path == root and not allow_root:
        raise WorkerRPCError("invalid_path", f"path must be below {root}, not the root itself")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkerRPCError("invalid_path", f"path {path} escapes {root}") from exc
    return path


# ---------------------------------------------------------------- PNG (no PIL dependency)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    chunk = kind + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def write_png(path: Path, frame: Any) -> tuple[int, int, int]:
    """Write an ``HxWx{1,3,4}`` uint8 frame as PNG; returns ``(height, width, channels)``."""
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise WorkerRPCError("capture_unavailable", f"unsupported frame shape {array.shape}")
    height, width, channels = array.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return height, width, channels


# ---------------------------------------------------------------- the worker


class SimWorker:
    """Own one adapter, one score authority, and a bounded request dispatcher."""

    def __init__(
        self,
        adapter: EnvAdapter,
        *,
        task_id: str,
        seed: int,
        run_dir: str | os.PathLike[str],
        budget_mode: str = "practice",
        session: str | None = None,
        generation: int = 0,
        reset_adapter: bool = True,
        device_root: str | os.PathLike[str] | None = None,
        execution_mode: str = "realtime",
    ) -> None:
        if budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be 'official' or 'practice'")
        self.adapter = adapter
        adapter.run_dir = Path(run_dir).resolve()
        self.world = WorldSession(adapter, execution_mode)
        self.task_id = str(task_id)
        self.seed = int(seed)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.budget_mode = budget_mode
        self.session = session
        self.generation = int(generation)
        self.started_monotonic = time.monotonic()
        self.shutdown_requested = False
        self.state = "starting"
        self.request_count = 0
        self.reset_count = 0
        self._checkpoints = {}
        self._closed = False
        self._final_result: dict[str, Any] | None = None
        self._final_observation: dict[str, Any] | None = None
        self._termination_reason = None
        if reset_adapter:
            self.world.reset(self.seed)
        self.scorer = SimpleScorer(self.adapter)
        self.devices = None
        self._device_root = device_root or os.environ.get("ROBOENV_SERVER_ROBOT")
        try:
            self._bind_devices()
            self.state = "ready"
            self.world.after_interval = self._check_terminal
        except BaseException:
            self.close()
            raise

    def _bind_devices(self):
        if not self._device_root:
            return
        from roboenv_runtime.bindings.assembly import assemble_devices

        sequence = self.devices.sequence if self.devices is not None else 0
        if self.devices is not None:
            self.devices.close()
        self.devices = assemble_devices(
            self._device_root,
            self.world,
            manifest_path=os.environ.get("ROBOENV_SERVER_MANIFEST"),
        )
        self.devices.sequence = sequence

    # ------------------------------------------------------------ operations

    def health(self) -> dict[str, Any]:
        """Liveness and counters."""
        return {
            "worker": self.state,
            "pid": os.getpid(),
            "adapter": getattr(self.adapter, "name", "?"),
            "task": self.task_id,
            "seed": self.seed,
            "budget_mode": self.budget_mode,
            "execution_mode": self.world.mode,
            "uptime_s": round(time.monotonic() - self.started_monotonic, 3),
            "requests": self.request_count,
            "resets": self.reset_count,
            "submitted": self.scorer.submitted,
            "success_latched": self.scorer.latched,
        }

    def observe(self) -> dict[str, Any]:
        """The adapter's observation plus worker bookkeeping."""
        if self._final_observation is not None:
            return copy.deepcopy(self._final_observation)
        observation = dict(self.adapter.observe())
        observation.update(
            {
                "task": self.task_id,
                "seed": self.seed,
                "budget_mode": self.budget_mode,
                "execution_mode": self.world.mode,
                "budget": self.adapter.budget().as_dict(),
                "success_latched": self.scorer.latched,
                "submitted": self.scorer.submitted,
            }
        )
        return observation

    def _check_terminal(self):
        """Called in the world's lane after each native control interval and each operation."""
        if self._final_result is not None or not getattr(self.adapter, "bounded_episodes", False):
            return
        success = getattr(self.adapter, "stop_on_success", False) and self.adapter.score().completed
        if success or self.adapter.budget().exhausted:
            self._termination_reason = "success" if success else "step_limit"
            self.submit(interrupt=False)

    def simulation_status(self, *, include_verdict=False):
        """Expose the judgment only for stop receipts and retained owner evidence."""
        result = self._final_result
        status = {
            "schema": "roboenv/simulation-status/1",
            "session": self.session,
            "generation": self.generation,
            "state": "terminal" if result is not None else self.state,
            "terminal": result is not None,
            "success": bool(result["completed"]) if result is not None else None,
            "reason": self._termination_reason,
            "budget": self._budget_snapshot(),
            "simulation_time_s": getattr(self.adapter, "simulation_time_s", None),
            "capabilities": {
                "terminal_feedback": True,
                "bounded_episodes": bool(getattr(self.adapter, "bounded_episodes", False)),
            },
        }
        return status if include_verdict else status_feedback(status)

    def _interrupt_devices(self):
        if self.devices is not None:
            self.devices.interrupt()
        self.world.poll_motions()

    def submit(self, *, interrupt=True) -> dict[str, Any]:
        """Seal one authoritative observation, budget and verdict in the world's lane.

        Some adapters charge their internal observation API queries. Capture
        those BEFORE scoring and retain every charge; never refund counters or
        replace a live budget with an earlier, more favorable value. Once sealed,
        verifier reads consume the snapshot rather than querying the simulator.
        """
        with self.world.lock:
            if self._final_result is not None:
                return copy.deepcopy(self._final_result)
            if self.state == "failed" or self.world.failure is not None:
                raise WorkerRPCError("world_failed", "owner recovery is required")
            if self.devices is not None:
                # Natural terminal frames complete current segments normally.
                # Explicit stop interrupts them; neither admits queued goals.
                self.devices.interrupt(active=interrupt)
            self.world.freeze()
            if hasattr(self.adapter, "freeze_episode"):
                self.adapter.freeze_episode()
            if self.scorer.submitted:
                raise WorkerRPCError("submit_failed", "the final evaluation did not complete")
            try:
                observation = self.observe()
                verdict = copy.deepcopy(self.scorer.submit())
                budget = self.adapter.budget().as_dict()
                if "budget" in verdict and any(
                    verdict["budget"].get(field) != budget.get(field)
                    for field in ("used", "max", "unit")
                ):
                    raise WorkerRPCError(
                        "budget_mismatch", "score authority and final native budget disagree"
                    )
                verdict["budget"] = copy.deepcopy(budget)
                observation.update(
                    budget=copy.deepcopy(budget),
                    submitted=True,
                    success_latched=self.scorer.latched,
                )
                self._final_observation = copy.deepcopy(observation)
                self._final_result = copy.deepcopy(verdict)
                self._termination_reason = self._termination_reason or (
                    "success" if verdict["completed"] else "stopped"
                )
                from roboenv_runtime.runtime.atomic import atomic_write_json

                atomic_write_json(
                    self.run_dir / "terminal.json",
                    {
                        "status": self.simulation_status(include_verdict=True),
                        "verdict": verdict,
                        "task": self.task_id,
                        "seed": self.seed,
                    },
                )
            except Exception:
                # An incomplete final evaluation is not a retryable opportunity
                # to take more actions or evaluate a different terminal state.
                self.state = "failed"
                raise
            # Publish the sealed verdict before completing terminal segments.
            self.world.poll_motions()
            return copy.deepcopy(self._final_result)

    def _budget_snapshot(self) -> dict[str, Any]:
        if self._final_result is not None:
            return copy.deepcopy(self._final_result["budget"])
        return self.adapter.budget().as_dict()

    def capture(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Write the latest camera frame as PNG inside ``run_dir/captures``."""
        raw_path = args.get("path", args.get("name"))
        if not isinstance(raw_path, str) or not raw_path:
            raise WorkerRPCError("invalid_args", "capture requires args.path")
        logical = Path(raw_path)
        if logical.is_absolute() or ".." in logical.parts:
            raise WorkerRPCError(
                "invalid_path", "capture path must be relative and may not contain '..'"
            )
        if logical.suffix.lower() != ".png":
            raise WorkerRPCError("invalid_path", "capture path must end in .png")
        path = _inside(self.run_dir / "captures", logical, allow_root=False)
        frame = self.adapter.capture()
        if frame is None:
            raise WorkerRPCError("capture_unavailable", "no simulator frame is available")
        height, width, channels = write_png(path, frame)
        return {"path": str(path), "height": height, "width": width, "channels": channels}

    def reset(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Reset the same task (optionally a new seed) and rebuild the scorer."""
        unknown = set(args) - {"seed"}
        if unknown:
            raise WorkerRPCError("invalid_args", f"reset does not accept: {sorted(unknown)}")
        seed = int(args.get("seed", self.seed))
        self._interrupt_devices()
        self.scorer.close()
        self.state = "resetting"
        try:
            if self.devices is not None:
                self.devices.close()  # release old entity handles before replacing the world
            self.world.reset(seed)
            self.seed = seed
            self.scorer = SimpleScorer(self.adapter)
            self._termination_reason = None
            self._final_result = None
            self._final_observation = None
            self._bind_devices()
            self.reset_count += 1
            self.state = "ready"
        except Exception:
            self.state = "failed"
            raise
        return {
            "task": self.task_id,
            "seed": self.seed,
            "reset_count": self.reset_count,
            "score": self.scorer.peek(),
            "budget": self.adapter.budget().as_dict(),
        }

    def checkpoint(self) -> Any:
        """Snapshot the world when the adapter supports it."""
        fn = getattr(self.adapter, "checkpoint", None)
        if not callable(fn):
            raise WorkerRPCError("unsupported_op", "this adapter has no checkpoint capability")
        import uuid

        token = uuid.uuid4().hex
        self._checkpoints[token] = fn()
        return token

    def restore(self, args: Mapping[str, Any]) -> Any:
        """Rewind the world to a snapshot."""
        if self.scorer.submitted:
            raise WorkerRPCError("already_submitted", "restore is frozen after submit")
        fn = getattr(self.adapter, "restore", None)
        if not callable(fn):
            raise WorkerRPCError("unsupported_op", "this adapter has no checkpoint capability")
        if "token" not in args:
            raise WorkerRPCError("invalid_args", "restore requires args.token")
        token = args["token"]
        if token not in self._checkpoints:
            raise WorkerRPCError("checkpoint_identity", "unknown checkpoint in this worker")
        self._interrupt_devices()
        self.state = "restoring"
        try:
            if self.devices is not None:
                self.devices.close()
            fn(self._checkpoints[token])
            self.world.epoch += 1
            self._bind_devices()
            self.state = "ready"
        except Exception:
            self.state = "failed"
            raise
        return {"restored": True}

    def dispatch(self, op: str, args: Mapping[str, Any]) -> Any:
        return self.world.resolve(self.dispatch_async(op, args))

    def dispatch_async(self, op: str, args: Mapping[str, Any]) -> Any:
        """Native sections are atomic; bounded segment completion yields the lane."""
        with self.world.lock:
            if self.world.failure is not None:
                self.state = "failed"
            try:
                return self._dispatch(op, args)
            finally:
                self.world.poll_motions()

    def _dispatch(self, op: str, args: Mapping[str, Any]) -> Any:
        if op == "simulation.status":
            if args:
                raise WorkerRPCError("invalid_args", "simulation.status takes no arguments")
            return self.simulation_status()
        if op == "simulation.stop":
            if args:
                raise WorkerRPCError("invalid_args", "simulation.stop takes no arguments")
            self.submit()
            return self.simulation_status(include_verdict=True)
        if self.state == "failed" and op not in {
            "health",
            "environment.describe",
            "devices.list",
            "devices.describe",
            "reset",
            "shutdown",
        }:
            raise WorkerRPCError(
                "world_failed", "World lifecycle failed; owner recovery is required"
            )
        if op not in WORKER_OPS:
            raise WorkerRPCError(
                "unsupported_op",
                f"operation {op!r} is not allowed",
                details={"allowed": sorted(WORKER_OPS)},
            )
        if not isinstance(args, Mapping):
            raise WorkerRPCError("invalid_args", "operation args must be an object")
        self.request_count += 1
        if op == "health":
            return self.health()
        if op == "observe":
            return self.observe()
        if op in {
            "environment.describe",
            "devices.list",
            "devices.describe",
            "devices.call",
            "devices.cancel",
        }:
            if self.devices is None:
                raise WorkerRPCError(
                    "devices_unconfigured", "No server-owned robot catalog configured"
                )
            from roboenv.devices.model import DeviceError

            try:
                if op in {"environment.describe", "devices.list"}:
                    if args:
                        raise DeviceError("invalid_args", "Discovery takes no arguments")
                    if op == "devices.list":
                        return self.devices.inventory()
                    description = self.devices.environment()
                    task = description.get("task", {})
                    return {
                        "task": {
                            "id": task.get("id", self.task_id),
                            "instruction": self.adapter.goal() or task.get("instruction", ""),
                            "context": {},
                        },
                        "devices": description["devices"],
                        "device_contract": "roboenv/devices/1",
                        "session": self.session,
                        "generation": self.generation,
                    }
                if op == "devices.describe":
                    if set(args) != {"device"}:
                        raise DeviceError("invalid_args", "Expected device name only")
                    return self.devices.describe(args["device"])
                if self._final_result is not None and getattr(
                    self.adapter, "bounded_episodes", False
                ):
                    from roboenv.devices.contract import READ_ACTIONS

                    if args.get("action") not in READ_ACTIONS:
                        # Refused before dispatch: the sealed episode is unchanged, and the
                        # verifier can certify a journal that holds this refusal.
                        raise WorkerRPCError(
                            "episode_finished",
                            "Episode finished; call roboenv stop for the final judgment",
                            details={**self.simulation_status(), "effects": "none"},
                        )
                operation = {
                    "devices.cancel": self.devices.cancel,
                }.get(op, self.devices.submit)
                result = operation(args, submitted=self.scorer.submitted)

                def finish(value):
                    self._check_terminal()
                    value["budget"] = self._budget_snapshot()
                    if self._final_result is not None:
                        value["simulation"] = self.simulation_status()
                    return value

                def failed(error):
                    if isinstance(error, DeviceError):
                        raise WorkerRPCError(
                            error.code, error.message, details=error.details
                        ) from error
                    raise error

                return map_result(result, finish, failed)
            except DeviceError as exc:
                raise WorkerRPCError(exc.code, exc.message, details=exc.details) from exc
        if op == "score.peek":
            if self._final_result is not None:
                return copy.deepcopy(self._final_result)
            return self.scorer.peek()
        if op == "capture":
            return self.capture(args)
        if op == "submit":
            if args:
                raise WorkerRPCError("invalid_args", "submit takes no arguments")
            return self.submit()
        if op == "reset":
            return self.reset(args)
        if op == "checkpoint":
            return self.checkpoint()
        if op == "restore":
            return self.restore(args)
        if op == "shutdown":
            if args:
                raise WorkerRPCError("invalid_args", "shutdown takes no arguments")
            self._interrupt_devices()
            self.shutdown_requested = True
            self.state = "shutting_down"
            return {"shutdown": True}
        raise AssertionError(op)

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release the scorer and the adapter (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._interrupt_devices()
        self.world.freeze()
        self.world.poll_motions()
        self.state = "closed"
        try:
            if self.devices is not None:
                with contextlib.suppress(Exception):
                    self.devices.close()
            self.scorer.close()
        finally:
            with contextlib.suppress(Exception):
                self.world.close()


# ---------------------------------------------------------------- wire handling


def _strict_error(raw: Any, code: str, message: str, *, details: Any = None) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    session = raw.get("session") if isinstance(raw.get("session"), str) else "invalid"
    generation = raw.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        generation = 0
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        request_id = "invalid"
    error: dict[str, Any] = {"code": code, "message": message, "retryable": False}
    if details is not None:
        error["details"] = jsonable(details)
    return {
        "protocol": PROTOCOL_VERSION,
        "session": session,
        "generation": generation,
        "request_id": request_id,
        "ok": False,
        "error": error,
    }


def handle_mapping(worker: SimWorker, raw: Mapping[str, Any], *, deferred=False):
    """Dispatch a full protocol envelope, or a minimal ``{id, op, args}`` test envelope."""
    strict = {"protocol", "session", "generation", "request_id", "deadline_s", "op", "args"} <= set(
        raw
    )
    if strict:
        try:
            request = Request.from_mapping(raw, allowed_ops=WORKER_OPS)
        except ProtocolError as exc:
            return _strict_error(raw, exc.code, exc.message, details=exc.details)
        if worker.session is not None and request.session != worker.session:
            return Response.failure(
                request, "session_mismatch", f"worker session is {worker.session!r}"
            ).as_dict()
        if request.generation != worker.generation:
            return Response.failure(
                request, "generation_mismatch", f"worker generation is {worker.generation}"
            ).as_dict()

        def failed(exc):
            if isinstance(exc, WorkerRPCError):
                return Response.failure(
                    request, exc.code, exc.message, details=exc.details
                ).as_dict()
            return Response.failure(
                request, "operation_failed", f"{type(exc).__name__}: {exc}"
            ).as_dict()

        try:
            dispatch = worker.dispatch_async if deferred else worker.dispatch
            from roboenv.devices.execution import request_identity

            token = request_identity.set(request.request_id)
            try:
                outcome = dispatch(request.op, request.args)
            finally:
                request_identity.reset(token)
            return map_result(
                outcome,
                lambda value: Response.success(request, jsonable(value)).as_dict(),
                failed,
            )
        except Exception as exc:
            return failed(exc)

    request_id = raw.get("id", raw.get("request_id"))
    op = raw.get("op")
    args = raw.get("args", {})
    if not isinstance(op, str) or not isinstance(args, Mapping):
        return {
            "id": request_id,
            "ok": False,
            "error": {"code": "invalid_request", "message": "op and object args are required"},
        }

    def failed(exc):
        if isinstance(exc, WorkerRPCError):
            error = {"code": exc.code, "message": exc.message}
            if exc.details is not None:
                error["details"] = jsonable(exc.details)
        else:
            error = {"code": "operation_failed", "message": f"{type(exc).__name__}: {exc}"}
        return {"id": request_id, "ok": False, "error": error}

    try:
        dispatch = worker.dispatch_async if deferred else worker.dispatch
        return map_result(
            dispatch(op, args),
            lambda value: {"id": request_id, "ok": True, "result": jsonable(value)},
            failed,
        )
    except Exception as exc:
        return failed(exc)


class WorkerUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Concurrent connections; the world owns atomic native state access."""

    daemon_threads = True
    block_on_close = False

    allow_reuse_address = False

    def __init__(self, path: str, worker: SimWorker, max_bytes: int = MAX_NDJSON_BYTES) -> None:
        self.worker = worker
        self.max_bytes = max_bytes
        super().__init__(socket_address(path), WorkerRequestHandler)

    def dispatch(self, raw):
        return handle_mapping(self.worker, raw)


class MainThreadWorkerUnixServer(WorkerUnixServer):
    """Decode connections in threads; run every native call on the owner thread.

    Native physics and graphics contexts can require their creating thread, in
    all backends. Idle connections and motion waits leave that thread free
    to service reads and pending commands.
    """

    daemon_threads = True
    block_on_close = False
    request_handoff_timeout = 0.002

    def __init__(self, *args, **kwargs):
        self._pending = queue.SimpleQueue()
        self._pending_lock = threading.Lock()
        self._closing = False
        self._stop_serving = threading.Event()
        self._native_stopped = threading.Event()
        super().__init__(*args, **kwargs)

    def serve_forever(self, poll_interval=0.2):
        # WorkerClient opens a new socket for every RPC. Accepting only from
        # the native loop would throttle all devices to one request per frame,
        # even when a batch could otherwise drain the pending request queue.
        acceptor = threading.Thread(
            target=super().serve_forever,
            kwargs={"poll_interval": poll_interval},
            name="roboenv-worker-accept",
            daemon=True,
        )
        self._native_stopped.clear()
        acceptor.start()
        try:
            while not self._stop_serving.is_set():
                self.service_native_requests()
        finally:
            super().shutdown()
            acceptor.join()
            self._native_stopped.set()

    def shutdown(self):
        self._stop_serving.set()
        self._native_stopped.wait()

    def dispatch(self, raw):
        future = Future()
        with self._pending_lock:
            if self._closing:
                raise WorkerRPCError("worker_closed", "worker is closing")
            self._pending.put((future, raw))
        return future.result()

    def service_native_requests(self):
        # A slow native tick must not limit all devices together to one RPC per
        # physics frame. Give decoding/reply threads a bounded opportunity to
        # hand over the next request, then service the clock even under a flood.
        for _ in range(16):
            try:
                timeout = self.request_handoff_timeout
                world = self.worker.world
                if world.mode == "realtime":
                    timeout = min(timeout, max(0, world.next_tick - time.monotonic()))
                future, raw = self._pending.get(timeout=timeout)
            except queue.Empty:
                break
            if future.set_running_or_notify_cancel():
                try:
                    result = handle_mapping(self.worker, raw, deferred=True)
                    if isinstance(result, Future):

                        def completed(source, target=future):
                            try:
                                target.set_result(source.result())
                            except Exception as error:
                                target.set_exception(error)

                        result.add_done_callback(completed)
                    else:
                        future.set_result(result)
                except Exception as exc:
                    future.set_exception(exc)
        self.worker.world.advance()
        self.worker.world.poll_motions()

    def server_close(self):
        with self._pending_lock:
            self._closing = True
            while not self._pending.empty():
                future, _raw = self._pending.get_nowait()
                future.cancel()
        super().server_close()


def create_worker_server(path, worker, max_bytes=MAX_NDJSON_BYTES):
    return MainThreadWorkerUnixServer(path, worker, max_bytes)


class WorkerRequestHandler(socketserver.StreamRequestHandler):
    """One NDJSON request per line; connections may carry several."""

    def handle(self) -> None:
        server: WorkerUnixServer = self.server  # type: ignore[assignment]
        while True:
            try:
                raw = read_ndjson(self.rfile, max_bytes=server.max_bytes)
            except EOFError:
                return
            except ProtocolError as exc:
                response = _strict_error({}, exc.code, exc.message, details=exc.details)
            else:
                response = server.dispatch(raw)
            try:
                self.wfile.write(encode_ndjson(response, max_bytes=server.max_bytes))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if server.worker.shutdown_requested:
                threading.Thread(target=server.shutdown, daemon=True).start()
                return


# ---------------------------------------------------------------- command line


def build_adapter(task):
    """Instantiate the selected native backend without implicit budget overrides."""
    return make_adapter(task.backend, **task.backend_kwargs)


def _remove_worker_socket(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket path: {path}")
    path.unlink()


def build_parser() -> argparse.ArgumentParser:
    """CLI of ``python -m roboenv_runtime.runtime.worker``."""
    parser = argparse.ArgumentParser(description="roboenv simulator worker")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, help="task id or yaml path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget-mode", choices=("official", "practice"))
    parser.add_argument("--session", default=os.environ.get("ROBOENV_SESSION"))
    parser.add_argument(
        "--generation",
        type=int,
        default=int(os.environ.get("ROBOENV_GENERATION") or 0),
    )
    parser.add_argument("--execution-mode", choices=("stepped", "realtime"))
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=None,
        help="verify dependency hashes before booting",
    )
    parser.add_argument("--max-ndjson-bytes", type=int, default=MAX_NDJSON_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Boot the adapter, serve until shutdown, exit cleanly."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not args.session:
        raise SystemExit("--session or $ROBOENV_SESSION is required")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    socket_path = args.socket.resolve()
    try:
        socket_address(socket_path)  # a long path is served through a short alias
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_worker_socket(socket_path)
    if args.source_lock is not None:
        from roboenv_runtime.paths import repo_root
        from roboenv_runtime.runtime.source_lock import verify

        verify(args.source_lock, repo_root())

    from roboenv.tasks.model import Task, task_path

    from roboenv_runtime.paths import tasks_root

    task = Task.load(task_path(args.task, tasks_root()))
    execution_mode = args.execution_mode or task.execution_mode
    adapter = build_adapter(task)
    worker: SimWorker | None = None
    server: WorkerUnixServer | None = None
    try:
        worker = SimWorker(
            adapter,
            task_id=task.id,
            seed=task.seed if args.seed is None else args.seed,
            run_dir=run_dir,
            budget_mode=args.budget_mode or task.budget_mode,
            execution_mode=execution_mode,
            session=args.session,
            generation=args.generation,
        )
        server = create_worker_server(str(socket_path), worker, args.max_ndjson_bytes)
        os.chmod(socket_path, 0o600)
        log.info(
            "worker ready: task=%s seed=%s mode=%s socket=%s",
            task.id,
            worker.seed,
            worker.budget_mode,
            socket_path,
        )

        def request_shutdown(_signum: int, _frame: Any) -> None:
            assert worker is not None and server is not None
            worker.shutdown_requested = True
            threading.Thread(target=server.shutdown, daemon=True).start()

        old_term = signal.signal(signal.SIGTERM, request_shutdown)
        old_int = signal.signal(signal.SIGINT, request_shutdown)
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0
    except BaseException:
        # Simulator cleanup may terminate the process before Python prints the exception.
        log.exception("Simulator worker failed")
        raise
    finally:
        if server is not None:
            server.server_close()
        if worker is not None:
            worker.close()
        else:
            with contextlib.suppress(Exception):
                adapter.close()
        with contextlib.suppress(FileNotFoundError):
            if socket_path.is_socket():
                socket_path.unlink()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":  # pragma: no cover
    code = main()
    os._exit(code)  # simulators leave non-daemon threads behind
