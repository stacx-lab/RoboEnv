"""The simulator worker: one adapter, one process, a small NDJSON socket API.

The worker owns the only simulator object.  It never accepts attribute paths
or arbitrary callables from the wire: ``call`` dispatches only the adapter's
allow-listed tools, scoring is owned by a :class:`roboenv_runtime.runtime.scoring.
ScoreAuthority`. Programs and callbacks execute only in the agent container;
neither official nor practice workers execute submitted Python.

Operations (:data:`roboenv_runtime.runtime.protocol.ALLOWED_OPS`):

======================  =====================================================
op                      what
======================  =====================================================
``health``              liveness and counters
``api``                 tools (signature/doc/read_only), capabilities, prompt context
``observe``             normalised ground truth + budget + latched/submitted
``call``                ``{"name": tool, ...kwargs}`` → tool result
``score.peek``          read-only evaluation
``submit``              the one final evaluation; mutating calls freeze after it
``reset``               same task, optional ``seed`` (``reset_same_task`` alias)
``checkpoint`` / ``restore``  when the adapter has ``Capability.CHECKPOINT``
``vla``                 bounded pretrained policy control (``Capability.VLA``)
``capture``             ``{"path": "captures/x.png"}`` → PNG inside the run dir
``exec``                always refused; programs execute in the agent container
``skill``               refused: primitives run client-side so each call is journaled
``shutdown``            stop serving
======================  =====================================================
"""

from __future__ import annotations

from roboenv.tasks.model import DEVICE_BACKENDS

import argparse
import contextlib
import copy
import logging
import os
import signal
import socketserver
import stat
import struct
import sys
import threading
import queue
from concurrent.futures import Future
import time
import zlib
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from roboenv_runtime.core.access import is_read_only
from roboenv_runtime.core.adapter import EnvAdapter, make_adapter
from roboenv_runtime.core.tools import ToolSpec
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
from roboenv_runtime.runtime.scoring import ScoreAuthority, make_score_authority
from roboenv_runtime.runtime.world import WorldSession

__all__ = ["SimWorker", "WorkerRPCError", "WorkerUnixServer", "handle_mapping", "main"]


log = logging.getLogger(__name__)

WORKER_OPS = ALLOWED_OPS - {"operations.get"}
READ_ONLY_PREFIXES = ("get_", "list_", "check_", "is_", "observe", "render", "peek")
"""Tools whose names start with these never move the world (allowed after submit)."""
SCORING_TOOL_MARKERS = ("score", "reward", "completed")
"""Tools that touch scoring are hidden from unprivileged clients."""
EXCLUDED_TOOLS = frozenset({"a", "sim", "exec", "checkpoint", "restore", "vla_run"})
"""Never exposed through ``call`` (checkpoint/restore/vla have their own ops)."""


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


def _parameter_names(fn: Any) -> list[str]:
    """Positional-capable parameter names of ``fn`` (for remote positional calls)."""
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [
        p.name
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


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
        privileged: bool = False,
        reset_adapter: bool = True,
        scorer_options: Mapping[str, Any] | None = None,
        device_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be 'official' or 'practice'")
        if privileged and budget_mode == "official":
            raise ValueError("privileged workers are practice-only")
        self.adapter = adapter
        adapter.run_dir = Path(run_dir).resolve()
        self.world = WorldSession(adapter)
        self.task_id = str(task_id)
        self.seed = int(seed)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.budget_mode = budget_mode
        self.session = session
        self.generation = int(generation)
        self.privileged = bool(privileged)
        self.scorer_options = dict(scorer_options or {})
        self.started_monotonic = time.monotonic()
        self.shutdown_requested = False
        self.state = "starting"
        self.request_count = 0
        self.basic_call_count = 0
        self.reset_count = 0
        self._vla_manager: Any = None
        self._policy_session = None
        self._checkpoints = {}
        self._closed = False
        self._final_result: dict[str, Any] | None = None
        self._final_observation: dict[str, Any] | None = None
        if reset_adapter:
            self.world.reset(self.seed)
        self.scorer: ScoreAuthority = make_score_authority(
            self.adapter, self.run_dir, budget_mode=self.budget_mode, **self.scorer_options
        )
        self._tools = self._collect_tools()
        self.devices = None
        self._device_root = device_root or os.environ.get("ROBOENV_SERVER_ROBOT")
        try:
            self._bind_devices()
            self.state = "ready"
            self.world.start()
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
            self._device_root, self.world, self.dispatch,
            manifest_path=os.environ.get("ROBOENV_SERVER_MANIFEST"),
            public_tools=self._tools,
        )
        self.devices.sequence = sequence

    # ------------------------------------------------------------ tools

    def _collect_tools(self) -> dict[str, Any]:
        registry = self.adapter.tools()
        if not isinstance(registry, Mapping):
            raise TypeError("adapter.tools() must return a mapping")
        tools: dict[str, Any] = {}
        for raw_name, fn in registry.items():
            name = str(raw_name)
            if not callable(fn) or name in EXCLUDED_TOOLS:
                continue
            if not self.privileged and any(m in name.lower() for m in SCORING_TOOL_MARKERS):
                continue
            tools[name] = self._counted_controller(name, fn)
        return tools

    def _counted_controller(self, name, fn):
        """Both device bindings and primitive RPC use this exact accounting point."""
        @wraps(fn)
        def invoke(*args, **kwargs):
            if self.scorer.submitted and not self.is_read_only(name):
                raise WorkerRPCError("already_submitted", "mutating calls are frozen after submit")
            self.basic_call_count += 1
            return fn(*args, **kwargs)

        return invoke

    @staticmethod
    def is_read_only(name: str) -> bool:
        """Heuristic: observation tools never mutate the world."""
        return is_read_only(name)

    def _tool(self, name: str) -> Any:
        fn = self._tools.get(name)
        if fn is None:
            raise WorkerRPCError(
                "unknown_tool",
                f"tool {name!r} is not available",
                details={"available": sorted(self._tools)},
            )
        return fn

    def _invoke_tool(self, name: str, kwargs: Mapping[str, Any]) -> Any:
        if self.scorer.submitted and not self.is_read_only(name):
            raise WorkerRPCError("already_submitted", "mutating calls are frozen after submit")
        if not isinstance(kwargs, Mapping):
            raise WorkerRPCError("invalid_args", "tool kwargs must be an object")
        return self._tool(name)(**dict(kwargs))

    @staticmethod
    def _named_kwargs(args: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        raw = dict(args)
        name = raw.pop("name", raw.pop("fn", None))
        if not isinstance(name, str) or not name:
            raise WorkerRPCError("invalid_args", "call requires args.name (or args.fn)")
        nested = raw.pop("kwargs", None)
        if nested is None:
            return name, raw
        if not isinstance(nested, Mapping):
            raise WorkerRPCError("invalid_args", "args.kwargs must be an object")
        if raw:
            raise WorkerRPCError(
                "invalid_args", "put tool arguments either in kwargs or alongside name"
            )
        return name, dict(nested)

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
            "privileged": self.privileged,
            "uptime_s": round(time.monotonic() - self.started_monotonic, 3),
            "requests": self.request_count,
            "basic_calls": self.basic_call_count,
            "resets": self.reset_count,
            "submitted": self.scorer.submitted,
            "success_latched": self.scorer.latched,
        }

    def api(self) -> dict[str, Any]:
        """Tool documentation plus the prompt context clients need."""
        tools = {}
        for name, fn in sorted(self._tools.items()):
            spec = ToolSpec.of(name, fn)
            tools[name] = {
                "signature": spec.signature,
                "parameters": _parameter_names(fn),
                "doc": spec.doc,
                "read_only": self.is_read_only(name),
            }
        capabilities = sorted(getattr(self.adapter, "capabilities", ()) or ())
        if self.budget_mode == "official":
            capabilities = [value for value in capabilities if value not in {"checkpoint", "privileged"}]
        vla_tool = None
        vla_fn = self.adapter.tools().get("vla_run")
        if callable(vla_fn):  # excluded from ``call``; served through the ``vla`` op instead
            spec = ToolSpec.of("vla_run", vla_fn)
            vla_tool = {
                "signature": spec.signature,
                "parameters": _parameter_names(vla_fn),
                "doc": spec.doc,
            }
        return {
            "protocol": PROTOCOL_VERSION,
            "device_contract": "roboenv/devices/1" if self.devices else None,
            "operations": sorted(WORKER_OPS - {"skill", "exec"} - ({"reset", "reset_same_task", "checkpoint", "restore"} if self.budget_mode == "official" else set())),
            "tools": tools,
            "context": {
                "adapter": getattr(self.adapter, "name", ""),
                "embodiment": getattr(self.adapter, "embodiment", "general"),
                "capabilities": capabilities,
                "situation": self.adapter.situation(),
                "facts": self.adapter.facts(),
                "goal": self.adapter.goal(),
                "task": self.task_id,
                "seed": self.seed,
                "budget_mode": self.budget_mode,
                "vla_tool": vla_tool,
                "vla_callback_protocol": "roboenv/vla-callback/1" if vla_tool and "until" in vla_tool["parameters"] else None,
            },
            "notes": {
                "skill": (
                    "primitives run client-side so every basic call is journaled; "
                    "the worker refuses the skill op"
                ),
                "score": "score.peek is read-only; submit performs the sole final evaluation",
                "exec": "programs execute in the agent container; server-side execution is disabled",
            },
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
                "budget": self.adapter.budget().as_dict(),
                "success_latched": self.scorer.latched,
                "submitted": self.scorer.submitted,
            }
        )
        from roboenv_runtime.vla.server import inventory
        observation["policy_servers"] = inventory()
        if self._vla_manager is not None and hasattr(self._vla_manager, "inventory"):
            observation["policy_servers"] += self._vla_manager.inventory()
        return observation

    def submit(self) -> dict[str, Any]:
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
            self.world.freeze()
            if self.scorer.submitted:
                raise WorkerRPCError("submit_failed", "the final evaluation did not complete")
            try:
                if self._policy_session is not None:
                    self._policy_session.close()
                    self._policy_session = None
                observation = self.observe()
                verdict = copy.deepcopy(self.scorer.submit())
                budget = self.adapter.budget().as_dict()
                if "budget" in verdict and any(
                    verdict["budget"].get(field) != budget.get(field)
                    for field in ("used", "max", "unit", "official_cap")
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
            except Exception:
                # An incomplete final evaluation is not a retryable opportunity
                # to take more actions or evaluate a different terminal state.
                self.state = "failed"
                raise
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
        if self._vla_manager is not None:
            with contextlib.suppress(Exception):
                self._vla_manager.close()
            self._vla_manager = None
        self.scorer.close()
        self.state = "resetting"
        try:
            if self.devices is not None:
                self.devices.close()  # release old entity handles before replacing the world
            self.world.reset(seed)
            self.seed = seed
            self.scorer = make_score_authority(
                self.adapter, self.run_dir, budget_mode=self.budget_mode, **self.scorer_options
            )
            self._final_result = None
            self._final_observation = None
            self._tools = self._collect_tools()
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
        fn = getattr(self.adapter, "checkpoint", None) or self.adapter.tools().get("checkpoint")
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
        fn = getattr(self.adapter, "restore", None) or self.adapter.tools().get("restore")
        if not callable(fn):
            raise WorkerRPCError("unsupported_op", "this adapter has no checkpoint capability")
        if "token" not in args:
            raise WorkerRPCError("invalid_args", "restore requires args.token")
        token = args["token"]
        if token not in self._checkpoints:
            raise WorkerRPCError("checkpoint_identity", "unknown checkpoint in this worker")
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

    def run_vla(self, args: Mapping[str, Any]) -> Any:
        """Bounded pretrained-policy control through the adapter's VLA manager."""
        if self.scorer.submitted:
            raise WorkerRPCError("already_submitted", "VLA is frozen after submit")
        if "_interactive" in args:
            from roboenv_runtime.runtime.policy_session import PolicySession

            action = args["_interactive"]
            if action == "start":
                if self._policy_session is not None:
                    raise WorkerRPCError("policy_active", "finish the current policy rollout first")
                tool = self.adapter.tools().get("vla_run")
                if not callable(tool):
                    raise WorkerRPCError("unsupported_op", "no VLA tool")
                params = dict(args.get("arguments", {}))
                if "until" in params:
                    raise WorkerRPCError("invalid_args", "callbacks must remain agent-side")
                factory = getattr(tool, "_interactive_factory", None)
                if factory is not None:
                    from roboenv_runtime.runtime.generator_session import GeneratorPolicySession
                    self._policy_session = GeneratorPolicySession(factory(**params))
                else:
                    self._policy_session = PolicySession(lambda until: tool(**params, until=until))
                state = self._policy_session.start()
                if state["status"] == "done":
                    self._policy_session = None
                return state
            session = self._policy_session
            if session is None or args.get("token") != session.token:
                raise WorkerRPCError("policy_identity", "unknown policy session")
            if action == "stop":
                session.close()
                self._policy_session = None
                return {"status": "done", "report": session.report, "cancelled": True}
            if action != "advance" or type(args.get("stop")) is not bool:
                raise WorkerRPCError("invalid_args", "invalid policy callback decision")
            state = session.advance(args["sequence"], args["stop"])
            if state["status"] == "done":
                self._policy_session = None
            return state
        if self._policy_session is not None:
            raise WorkerRPCError("policy_active", "a continuous policy rollout is already active")
        factory = getattr(self.adapter, "vla_manager", None)
        if callable(factory):
            if self._vla_manager is None:
                self._vla_manager = factory(self.scorer, self.run_dir)
            try:
                return self._vla_manager.dispatch(dict(args))
            except ValueError as exc:
                raise WorkerRPCError("invalid_args", str(exc)) from exc
        # adapters without a dedicated manager expose the shared ``vla_run`` tool
        # (roboenv_runtime.vla): same report shape, ``actions_taken`` included for finalize
        tool = self.adapter.tools().get("vla_run")
        if not callable(tool):
            raise WorkerRPCError("unsupported_op", "this adapter has no VLA capability")
        kwargs = dict(args)
        kwargs.pop("func_name", None)
        try:
            return tool(**kwargs)
        except (TypeError, ValueError) as exc:
            raise WorkerRPCError("invalid_args", str(exc)) from exc

    def dispatch(self, op: str, args: Mapping[str, Any]) -> Any:
        """All public operations and native devices share the same world's lane."""
        with self.world.lock:
            if self.world.failure is not None:
                self.state = "failed"
            return self._dispatch(op, args)

    def _dispatch(self, op: str, args: Mapping[str, Any]) -> Any:
        if self.state == "failed" and op not in {
            "health", "api", "environment.describe", "devices.list", "devices.describe",
            "reset", "reset_same_task", "shutdown",
        }:
            raise WorkerRPCError("world_failed", "World lifecycle failed; owner recovery is required")
        if self.adapter.name in DEVICE_BACKENDS and op in {
            "call", "skill", "exec", "vla", "reset", "reset_same_task", "checkpoint", "restore",
        }:
            raise WorkerRPCError("unsupported_op", "This backend accepts basic device operations; start a new service for a new episode")
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
        if op == "api":
            return self.api()
        if op == "observe":
            return self.observe()
        if op in {"environment.describe", "devices.list", "devices.describe", "devices.call"}:
            if self.devices is None:
                raise WorkerRPCError("devices_unconfigured", "No server-owned robot catalog configured")
            from roboenv.devices.model import DeviceError

            try:
                if op in {"environment.describe", "devices.list"}:
                    if args:
                        raise DeviceError("invalid_args", "Discovery takes no arguments")
                    if op == "devices.list":
                        return self.devices.inventory()
                    description = self.devices.environment()
                    task = description.get("task", {})
                    if self.adapter.name in DEVICE_BACKENDS:
                        description = {
                            "task": {"id": task.get("id", self.task_id),
                                     "instruction": self.adapter.goal() or task.get("instruction", ""),
                                     "context": {}},
                            "devices": description["devices"],
                        }
                        return {**description, "session": self.session, "generation": self.generation}
                    return {**description, "session": self.session,
                            "generation": self.generation, "task_id": self.task_id,
                            "seed": self.seed, "budget_mode": self.budget_mode}
                if op == "devices.describe":
                    if set(args) != {"device"}:
                        raise DeviceError("invalid_args", "Expected device name only")
                    return self.devices.describe(args["device"])
                result = self.devices.call(args, submitted=self.scorer.submitted)
                result["budget"] = self._budget_snapshot()
                return result
            except DeviceError as exc:
                raise WorkerRPCError(exc.code, exc.message, details=exc.details) from exc
        if op == "call":
            name, kwargs = self._named_kwargs(args)
            return self._invoke_tool(name, kwargs)
        if op == "skill":
            raise WorkerRPCError(
                "client_side_skill", "skills execute client-side as basic-call compositions"
            )
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
        if op in ("reset", "reset_same_task"):
            return self.reset(args)
        if op == "checkpoint":
            return self.checkpoint()
        if op == "restore":
            return self.restore(args)
        if op == "vla":
            return self.run_vla(args)
        if op == "exec":
            raise WorkerRPCError("client_side_exec", "programs execute in the agent container, never inside the simulator")
        if op == "shutdown":
            if args:
                raise WorkerRPCError("invalid_args", "shutdown takes no arguments")
            self.shutdown_requested = True
            self.state = "shutting_down"
            return {"shutdown": True}
        raise AssertionError(op)

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release the VLA manager, the scorer and the adapter (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.world.stop_clock()
        if self._policy_session is not None:
            self._policy_session.close()
            self._policy_session = None
        self.state = "closed"
        if self._vla_manager is not None:
            with contextlib.suppress(Exception):
                self._vla_manager.close()
            self._vla_manager = None
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


def handle_mapping(worker: SimWorker, raw: Mapping[str, Any]) -> dict[str, Any]:
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
        try:
            return Response.success(
                request, jsonable(worker.dispatch(request.op, request.args))
            ).as_dict()
        except WorkerRPCError as exc:
            return Response.failure(request, exc.code, exc.message, details=exc.details).as_dict()
        except Exception as exc:
            return Response.failure(
                request, "operation_failed", f"{type(exc).__name__}: {exc}"
            ).as_dict()

    request_id = raw.get("id", raw.get("request_id"))
    op = raw.get("op")
    args = raw.get("args", {})
    if not isinstance(op, str) or not isinstance(args, Mapping):
        return {
            "id": request_id,
            "ok": False,
            "error": {"code": "invalid_request", "message": "op and object args are required"},
        }
    try:
        return {"id": request_id, "ok": True, "result": jsonable(worker.dispatch(op, args))}
    except WorkerRPCError as exc:
        error: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.details is not None:
            error["details"] = jsonable(exc.details)
        return {"id": request_id, "ok": False, "error": error}
    except Exception as exc:
        return {
            "id": request_id,
            "ok": False,
            "error": {"code": "operation_failed", "message": f"{type(exc).__name__}: {exc}"},
        }


class WorkerUnixServer(socketserver.UnixStreamServer):
    """Serial Unix-socket server around one :class:`SimWorker`."""

    allow_reuse_address = False

    def __init__(self, path: str, worker: SimWorker, max_bytes: int = MAX_NDJSON_BYTES) -> None:
        self.worker = worker
        self.max_bytes = max_bytes
        super().__init__(socket_address(path), WorkerRequestHandler)

    def dispatch(self, raw):
        return handle_mapping(self.worker, raw)


class MainThreadWorkerUnixServer(socketserver.ThreadingMixIn, WorkerUnixServer):
    """Decode connections in threads; run every native call on the owner thread.

    Isaac cannot advance physics on the background clock thread. Idle or partial
    connections must also leave its main execution lane available for clock ticks.
    """

    daemon_threads = True
    block_on_close = False
    request_handoff_timeout = 0.02

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
            target=super().serve_forever, kwargs={"poll_interval": poll_interval},
            name="roboenv-worker-accept", daemon=True,
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
                timeout = min(self.request_handoff_timeout, self.worker.world.poll_interval)
                future, raw = self._pending.get(timeout=timeout)
            except queue.Empty:
                break
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(handle_mapping(self.worker, raw))
                except Exception as exc:
                    future.set_exception(exc)
        self.worker.world.poll()

    def server_close(self):
        with self._pending_lock:
            self._closing = True
            while not self._pending.empty():
                future, _raw = self._pending.get_nowait()
                future.cancel()
        super().server_close()


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


def build_adapter(
    task_ref: str, budget_mode: str, *, practice_max_steps: int = 240_000
) -> tuple[EnvAdapter, str]:
    """Instantiate the adapter of ``task_ref`` for ``budget_mode``.

    Returns ``(adapter, task_id)``.
    """
    from roboenv_runtime.tasks.loader import find_task

    task = find_task(task_ref)
    kwargs = dict(task.adapter_kwargs)
    if task.adapter == "robodojo":
        # Typed RoboDojo always clamps action frames to the native task horizon.
        # The retired primitive wrapper's raw-step override is not its contract.
        kwargs.setdefault("max_steps", practice_max_steps)
    adapter = make_adapter(task.adapter, **kwargs)
    return adapter, task.id


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-mode", choices=("official", "practice"), default="practice")
    parser.add_argument(
        "--session", default=os.environ.get("ROBOENV_SESSION") or os.environ.get("DOJO_SESSION")
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=int(os.environ.get("ROBOENV_GENERATION") or os.environ.get("DOJO_GENERATION") or 0),
    )
    parser.add_argument("--privileged", action="store_true", help="expose environment diagnostics (practice only); never server-side code")
    parser.add_argument("--practice-max-steps", type=int, default=240_000)
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

    adapter, task_id = build_adapter(
        args.task, args.budget_mode, practice_max_steps=args.practice_max_steps
    )
    worker: SimWorker | None = None
    server: WorkerUnixServer | None = None
    try:
        worker = SimWorker(
            adapter,
            task_id=task_id,
            seed=args.seed,
            run_dir=run_dir,
            budget_mode=args.budget_mode,
            session=args.session,
            generation=args.generation,
            privileged=args.privileged,
        )
        server_type = MainThreadWorkerUnixServer if worker.world.main_thread_clock else WorkerUnixServer
        server = server_type(str(socket_path), worker, args.max_ndjson_bytes)
        os.chmod(socket_path, 0o600)
        log.info(
            "worker ready: task=%s seed=%s mode=%s socket=%s",
            task_id,
            args.seed,
            args.budget_mode,
            socket_path,
        )

        def request_shutdown(_signum: int, _frame: Any) -> None:
            assert worker is not None and server is not None
            worker.shutdown_requested = True
            threading.Thread(target=server.shutdown, daemon=True).start()

        old_term = signal.signal(signal.SIGTERM, request_shutdown)
        old_int = signal.signal(signal.SIGINT, request_shutdown)
        try:
            server.serve_forever(poll_interval=worker.world.poll_interval)
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
