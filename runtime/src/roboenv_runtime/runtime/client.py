"""Client side of the worker protocol.

A strict RPC client (:class:`WorkerClient`), the agent-facing :class:`RemoteAPI`,
and the ``roboenv robot`` command line for shell agents.

Environment variables (set by the supervisor for anything it launches, and
by ``roboenv run --remote`` for local clients): ``ROBOENV_SOCKET``,
``ROBOENV_RUN_DIR``, ``ROBOENV_SESSION``, ``ROBOENV_GENERATION``.  The historical
``DOJO_*`` names are honoured as fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

__all__ = ["ClientError", "RemoteAPI", "WorkerClient", "main"]

# Moved (rewritten) from codex_dojo_alltasks/harness/client.py on 2026-08-26 (refactor P3).


class ClientError(RuntimeError):
    """A remote operation failed; the message carries ``code: message``."""


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def logical_capture_path(path: str | os.PathLike[str]) -> str:
    """Normalise a capture path to the relative form the worker accepts."""
    text = str(path)
    run_dir = _env("ROBOENV_RUN_DIR", "DOJO_RUN_DIR")
    if run_dir and Path(text).is_absolute():
        try:
            return str(Path(text).resolve().relative_to(Path(run_dir).resolve()))
        except ValueError:
            pass
    return text


class WorkerClient:
    """One request per connection, newline-delimited JSON, identity-checked replies."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        session: str,
        generation: int = 0,
        run_dir: str | os.PathLike[str] | None = None,
        max_bytes: int = MAX_NDJSON_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.session = session
        self.generation = int(generation)
        self.run_dir = Path(run_dir) if run_dir is not None else self.socket_path.parent
        self.max_bytes = max_bytes

    def build_request(
        self,
        op: str,
        args: Mapping[str, Any] | None = None,
        *,
        deadline_s: float,
        request_id: str | None = None,
    ) -> Request:
        """A validated request envelope with a fresh (or explicit) request id."""
        if op not in ALLOWED_OPS:
            raise ValueError(
                f"operation {op!r} is not in the protocol; allowed: {sorted(ALLOWED_OPS)}"
            )
        return Request.from_mapping(
            {
                "protocol": PROTOCOL_VERSION,
                "session": self.session,
                "generation": self.generation,
                "request_id": request_id or f"{op.replace('.', '-')}-{uuid.uuid4().hex[:12]}",
                "deadline_s": float(deadline_s),
                "op": op,
                "args": dict(args or {}),
            }
        )

    def send(self, request: Request, *, connect_timeout_s: float = 30.0) -> Response:
        """Send one request and return the (identity-checked) response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(connect_timeout_s)
        try:
            sock.connect(socket_address(self.socket_path))
            sock.settimeout(request.deadline_s + 5.0)
            sock.sendall(encode_ndjson(request.as_dict(), max_bytes=self.max_bytes))
            stream = sock.makefile("rb")
            try:
                raw = read_ndjson(stream, max_bytes=self.max_bytes)
            finally:
                stream.close()
        finally:
            sock.close()
        response = Response.from_mapping(raw)
        if (
            response.session != request.session
            or response.generation != request.generation
            or response.request_id != request.request_id
        ):
            raise ProtocolError("invalid_response", "response identity does not match the request")
        return response

    def request(
        self,
        op: str,
        args: Mapping[str, Any] | None = None,
        *,
        deadline_s: float = 300.0,
        request_id: str | None = None,
    ) -> Response:
        """Build and send in one go."""
        return self.send(self.build_request(op, args, deadline_s=deadline_s, request_id=request_id))


class RemoteAPI:
    """Agent-facing fixed RPC surface; successful methods return result values.

    This is also the only interface a primitive pack may use (``call`` /
    ``observe`` / ``peek``), which keeps skills honest: no simulator handle,
    no adapter internals.
    """

    def __init__(
        self,
        socket_path: str | os.PathLike[str] | None = None,
        *,
        run_dir: str | os.PathLike[str] | None = None,
        session: str | None = None,
        generation: int | None = None,
        default_deadline_s: float = 300.0,
    ) -> None:
        raw_socket = socket_path or _env("ROBOENV_SOCKET", "DOJO_SOCKET")
        if raw_socket is None:
            raise ValueError("socket_path or $ROBOENV_SOCKET is required")
        raw_run_dir = run_dir or _env("ROBOENV_RUN_DIR", "DOJO_RUN_DIR") or Path(raw_socket).parent
        raw_session = session or _env("ROBOENV_SESSION", "DOJO_SESSION")
        if not raw_session:
            raise ValueError("session or $ROBOENV_SESSION is required")
        if generation is None:
            generation = int(_env("ROBOENV_GENERATION", "DOJO_GENERATION") or 0)
        self.default_deadline_s = float(default_deadline_s)
        self._client = WorkerClient(
            raw_socket, run_dir=raw_run_dir, session=raw_session, generation=generation
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> RemoteAPI:
        """Build from the ``ROBOENV_*`` environment."""
        return cls(**overrides)

    @property
    def client(self) -> WorkerClient:
        """The underlying transport client."""
        return self._client

    def _rpc(
        self, op: str, args: Mapping[str, Any] | None = None, *, deadline_s: float | None = None
    ) -> Any:
        response = self._client.request(
            op, args, deadline_s=self.default_deadline_s if deadline_s is None else deadline_s
        )
        if not response.ok:
            error = response.error or {}
            code = error.get("code", "remote_error")
            message = error.get("message", "remote operation failed")
            raise ClientError(f"{code}: {message}")
        return response.result

    # ------------------------------------------------------------ operations

    def health(self) -> Any:
        """Supervisor/worker liveness."""
        return self._rpc("health")

    def api(self) -> Any:
        """Tool documentation, capabilities and prompt context of the worker's adapter."""
        return self._rpc("api")

    def observe(self) -> Any:
        """Normalised ground-truth observation."""
        return self._rpc("observe")

    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke one allow-listed tool."""
        if not isinstance(name, str) or not name:
            raise ValueError("call name must be a non-empty string")
        return self._rpc("call", {"name": name, **kwargs})

    def skill(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Run a primitive from the installed packs client-side (each inner call is journaled)."""
        if not isinstance(name, str) or not name:
            raise ValueError("skill name must be a non-empty string")
        try:
            from roboenv_runtime.skills.packs import (
                get_primitives,  # available from refactor phase P4
            )
        except ImportError as exc:  # pragma: no cover - until P4 lands
            raise ClientError(f"primitive packs are not installed: {exc}") from exc
        primitive = get_primitives().get(name)
        if primitive is None:
            raise ValueError(f"unknown primitive {name!r}")
        result = primitive(self, **kwargs)
        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise ClientError(f"primitive {name!r} returned an invalid result")
        value = to_dict()
        if not isinstance(value, dict):
            raise ClientError(f"primitive {name!r} result is not a dictionary")
        return value

    def vla(self, **kwargs: Any) -> Any:
        """Bounded pretrained-policy control (adapters with ``Capability.VLA``)."""
        return self._rpc("vla", kwargs, deadline_s=max(self.default_deadline_s, 1800.0))

    def score_peek(self) -> Any:
        """Read-only evaluation."""
        return self._rpc("score.peek")

    def capture(self, path: str | os.PathLike[str]) -> Any:
        """Save a camera frame as PNG inside the run directory."""
        return self._rpc("capture", {"path": logical_capture_path(path)})

    def submit(self) -> Any:
        """The one final evaluation; freezes mutating calls afterwards."""
        return self._rpc("submit")

    def reset(self, **kwargs: Any) -> Any:
        """Reset the same task (optionally with a new ``seed``)."""
        return self._rpc("reset", kwargs, deadline_s=max(self.default_deadline_s, 900.0))

    def reset_same_task(self, **kwargs: Any) -> Any:
        """Legacy name of :meth:`reset`."""
        return self.reset(**kwargs)

    def checkpoint(self) -> Any:
        """Snapshot the world (``Capability.CHECKPOINT``)."""
        return self._rpc("checkpoint")

    def restore(self, token: Any) -> Any:
        """Rewind to a snapshot."""
        return self._rpc("restore", {"token": token})

    def exec(self, code: str, **kwargs: Any) -> Any:
        """Run a code turn on the worker (privileged practice workers only)."""
        return self._rpc(
            "exec", {"code": code, **kwargs}, deadline_s=max(self.default_deadline_s, 3600.0)
        )

    def shutdown(self) -> Any:
        """Stop the worker."""
        return self._rpc("shutdown")

    # ------------------------------------------------------------ metadata

    def peek(self, kind: str | None = None, **kwargs: Any) -> Any:
        """Read normalised public metadata (geometry, gripper calibration, home poses)."""
        observation = self.observe()
        if not isinstance(observation, Mapping):
            raise ClientError("observe did not return a metadata mapping")
        if kind is None:
            return {
                key: observation[key]
                for key in ("geometry", "gripper_calibration", "home_poses")
                if key in observation
            }
        if kind == "geometry":
            label = kwargs.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError("geometry peek requires label")
            geometry = observation.get("geometry")
            if isinstance(geometry, Mapping) and isinstance(geometry.get(label), Mapping):
                return dict(geometry[label])
            objects = observation.get("objects")
            if isinstance(objects, Mapping) and isinstance(objects.get(label), Mapping):
                state = objects[label]
                for key in ("geometry", "dimensions_m", "half_extents_m", "bbox_local"):
                    if key in state:
                        value = state[key]
                        return dict(value) if isinstance(value, Mapping) else {key: value}
            extent = self.call("get_object_bbox_extent", object_name=label)
            if isinstance(extent, (list, tuple)) and len(extent) == 3:
                return {"dimensions_m": list(extent)}
            raise ClientError(f"geometry metadata unavailable for {label!r}")
        if kind in ("gripper_calibration", "home_pose"):
            arm = kwargs.get("arm")
            if not isinstance(arm, str) or not arm:
                raise ValueError(f"{kind} peek requires arm")
            plural = "gripper_calibration" if kind == "gripper_calibration" else "home_poses"
            values = observation.get(plural)
            if isinstance(values, Mapping) and isinstance(values.get(arm), Mapping):
                return dict(values[arm])
            grippers = observation.get("grippers")
            if isinstance(grippers, Mapping) and isinstance(grippers.get(arm), Mapping):
                nested = grippers[arm].get(
                    "calibration" if kind == "gripper_calibration" else "home_pose"
                )
                if isinstance(nested, Mapping):
                    return dict(nested)
            raise ClientError(f"{kind} unavailable for {arm!r}")
        raise ValueError(f"unsupported metadata kind {kind!r}")


# ---------------------------------------------------------------- command line


def _json_object(text: str) -> dict[str, Any]:
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    value = json.loads(text) if text.strip() else {}
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    """``roboenv robot`` — the shell agent's window onto the robot."""
    parser = argparse.ArgumentParser(
        prog="roboenv robot",
        description=(
            "Talk to a running worker/supervisor. Exit status: 0 ok, 1 remote error, 2 usage.\n"
            "  api                     list robot API functions with docs\n"
            "  call <fn> ['{json}']    call one tool, kwargs as JSON (or @file.json)\n"
            "  poses                   ground-truth poses of all objects (observe.objects)\n"
            "  observe                 the full observation\n"
            "  capture <out.png>       save a camera image inside the run directory\n"
            "  budget / score          simulation budget / current score\n"
            "  submit                  END the episode and get the final score\n"
            "  health / shutdown"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--socket", default=None, help="default: $ROBOENV_SOCKET")
    parser.add_argument("--session", default=None, help="default: $ROBOENV_SESSION")
    parser.add_argument("--deadline", type=float, default=300.0)
    parser.add_argument("-o", "--out", default=None, help="write the JSON result to this file")
    parser.add_argument(
        "op",
        choices=[
            "api",
            "call",
            "poses",
            "observe",
            "capture",
            "budget",
            "score",
            "submit",
            "health",
            "shutdown",
        ],
    )
    parser.add_argument("name", nargs="?", help="tool name (call) or output path (capture)")
    parser.add_argument("args", nargs="?", default="{}", help="JSON kwargs for call")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``roboenv robot``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        api = RemoteAPI(args.socket, session=args.session, default_deadline_s=args.deadline)
        if args.op == "api":
            described = api.api()
            tools = described.get("tools") or {}
            lines = []
            for name, spec in sorted(tools.items()):
                doc = str(spec.get("doc") or "")
                summary = doc.splitlines()[0] if doc else ""
                lines.append(f"{name}{spec.get('signature', '')}\n    {summary}")
            result: Any = "\n".join(lines) if not args.out else described
        elif args.op == "call":
            if not args.name:
                parser.error("call requires a tool name")
            result = api.call(args.name, **_json_object(args.args))
        elif args.op == "poses":
            result = (api.observe() or {}).get("objects", {})
        elif args.op == "observe":
            result = api.observe()
        elif args.op == "capture":
            if not args.name:
                parser.error("capture requires an output path")
            result = api.capture(args.name)
        elif args.op == "budget":
            result = (api.observe() or {}).get("budget", {})
        elif args.op == "score":
            result = api.score_peek()
        elif args.op == "submit":
            result = api.submit()
        elif args.op == "health":
            result = api.health()
        else:
            result = api.shutdown()
    except (ClientError, ProtocolError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = result if isinstance(result, str) else json.dumps(result, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(
            text if isinstance(text, str) else json.dumps(result), encoding="utf-8"
        )
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
