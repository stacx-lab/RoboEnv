"""Wire protocol primitives for the RoboDojo campaign harness.

The public protocol is newline-delimited JSON (NDJSON) over Unix-domain
sockets.  Every request is self-identifying so a supervisor can reject stale
clients and safely deduplicate retries after a lost response.

This module deliberately contains no simulator imports.
"""


from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import socket
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from roboenv.validation import json_loads

PROTOCOL_VERSION = "roboenv/1"
LEGACY_PROTOCOL_VERSIONS = frozenset({"codex-dojo/1"})
MAX_NDJSON_BYTES = 8 * 1024 * 1024

# These are the only operations exposed by the public client.  ``call`` is
# the controlled tool-dispatch operation; arbitrary Python ``exec`` is not a
# protocol operation.
ALLOWED_OPS = frozenset(
    {
        "health",
        "simulation.status",
        "simulation.stop",
        "environment.describe",
        "devices.list",
        "devices.describe",
        "devices.call",
        "operations.get",
        "api",
        "observe",
        "call",
        "skill",
        "score.peek",
        "capture",
        "submit",
        "reset",
        "reset_same_task",  # legacy alias of reset (codex-dojo/1)
        "checkpoint",
        "restore",
        "exec",  # practice-mode privileged REPL turns only; refused in official runs
        "shutdown",
    }
)

_REQUEST_FIELDS = frozenset(
    {
        "protocol",
        "session",
        "generation",
        "request_id",
        "deadline_s",
        "op",
        "args",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")


#: AF_UNIX addresses are capped at 108 bytes on Linux; leave room for a NUL and slack.
SOCKET_PATH_LIMIT = 100


def socket_address(path: Path | str) -> str:
    """The string to bind or connect a Unix socket at ``path`` — the path itself, or a
    short alias when it would exceed the kernel's limit.

    Measured 2026-08-28: a clean clone whose root was 45 characters long failed seven
    runtime tests with ``OSError: AF_UNIX path too long`` because pytest builds its
    socket under ``$TMPDIR`` and ``env.sh`` puts ``$TMPDIR`` inside the repository.
    The socket file still lives where the caller said; only the *address* changes: a
    symlink ``$ROBOENV_SOCKET_ALIAS_ROOT/<hash of the directory>`` (default
    ``/tmp/roboenv-<uid>``) points at the directory, and the alias is ``<link>/<name>``.
    Both ends compute the same alias from the same path, so nothing else has to agree.
    """
    text = str(path)
    if len(os.fsencode(text)) < SOCKET_PATH_LIMIT:
        return text
    target = Path(text)
    parent = target.parent.resolve()
    root = Path(os.environ.get("ROBOENV_SOCKET_ALIAS_ROOT") or f"/tmp/roboenv-{os.getuid()}")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    alias = root / hashlib.sha1(os.fsencode(str(parent))).hexdigest()[:16]
    if alias.is_symlink() and os.readlink(alias) != str(parent):
        alias.unlink()
    if not alias.exists():
        os.symlink(parent, alias)
    short = str(alias / target.name)
    if len(os.fsencode(short)) >= 108:
        raise ValueError(f"Unix socket path is too long even through an alias: {short}")
    return short


class ProtocolError(ValueError):
    """A malformed or semantically invalid wire message."""

    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_error(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out


class RequestIdConflict(ProtocolError):
    """The same deduplication key was reused for a different operation."""

    def __init__(self, request_id: str) -> None:
        super().__init__(
            "request_id_conflict",
            f"request_id {request_id!r} was already used with different semantics",
        )


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ProtocolError(
            "invalid_request",
            f"{field} must match {_IDENTIFIER_RE.pattern}",
            details={"field": field},
        )
    return value


def _json_roundtrip(value: Any, *, field: str) -> Any:
    """Validate JSON compatibility and reject NaN/Infinity."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json_loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "invalid_request",
            f"{field} must contain only finite JSON values: {exc}",
            details={"field": field},
        ) from exc


@dataclass(frozen=True)
class Request:
    protocol: str
    session: str
    generation: int
    request_id: str
    deadline_s: float
    op: str
    args: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        allowed_ops: frozenset[str] | set[str] | None = ALLOWED_OPS,
        reject_unknown: bool = True,
    ) -> Request:
        if not isinstance(raw, Mapping):
            raise ProtocolError("invalid_request", "request must be a JSON object")
        fields = set(raw)
        missing = sorted(_REQUEST_FIELDS - fields)
        if missing:
            raise ProtocolError(
                "invalid_request",
                "request is missing required fields",
                details={"missing": missing},
            )
        unknown = sorted(fields - _REQUEST_FIELDS)
        if reject_unknown and unknown:
            raise ProtocolError(
                "invalid_request",
                "request contains unknown fields",
                details={"unknown": unknown},
            )

        protocol = raw["protocol"]
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                "protocol_mismatch",
                f"expected protocol {PROTOCOL_VERSION!r}, got {protocol!r}",
            )
        session = _require_identifier(raw["session"], "session")
        request_id = _require_identifier(raw["request_id"], "request_id")

        generation = raw["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ProtocolError("invalid_request", "generation must be a non-negative integer")

        deadline = raw["deadline_s"]
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise ProtocolError("invalid_request", "deadline_s must be a number")
        deadline_s = float(deadline)
        if not math.isfinite(deadline_s) or deadline_s <= 0:
            raise ProtocolError(
                "invalid_request", "deadline_s must be finite and greater than zero"
            )

        op = raw["op"]
        if not isinstance(op, str) or not op:
            raise ProtocolError("invalid_request", "op must be a non-empty string")
        if allowed_ops is not None and op not in allowed_ops:
            raise ProtocolError(
                "unsupported_op",
                f"operation {op!r} is not allowed",
                details={"allowed": sorted(allowed_ops)},
            )

        args = raw["args"]
        if not isinstance(args, dict):
            raise ProtocolError("invalid_request", "args must be a JSON object")
        args = _json_roundtrip(args, field="args")
        return cls(
            protocol=protocol,
            session=session,
            generation=generation,
            request_id=request_id,
            deadline_s=deadline_s,
            op=op,
            args=args,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "session": self.session,
            "generation": self.generation,
            "request_id": self.request_id,
            "deadline_s": self.deadline_s,
            "op": self.op,
            "args": copy.deepcopy(self.args),
        }

    def semantic_fingerprint(self) -> str:
        """Fingerprint operation semantics, excluding retry-local deadline.

        A caller may retransmit the same request with a shorter remaining
        deadline.  It may not reuse the id for another operation or arguments.
        """

        semantic = {
            "protocol": self.protocol,
            "session": self.session,
            "generation": self.generation,
            "op": self.op,
            "args": self.args,
        }
        blob = json.dumps(
            semantic,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class Response:
    protocol: str
    session: str
    generation: int
    request_id: str
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None

    @classmethod
    def success(cls, request: Request, result: Any = None) -> Response:
        return cls(
            protocol=PROTOCOL_VERSION,
            session=request.session,
            generation=request.generation,
            request_id=request.request_id,
            ok=True,
            result=_json_roundtrip(result, field="result"),
        )

    @classmethod
    def failure(
        cls,
        request: Request,
        code: str,
        message: str,
        *,
        details: Any = None,
        retryable: bool = False,
    ) -> Response:
        error: dict[str, Any] = {
            "code": str(code),
            "message": str(message),
            "retryable": bool(retryable),
        }
        if details is not None:
            error["details"] = _json_roundtrip(details, field="error.details")
        return cls(
            protocol=PROTOCOL_VERSION,
            session=request.session,
            generation=request.generation,
            request_id=request.request_id,
            ok=False,
            error=error,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Response:
        if not isinstance(raw, Mapping):
            raise ProtocolError("invalid_response", "response must be a JSON object")
        required = {"protocol", "session", "generation", "request_id", "ok"}
        missing = sorted(required - set(raw))
        if missing:
            raise ProtocolError(
                "invalid_response", "response is missing required fields", details=missing
            )
        if raw["protocol"] != PROTOCOL_VERSION:
            raise ProtocolError("protocol_mismatch", "worker response protocol mismatch")
        session = _require_identifier(raw["session"], "session")
        request_id = _require_identifier(raw["request_id"], "request_id")
        generation = raw["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ProtocolError(
                "invalid_response",
                "response generation must be a non-negative integer",
            )
        ok = raw["ok"]
        if not isinstance(ok, bool):
            raise ProtocolError("invalid_response", "response ok must be boolean")
        if ok:
            return cls(
                PROTOCOL_VERSION,
                session,
                generation,
                request_id,
                True,
                _json_roundtrip(raw.get("result"), field="result"),
            )
        error = raw.get("error")
        if not isinstance(error, dict):
            raise ProtocolError("invalid_response", "failed response needs error object")
        return cls(
            PROTOCOL_VERSION,
            session,
            generation,
            request_id,
            False,
            error=_json_roundtrip(error, field="error"),
        )

    def as_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "protocol": self.protocol,
            "session": self.session,
            "generation": self.generation,
            "request_id": self.request_id,
            "ok": self.ok,
        }
        if self.ok:
            base["result"] = copy.deepcopy(self.result)
        else:
            base["error"] = copy.deepcopy(self.error)
        return base


def encode_ndjson(message: Mapping[str, Any], *, max_bytes: int = MAX_NDJSON_BYTES) -> bytes:
    try:
        payload = (
            json.dumps(
                dict(message),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_json", f"message is not JSON serializable: {exc}") from exc
    if len(payload) > max_bytes:
        raise ProtocolError(
            "message_too_large",
            f"encoded message is {len(payload)} bytes; limit is {max_bytes}",
        )
    return payload


def decode_ndjson(line: bytes | str, *, max_bytes: int = MAX_NDJSON_BYTES) -> dict[str, Any]:
    raw = line.encode("utf-8") if isinstance(line, str) else line
    if len(raw) > max_bytes:
        raise ProtocolError("message_too_large", f"message exceeds {max_bytes} bytes")
    if not raw.endswith(b"\n"):
        raise ProtocolError("invalid_ndjson", "message must end with a newline")
    if b"\n" in raw[:-1] or b"\r" in raw[:-1]:
        raise ProtocolError("invalid_ndjson", "message must occupy exactly one line")
    try:
        value = json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("invalid_json", "cannot decode strict finite JSON object") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_json", "NDJSON value must be a JSON object")
    return value


def read_ndjson(stream: BinaryIO, *, max_bytes: int = MAX_NDJSON_BYTES) -> dict[str, Any]:
    line = stream.readline(max_bytes + 1)
    if not line:
        raise EOFError("peer closed before sending an NDJSON message")
    if len(line) > max_bytes:
        raise ProtocolError("message_too_large", f"message exceeds {max_bytes} bytes")
    return decode_ndjson(line, max_bytes=max_bytes)


def recv_ndjson(sock: socket.socket, *, max_bytes: int = MAX_NDJSON_BYTES) -> dict[str, Any]:
    """Receive one message without consuming bytes from a following message."""

    chunks = bytearray()
    while True:
        part = sock.recv(min(65536, max_bytes + 1 - len(chunks)))
        if not part:
            raise EOFError("peer closed before sending an NDJSON message")
        chunks.extend(part)
        newline = chunks.find(b"\n")
        if newline >= 0:
            if newline != len(chunks) - 1:
                # This helper represents one-request-per-connection.  Server
                # handlers that support multiple lines use ``read_ndjson``.
                raise ProtocolError("invalid_ndjson", "unexpected bytes after first message")
            return decode_ndjson(bytes(chunks), max_bytes=max_bytes)
        if len(chunks) > max_bytes:
            raise ProtocolError("message_too_large", f"message exceeds {max_bytes} bytes")


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    response: dict[str, Any] | None = None


class RequestCache:
    """Bounded LRU cache keyed by (session, generation, request_id).

    The cache stores both successes and failures.  Retrying a timed-out
    operation with the same id therefore cannot accidentally execute it again.
    """

    def __init__(self, max_entries: int = 2048) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._items: OrderedDict[tuple[str, int, str], tuple[str, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _key(request: Request) -> tuple[str, int, str]:
        return (request.session, request.generation, request.request_id)

    def lookup(self, request: Request) -> CacheLookup:
        key = self._key(request)
        item = self._items.get(key)
        if item is None:
            return CacheLookup(False)
        fingerprint, response = item
        if fingerprint != request.semantic_fingerprint():
            raise RequestIdConflict(request.request_id)
        self._items.move_to_end(key)
        return CacheLookup(True, copy.deepcopy(response))

    def store(self, request: Request, response: Mapping[str, Any]) -> None:
        key = self._key(request)
        fingerprint = request.semantic_fingerprint()
        existing = self._items.get(key)
        if existing is not None and existing[0] != fingerprint:
            raise RequestIdConflict(request.request_id)
        self._items[key] = (fingerprint, _json_roundtrip(dict(response), field="response"))
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
