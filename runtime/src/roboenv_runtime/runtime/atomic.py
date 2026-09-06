"""Durable append-only journals and atomic JSON replacement.

Journal entries use one ``O_APPEND`` write while holding an advisory lock and
are fsynced before returning.  Authoritative JSON files are written to a
same-directory temporary file, fsynced, replaced atomically, and followed by
a directory fsync.
"""

# Moved from codex_dojo_alltasks/harness/atomic.py on 2026-08-26 (refactor P3).

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping

JOURNAL_VERSION = 1
_PHASES = frozenset({"intent", "commit", "error"})


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _json_bytes(value: Mapping[str, Any], *, newline: bool = False) -> bytes:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while persisting journal")
        view = view[written:]


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Durably append exactly one compact JSON object and newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(record, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(target, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    mode: int = 0o600,
    indent: int | None = 2,
) -> None:
    """Durably replace a JSON file without exposing a partial candidate."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{target.name}."
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    replaced = False
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        replaced = True
        _fsync_directory(target.parent)
    finally:
        if not replaced:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    mode: int = 0o600,
) -> None:
    """Durably replace a UTF-8 text file without exposing partial content."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{target.name}."
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    replaced = False
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="", closefd=True) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        replaced = True
        _fsync_directory(target.parent)
    finally:
        if not replaced:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(
    path: str | os.PathLike[str], *, tolerate_truncated_tail: bool = True
) -> Iterator[dict[str, Any]]:
    """Read a journal, optionally ignoring only a malformed final line."""

    target = Path(path)
    if not target.exists():
        return
    data = target.read_bytes()
    lines = data.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        complete = raw.endswith(b"\n")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            is_tail = index == len(lines) - 1
            if tolerate_truncated_tail and is_tail and not complete:
                return
            raise
        if not isinstance(value, dict):
            raise ValueError(f"journal line {index + 1} is not an object")
        yield value


class AtomicJournal:
    """Thread-safe action journal with explicit intent/commit/error phases."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, phase: str, **fields: Any) -> dict[str, Any]:
        if phase not in _PHASES:
            raise ValueError(f"phase must be one of {sorted(_PHASES)}")
        record = {
            "journal_version": JOURNAL_VERSION,
            "phase": phase,
            "ts": _utc_now(),
            "time_ns": __import__("time").time_ns(),
            **fields,
        }
        # Validate before opening the journal so a bad value cannot leave a
        # misleading intent without its accompanying request data.
        _json_bytes(record)
        with self._lock:
            append_jsonl(self.path, record)
        return record

    def intent(self, request: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
        return self.append("intent", request=dict(request), **fields)

    def commit(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        return self.append("commit", request=dict(request), response=dict(response), **fields)

    def error(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        return self.append("error", request=dict(request), response=dict(response), **fields)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(iter_jsonl(self.path))

    def completed(self) -> dict[tuple[str, int, str], dict[str, Any]]:
        """Latest durable terminal response for each deduplication key."""

        out: dict[tuple[str, int, str], dict[str, Any]] = {}
        for event in iter_jsonl(self.path):
            if event.get("phase") not in {"commit", "error"}:
                continue
            request = event.get("request")
            response = event.get("response")
            if not isinstance(request, dict) or not isinstance(response, dict):
                continue
            try:
                key = (
                    str(request["session"]),
                    int(request["generation"]),
                    str(request["request_id"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = {"request": request, "response": response}
        return out
