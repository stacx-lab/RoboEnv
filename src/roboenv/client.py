"""The stable environment wire client, independent of any agent runtime.

All peers use the roboenv/1 contract. SDK and runtime versions must match;
older wire identifiers are not aliases. This module uses only the standard library.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .contract import SERVICE_OPS
from .validation import json_loads

PROTOCOL = "roboenv/1"
MAX_BYTES = 8 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
PUBLIC_OPS = (
    frozenset({"health", "api", "observe", "call", "score.peek", "capture", "vla"}) | SERVICE_OPS
)


class RemoteError(RuntimeError):
    """Structured server rejection; transport failures remain ordinary I/O errors."""

    def __init__(self, op, error, request_id):
        self.code = error.get("code", "operation_failed")
        self.details = error.get("details")
        self.retryable = error.get("retryable", False)
        self.request_id = request_id
        super().__init__(f"{op}: {self.code}: {error.get('message', error)}")


class UncertainResult(RuntimeError):
    """A sent request has no trustworthy reply. Inspect its id; do not replay blindly."""

    def __init__(self, op, request_id, message):
        self.op, self.request_id = op, request_id
        super().__init__(f"{op}: outcome unknown for {request_id}: {message}")


def jsonable(value: Any) -> Any:
    for name in ("tolist", "item"):
        convert = getattr(value, name, None)
        if callable(convert):
            return convert()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class Client:
    """One request per connection; identity-checked replies and no motion resubmission."""

    allowed_ops = PUBLIC_OPS

    def __init__(self, socket_path=None, *, session=None, generation=None):
        self.socket_path = str(
            socket_path or os.environ.get("ROBOENV_SOCKET") or "/run/roboenv/sim.sock"
        )
        self.session = (
            session if session is not None else os.environ.get("ROBOENV_SESSION", "harbor")
        )
        if not isinstance(self.session, str) or not IDENTIFIER.fullmatch(self.session):
            raise ValueError("session must be a protocol identifier")
        if generation is None:
            generation = int(os.environ.get("ROBOENV_GENERATION", "0"))
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        self.generation = generation
        self._api = None

    def request(self, op, args=None, *, timeout=600.0, request_id=None):
        if op not in self.allowed_ops:
            raise ValueError(f"{op} is not an agent operation")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        if request_id is not None and (
            not isinstance(request_id, str) or not IDENTIFIER.fullmatch(request_id)
        ):
            raise ValueError("request_id must be a protocol identifier")
        if args is not None and not isinstance(args, dict):
            raise ValueError("args must be a JSON object")
        request = {
            "protocol": PROTOCOL,
            "session": self.session,
            "generation": self.generation,
            "request_id": request_id or f"roboenv-{uuid.uuid4().hex}",
            "deadline_s": float(timeout),
            "op": op,
            "args": args or {},
        }
        payload = json.dumps(request, default=jsonable, allow_nan=False).encode() + b"\n"
        if len(payload) > MAX_BYTES:
            raise ValueError("request exceeds the protocol size limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout + 5)
            deadline = time.monotonic() + min(timeout, 30)
            while True:
                try:
                    sock.connect(self.socket_path)
                    break
                except (
                    BlockingIOError,
                    InterruptedError,
                    ConnectionRefusedError,
                    FileNotFoundError,
                ):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            try:
                sock.sendall(payload)
                with sock.makefile("rb") as stream:
                    raw = stream.readline(MAX_BYTES + 1)
            except OSError as exc:
                raise UncertainResult(op, request["request_id"], str(exc)) from exc
        if len(raw) > MAX_BYTES or not raw.endswith(b"\n"):
            raise UncertainResult(op, request["request_id"], "incomplete or oversized response")
        try:
            response = json_loads(raw)
        except ValueError as exc:
            raise UncertainResult(op, request["request_id"], "invalid JSON response") from exc
        if not isinstance(response, dict):
            raise UncertainResult(op, request["request_id"], "response must be an object")
        if type(response.get("generation")) is not int:
            raise UncertainResult(op, request["request_id"], "response generation must be typed")
        for key in ("protocol", "session", "generation", "request_id"):
            if response.get(key) != request[key]:
                raise UncertainResult(op, request["request_id"], f"response {key} mismatch")
        if not isinstance(response.get("ok"), bool) or (
            response["ok"] is False and not isinstance(response.get("error"), dict)
        ):
            raise UncertainResult(op, request["request_id"], "invalid response verdict")
        if response.get("ok") is not True:
            raise RemoteError(op, response.get("error") or {}, request["request_id"])
        return response.get("result")

    def describe(self):
        return self.request("environment.describe")

    def devices(self):
        return self.request("devices.list")

    def device(self, name):
        return Device(self, name)

    def operation(self, request_id):
        return self.request("operations.get", {"request_id": request_id})

    def api(self):
        if self._api is None:
            self._api = self.request("api")
        return self._api

    def vla(self, n_actions=50, *, instruction=None, policy=None, until=None):
        arguments = {"n_actions": n_actions}
        if instruction is not None:
            arguments["instruction"] = instruction
        if policy is not None:
            arguments["policy"] = policy
        if until is not None:
            if (
                self.api().get("context", {}).get("vla_callback_protocol")
                != "roboenv/vla-callback/1"
            ):
                raise ValueError("environment does not advertise per-action policy callbacks")
            from .policy import run_with_callback

            return run_with_callback(self, arguments, until)
        return self.request("vla", arguments)

    def call(self, _name, *args, **kwargs):
        entry = self.api().get("tools", {}).get(_name)
        if entry is None:
            raise ValueError(f"unknown public function: {_name}")
        parameters = entry.get("parameters", [])
        if len(args) > len(parameters):
            raise TypeError(f"too many arguments for {_name}")
        for name, value in zip(parameters, args, strict=False):
            if name in kwargs:
                raise TypeError(f"duplicate argument: {name}")
            kwargs[name] = value
        return self.request("call", {"name": _name, "kwargs": kwargs})


class Device:
    """A remote logical device. Command completion is not task success."""

    def __init__(self, client, name):
        self.client, self.name = client, name

    def describe(self):
        return self.client.request("devices.describe", {"device": self.name})

    def call(self, action, args=None, *, timeout=600.0, request_id=None):
        return self.client.request(
            "devices.call",
            {"device": self.name, "action": action, "args": {} if args is None else args},
            timeout=timeout,
            request_id=request_id,
        )

    def read(self, **kwargs):
        return self.call("read", kwargs)

    def write(self, target, **kwargs):
        return self.call("write", {"target": target, **kwargs})

    def stop(self):
        return self.call("stop")


_default = None
_bound_client = ContextVar("roboenv_bound_client", default=None)


@contextmanager
def _bind_client(client):
    """Private hook for trusted runtime drivers; never accepts code from the wire."""
    token = _bound_client.set(client)
    try:
        yield
    finally:
        _bound_client.reset(token)


def default_client():
    global _default
    if _bound_client.get() is not None:
        return _bound_client.get()
    if _default is None:
        _default = Client()
    return _default


def rpc(op, args=None, timeout=600.0):
    """Compatibility entry used by imported environment drivers."""
    return default_client().request(op, args, timeout=timeout)


def api():
    return default_client().api()


def call(_name, *args, **kwargs):
    return default_client().call(_name, *args, **kwargs)


class _Functions:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: call(name, *args, **kwargs)


sim = _Functions()


def capture(relative_path):
    result = rpc("capture", {"path": relative_path})
    if isinstance(result, dict) and result.get("path"):
        return result["path"]
    return os.path.join(
        os.environ.get("ROBOENV_RUN_DIR", "/run/roboenv/run"), "captures", relative_path
    )
