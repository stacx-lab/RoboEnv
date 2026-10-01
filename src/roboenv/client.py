"""The stable environment wire client, independent of any agent runtime.

All peers use the roboenv/1 contract. SDK and runtime versions must match;
older wire identifiers are not aliases. This module uses only the standard library.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import socket
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

from roboenv.devices.contract import SERVICE_OPS
from roboenv.validation import json_loads

PROTOCOL = "roboenv/1"
MAX_BYTES = 8 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
PUBLIC_OPS = (
    frozenset(
        {
            "health",
            "observe",
            "score.peek",
            "capture",
            "simulation.status",
            "simulation.stop",
        }
    )
    | SERVICE_OPS
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

    def __init__(self, socket_path=None, *, endpoint=None, session=None, generation=None):
        if socket_path is not None and str(socket_path).startswith(("http://", "https://")):
            if endpoint is not None:
                raise ValueError("Choose one endpoint")
            endpoint, socket_path = str(socket_path), None
        if socket_path is not None and endpoint is not None:
            raise ValueError("Choose a Unix socket or an HTTP endpoint")
        endpoint = endpoint or (os.environ.get("ROBOENV_ENDPOINT") if socket_path is None else None)
        self.endpoint = None
        if endpoint:
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                raise ValueError("endpoint must be http(s)://HOST:PORT")
            if parsed.port == 0:
                raise ValueError("endpoint must identify an assigned nonzero port")
            self.endpoint = endpoint.rstrip("/")
        self._discover_identity = bool(self.endpoint and session is None)
        self.socket_path = str(
            socket_path or os.environ.get("ROBOENV_SOCKET") or "/run/roboenv/sim.sock"
        )
        self.session = (
            session if session is not None else os.environ.get("ROBOENV_SESSION", "roboenv")
        )
        if not isinstance(self.session, str) or not IDENTIFIER.fullmatch(self.session):
            raise ValueError("session must be a protocol identifier")
        if generation is None:
            generation = 0 if self.endpoint else int(os.environ.get("ROBOENV_GENERATION", "0"))
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        self.generation = generation

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
        if self._discover_identity:
            self._identify(timeout)
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
        if self.endpoint:
            raw = self._http_exchange(payload, timeout, op, request["request_id"])
        else:
            raw = self._unix_exchange(payload, timeout, op, request["request_id"])
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

    def _identify(self, timeout):
        """Discover internal journal identity; users select services by host and port."""
        raw = self._http_exchange(None, timeout, "connect", "connect", method="GET")
        identity = json_loads(raw)
        if (
            not isinstance(identity, dict)
            or identity.get("protocol") != PROTOCOL
            or not isinstance(identity.get("session"), str)
            or not IDENTIFIER.fullmatch(identity["session"])
            or type(identity.get("generation")) is not int
            or identity["generation"] < 0
        ):
            raise ValueError("Endpoint did not return a valid environment identity")
        self.session, self.generation = identity["session"], identity["generation"]
        self._discover_identity = False

    def _http_exchange(self, payload, timeout, op, request_id, *, method="POST"):
        url = urlsplit(self.endpoint)
        connection_type = (
            http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(url.hostname, url.port, timeout=timeout + 5)
        try:
            connection.connect()  # failure here proves no command has been sent
            try:
                connection.request(
                    method,
                    url.path or "/",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                raw = response.read(MAX_BYTES + 1)
                if response.status != 200:
                    raise UncertainResult(
                        op, request_id, f"HTTP endpoint returned {response.status}"
                    )
                return raw
            except (OSError, http.client.HTTPException) as exc:
                raise UncertainResult(op, request_id, str(exc)) from exc
        finally:
            connection.close()

    def _unix_exchange(self, payload, timeout, op, request_id):
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
                    return stream.readline(MAX_BYTES + 1)
            except OSError as exc:
                raise UncertainResult(op, request_id, str(exc)) from exc

    def describe(self):
        return self.request("environment.describe")

    def devices(self):
        return self.request("devices.list")

    def device(self, name):
        return Device(self, name)

    def operation(self, request_id):
        return self.request("operations.get", {"request_id": request_id})




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

    def write(self, *, duration_s, timeout=600.0, request_id=None, **kwargs):
        return self.call(
            "write", {**kwargs, "duration_s": duration_s}, timeout=timeout, request_id=request_id
        )

    def wait(self, duration_s):
        return self.call("wait", {"duration_s": duration_s})

    def stop(self):
        return self.call("stop")

    def fk(self, joints):
        return self.call("fk", {"joints": joints})

    def ik(self, pose, *, seed=None):
        return self.call("ik", {"pose": pose, **({"seed": seed} if seed is not None else {})})
