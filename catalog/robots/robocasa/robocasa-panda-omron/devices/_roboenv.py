"""Portable driver dialog and transport, shared by one exported device collection."""

import contextlib
import http.client
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property
from pathlib import Path
from urllib.parse import quote, urlsplit

MAX_BYTES = 8 * 1024 * 1024


class WireError(RuntimeError):
    def __init__(self, message, code, request_id, details=None):
        super().__init__(message)
        self.code, self.request_id, self.details = code, request_id, details


class Transport:
    def __init__(self, driver, *, device=None):
        self.directory = Path(driver).resolve().parent
        self.device = device
        self._local = threading.local()
        self._connections = set()
        self._connections_lock = threading.Lock()
        self.connection = None
        self.last_exchange = 0
        self.used = False
        self.unloaded = False
        self.deadline = None
        self._motions = {}
        self._motion_lock = threading.Lock()

    @property
    def connection(self):
        return getattr(self._local, "connection", None)

    @connection.setter
    def connection(self, value):
        self._local.connection = value
        if value is not None:
            with self._connections_lock:
                self._connections.add(value)

    @property
    def last_exchange(self):
        return getattr(self._local, "last_exchange", 0)

    @last_exchange.setter
    def last_exchange(self, value):
        self._local.last_exchange = value

    def close_all(self):
        with self._connections_lock:
            connections, self._connections = self._connections, set()
        for connection in connections:
            if connection.sock is not None:
                with contextlib.suppress(OSError):
                    connection.sock.shutdown(socket.SHUT_RDWR)
            connection.close()

    @cached_property
    def profile(self):
        import yaml

        return yaml.safe_load((self.directory / "profile.yaml").read_text())

    def settings(self):
        return self.profile

    @cached_property
    def config(self):
        file = self.directory / "connection.json"
        return (
            json.loads(file.read_text())
            if file.is_file()
            else {
                "endpoint": os.environ.get("ROBOENV_ENDPOINT"),
                "socket": os.environ.get("ROBOENV_SOCKET"),
                "session": os.environ.get("ROBOENV_SESSION"),
                "generation": int(os.environ.get("ROBOENV_GENERATION", "0")),
            }
        )

    @cached_property
    def endpoint(self):
        if self.config.get("endpoint") or self.config.get("socket"):
            return self.config.get("endpoint")
        return self.settings()["access"]["connection"]["url"]

    @cached_property
    def identity(self):
        identity = {
            "protocol": "roboenv/1",
            "session": self.config.get("session") or "local",
            "generation": self.config.get("generation", 0),
        }
        if self.endpoint and not self.config.get("session"):
            try:
                found = json.loads(self.exchange(self.endpoint, "GET"))
                if (
                    not isinstance(found, dict)
                    or found.get("protocol") != "roboenv/1"
                    or not isinstance(found.get("session"), str)
                    or type(found.get("generation")) is not int
                    or found["generation"] < 0
                ):
                    raise ValueError("Endpoint returned an invalid environment identity")
                identity = {key: found[key] for key in identity}
            except Exception:
                self.close()
                raise
        return identity

    @cached_property
    def target(self):
        if self.device is not None:
            return self.device
        profile = self.settings()
        key = "camera" if profile["type"] in {"rgb", "rgbd"} else "device"
        return profile["access"]["connection"][key]

    @cached_property
    def can_stop(self):
        return "stop" in self.settings()["access"]["driver"]["actions"]

    @cached_property
    def is_world(self):
        # Typed drivers route by profile; an explicit device name is a generic client.
        return self.device is None and self.settings()["type"] == "world"

    def close(self):
        if self.connection is not None:
            with self._connections_lock:
                self._connections.discard(self.connection)
            self.connection.close()
            self.connection = None

    def timeout(self):
        if self.deadline is None:
            return 605
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Driver cleanup deadline exceeded")
        return remaining

    def exchange(self, endpoint, method, path="/", body=None, destination=None):
        if self.connection is not None and time.monotonic() - self.last_exchange >= 5:
            self.close()
        try:
            if self.connection is None:
                url = urlsplit(endpoint)
                if (
                    url.scheme not in {"http", "https"}
                    or not url.hostname
                    or url.username is not None
                    or url.password is not None
                    or url.path not in {"", "/"}
                    or url.query
                    or url.fragment
                ):
                    raise ValueError("Expected an HTTP(S) device endpoint")
                cls = (
                    http.client.HTTPSConnection
                    if url.scheme == "https"
                    else http.client.HTTPConnection
                )
                self.connection = cls(url.hostname, url.port, timeout=self.timeout())
            self.connection.timeout = self.timeout()
            if self.connection.sock is not None:
                self.connection.sock.settimeout(self.timeout())
            self.connection.request(
                method, path, body=body, headers={"Content-Type": "application/json"}
            )
            response = self.connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            expected = response.length
            if expected is None:
                raise ValueError("Response requires Content-Length")
            if destination is None:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES or len(raw) != expected:
                    raise ValueError("Oversized or incomplete response")
            else:
                size = 0
                with destination.open("wb") as stream:
                    while chunk := response.read(65536):
                        size += stream.write(chunk)
                if size != expected:
                    raise ValueError("Incomplete artifact")
                raw = None
            self.last_exchange = time.monotonic()
            if response.will_close:
                self.close()
            return raw
        except BaseException:
            # Also discard an interrupted request before SIGTERM cleanup uses the wire.
            self.close()
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise

    def cancel_motions(self):
        with self._motion_lock:
            identities = set(self._motions.values())
            self._motions.clear()
        for identity in identities:
            self.remote("cancel_motion", {"execution_id": identity})

    def remote(self, action, args):
        identity = self.identity
        request_id = uuid.uuid4().hex
        if action == "wait":
            request_id = getattr(self._local, "request_id", request_id)
        request_id = f"driver-{request_id}"
        request = {
            **identity,
            "request_id": request_id,
            "deadline_s": 600 if self.deadline is None else self.timeout(),
            "op": {"cancel_motion": "devices.cancel"}.get(action, "devices.call"),
            "args": args
            if action == "cancel_motion"
            else {"device": self.target, "action": action, "args": args},
        }
        payload = json.dumps(request, allow_nan=False).encode() + b"\n"
        if len(payload) > MAX_BYTES:
            raise ValueError("Device request is too large")
        if action in {"write", "wait"}:
            with self._motion_lock:
                self._motions[request_id if action == "wait" else self.target] = request_id
        self.used, self.unloaded = True, False
        # A broken connection never causes this request to be resubmitted.
        try:
            if self.endpoint:
                raw = self.exchange(self.endpoint, "POST", body=payload)
            else:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout())
                    sock.connect(self.config["socket"])
                    sock.sendall(payload)
                    with sock.makefile("rb") as stream:
                        raw = stream.readline(MAX_BYTES + 1)
            response = json.loads(raw)
            if (
                len(raw) > MAX_BYTES
                or not raw.endswith(b"\n")
                or any(response.get(key) != request[key] for key in (*identity, "request_id"))
                or type(response.get("generation")) is not int
                or type(response.get("ok")) is not bool
            ):
                raise ValueError("Incomplete or mismatched device response")
            if response["ok"] and (
                not isinstance(response.get("result"), dict) or "data" not in response["result"]
            ):
                raise ValueError("Device response has no result data")
            if not response["ok"] and not isinstance(response.get("error"), dict):
                raise ValueError("Device response has no error details")
        except Exception as exc:
            self.close()
            raise WireError(str(exc), "outcome_unknown", request_id) from exc
        if action == "wait":
            with self._motion_lock:
                self._motions.pop(request_id, None)
        if not response["ok"]:
            error = response["error"]
            raise WireError(
                error.get("message", str(error)),
                error.get("code"),
                request_id,
                error.get("details"),
            )
        self.unloaded = action == "unload"
        data = response["result"]["data"]
        if "simulation" in response["result"] and isinstance(data, dict):
            data = {**data, "simulation": response["result"]["simulation"]}
        # Agents see only this data, so the world clock also reports the episode budget
        # to let them pace waits; other devices' readings stay free of it.
        if (
            action in {"read", "wait", "status"}
            and isinstance(data, dict)
            and "budget" in response["result"]
            and self.is_world
        ):
            data = {**data, "budget": response["result"]["budget"]}
        if action == "read" and isinstance(data, dict):
            data["received_time_s"] = time.time()
        if self.endpoint and action == "read" and isinstance(data, dict) and data.get("file"):
            directory = Path(tempfile.mkdtemp(prefix="roboenv-images-"))
            try:
                for key in ("file", "depth_file"):
                    if data.get(key):
                        name = Path(data[key]).name
                        route = (
                            f"/artifacts/{identity['session']}/"
                            f"{identity['generation']}/{quote(name)}"
                        )
                        self.exchange(self.endpoint, "GET", route, destination=directory / name)
                        data[key] = str(directory / name)
            except BaseException:
                shutil.rmtree(directory)
                raise
        if self.unloaded:
            self.close()
        return data


def main(handle, transport):
    serving = "--serve" in sys.argv
    can_stop = transport.can_stop if serving else False
    pools = {}
    writes = []
    pending = set()
    output_lock = threading.Lock()
    stop_tail = None

    def respond(action, args):
        try:
            if not isinstance(action, str) or not isinstance(args, dict):
                raise ValueError("Expected an action name and an arguments object")
            return handle(action, args)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def terminate(_signum, _frame):
        raise SystemExit(0)

    def expired(_signum, _frame):
        raise TimeoutError("Driver cleanup exceeded 500 ms")

    previous = signal.signal(signal.SIGTERM, terminate) if serving else None
    # Keep the response descriptor private; native SDK writes to fd 1 are logs too.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    if serving:
                        pools = {
                            name: ThreadPoolExecutor(max_workers=count)
                            for name, count in (
                                ("read", 8),
                                ("write", 1),
                                ("stop", 1),
                                ("capture", 1),
                            )
                        }

                        def send(request, cancel=None, predecessor=None):
                            transport._local.request_id = request["id"]
                            if predecessor is not None:
                                predecessor.result()
                            if cancel is not None and cancel.is_set():
                                result = {"ok": False, "error": "queued write interrupted by stop"}
                            else:
                                result = respond(request["action"], request.get("args", {}))
                            if cancel is not None and cancel.is_set():
                                result = {"ok": False, "error": "device write interrupted by stop"}
                            with output_lock:
                                output.write(
                                    json.dumps({**result, "id": request["id"]}, allow_nan=False)
                                    + "\n"
                                )
                                output.flush()

                        for line in sys.stdin:
                            try:
                                request = json.loads(line)
                                if not isinstance(request, dict) or not isinstance(
                                    request.get("id"), str
                                ):
                                    raise ValueError(
                                        "persistent requests require a string id; "
                                        "migrate the caller"
                                    )
                            except ValueError as error:
                                with output_lock:
                                    output.write(
                                        json.dumps({"id": None, "ok": False, "error": str(error)})
                                        + "\n"
                                    )
                                    output.flush()
                                break
                            target = request.get("interrupt")
                            if isinstance(target, str):
                                transport.remote(
                                    "cancel_motion", {"execution_id": f"driver-{target}"}
                                )
                                with output_lock:
                                    output.write(
                                        json.dumps({"id": request["id"], "ok": True}) + "\n"
                                    )
                                    output.flush()
                                continue
                            if target is True:
                                for _, event in writes:
                                    event.set()
                                if any(not future.done() for future, _ in writes):
                                    transport.cancel_motions()
                                with output_lock:
                                    output.write(
                                        json.dumps({"id": request["id"], "ok": True}) + "\n"
                                    )
                                    output.flush()
                                continue
                            action = request["action"]
                            pending = {future for future in pending if not future.done()}
                            writes = [
                                (future, event) for future, event in writes if not future.done()
                            ]
                            if action in {"stop", "unload"}:
                                for _, event in writes:
                                    event.set()
                            if action in {"load", "unload"}:
                                if (
                                    action == "unload"
                                    and can_stop
                                    and any(not future.done() for future, _ in writes)
                                ):
                                    respond("stop", {})
                                if action == "unload":
                                    transport.cancel_motions()
                                wait(pending)
                                send(request)
                                continue
                            kind = (
                                "read"
                                if action in {"read", "wait", "status", "fk", "ik"}
                                else "stop"
                                if action == "stop"
                                else "capture"
                                if action in {"record_start", "record_stop"}
                                else "write"
                            )
                            cancel = (
                                threading.Event() if kind == "write" or action == "wait" else None
                            )
                            future = pools[kind].submit(
                                send, request, cancel, stop_tail if kind == "write" else None
                            )
                            if kind == "stop":
                                stop_tail = future
                            pending.add(future)
                            if cancel is not None:
                                writes.append((future, cancel))
                        wait(pending - {future for future, _ in writes})
                        return
                    action = sys.argv[1] if len(sys.argv) > 1 else "read"
                    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
                    output.write(json.dumps(respond(action, args), allow_nan=False) + "\n")
                    output.flush()
                finally:
                    for _, event in writes:
                        event.set()
                    if serving:
                        signal.signal(signal.SIGTERM, signal.SIG_IGN)
                        alarm = signal.signal(signal.SIGALRM, expired)
                        transport.deadline = time.monotonic() + 0.5
                        signal.setitimer(signal.ITIMER_REAL, 0.5)
                        try:
                            if transport.used and not transport.unloaded:
                                transport.cancel_motions()
                        except Exception as exc:
                            print(str(exc), file=sys.stderr)
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, alarm)
                    transport.close_all()
                    for pool in pools.values():
                        pool.shutdown(wait=True, cancel_futures=True)
        finally:
            os.dup2(output.fileno(), sys.stdout.fileno())
            if serving:
                signal.signal(signal.SIGTERM, previous)
