"""Probe supervisor readiness with a fresh request identity on every invocation."""

import json
import os
import socket
import sys
import uuid
from pathlib import Path


def main():
    """Return zero only for a live, ready worker behind the supervisor."""
    identity = Path("/run/roboenv/identity.json")
    session = (json.loads(identity.read_text())["session"] if identity.is_file()
               else os.environ.get("ROBOENV_SESSION", "roboenv"))
    request = {
        "protocol": "roboenv/1",
        "session": session,
        "generation": 0,
        "request_id": "healthcheck-" + uuid.uuid4().hex,
        "deadline_s": 5,
        "op": "health",
        "args": {},
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(8)
            sock.connect(os.environ.get("ROBOENV_SOCKET", "/run/roboenv/sim.sock"))
            sock.sendall(json.dumps(request).encode() + b"\n")
            with sock.makefile("rb") as stream:
                response = json.loads(stream.readline(65536))
        matching = all(
            response.get(key) == request[key]
            for key in ("protocol", "session", "generation", "request_id")
        )
        worker = response.get("result", {}).get("worker", {})
        if not matching or response.get("ok") is not True or worker.get("state") != "ready":
            return 1
        if os.environ.get("ROBOENV_SERVER_ROBOT"):
            registry = response.get("result", {}).get("device_service", {})
            if registry.get("ready") is True and registry.get("contract") == "roboenv/devices/1":
                return 0  # Registry is immutable; do not queue behind ongoing motion.
            request.update(op="devices.list", request_id="devices-ready-" + uuid.uuid4().hex)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(8)
                sock.connect(os.environ.get("ROBOENV_SOCKET", "/run/roboenv/sim.sock"))
                sock.sendall(json.dumps(request).encode() + b"\n")
                with sock.makefile("rb") as stream:
                    response = json.loads(stream.readline(8 * 1024 * 1024 + 1))
            matching = all(
                response.get(key) == request[key]
                for key in ("protocol", "session", "generation", "request_id")
            )
            result = response.get("result", {})
            if (
                not matching
                or response.get("ok") is not True
                or result.get("schema_version") != "roboenv/devices/1"
                or not result.get("devices")
            ):
                return 1
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"supervisor health: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
