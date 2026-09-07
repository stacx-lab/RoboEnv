"""Portable device driver: stdlib transport and Complex's local kinematics."""

import http.client
import json
import os
import socket
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

MAX_BYTES = 8 * 1024 * 1024


class WireError(RuntimeError):
    def __init__(self, message, code, request_id):
        super().__init__(message)
        self.code, self.request_id = code, request_id


def exchange(endpoint, method, path="/", body=None, destination=None):
    url = urlsplit(endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("Expected an HTTP(S) device endpoint")
    connection = (http.client.HTTPSConnection if url.scheme == "https"
                  else http.client.HTTPConnection)(url.hostname, url.port, timeout=605)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        if destination is not None:
            with destination.open("wb") as stream:
                while chunk := response.read(65536):
                    stream.write(chunk)
            return None
        return response.read(MAX_BYTES + 1)
    finally:
        connection.close()


def remote(action, args):
    connection = Path(__file__).with_name("connection.json")
    config = json.loads(connection.read_text()) if connection.is_file() else {
        "endpoint": os.environ.get("ROBOENV_ENDPOINT"),
        "socket": os.environ.get("ROBOENV_SOCKET"),
        "session": os.environ.get("ROBOENV_SESSION"),
        "generation": int(os.environ.get("ROBOENV_GENERATION", "0")),
    }
    endpoint = config.get("endpoint")
    if not endpoint and not config.get("socket"):
        endpoint = "http://127.0.0.1:8123"
    identity = {"protocol": "roboenv/1", "session": config.get("session") or "harbor",
                "generation": config.get("generation", 0)}
    if endpoint and not config.get("session"):
        found = json.loads(exchange(endpoint, "GET"))
        if found.get("protocol") != "roboenv/1" or not isinstance(found.get("session"), str) or (
            type(found.get("generation")) is not int or found["generation"] < 0
        ):
            raise ValueError("Endpoint returned an invalid environment identity")
        identity = {key: found[key] for key in identity}
    request_id = f"driver-{uuid.uuid4().hex}"
    request = {**identity, "request_id": request_id, "deadline_s": 600,
               "op": "devices.call", "args": {"device": Path(__file__).parent.name,
                                               "action": action, "args": args}}
    payload = json.dumps(request, allow_nan=False).encode() + b"\n"
    if len(payload) > MAX_BYTES:
        raise ValueError("Device request is too large")
    # Never resubmit a request after a lost or mismatched response.
    try:
        if endpoint:
            raw = exchange(endpoint, "POST", body=payload)
        else:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(605)
                sock.connect(config["socket"])
                sock.sendall(payload)
                with sock.makefile("rb") as stream:
                    raw = stream.readline(MAX_BYTES + 1)
        response = json.loads(raw)
        if len(raw) > MAX_BYTES or not raw.endswith(b"\n") or any(
            response.get(key) != request[key] for key in (*identity, "request_id")
        ) or type(response.get("generation")) is not int or type(response.get("ok")) is not bool:
            raise ValueError("Incomplete or mismatched device response")
        if response["ok"] and (not isinstance(response.get("result"), dict)
                               or "data" not in response["result"]):
            raise ValueError("Device response has no result data")
        if not response["ok"] and not isinstance(response.get("error"), dict):
            raise ValueError("Device response has no error details")
    except Exception as exc:
        raise WireError(str(exc), "outcome_unknown", request_id) from exc
    if not response["ok"]:
        error = response["error"]
        raise WireError(error.get("message", str(error)), error.get("code"), request_id)
    data = response["result"]["data"]
    if endpoint and action == "read" and isinstance(data, dict) and data.get("file"):
        directory = Path(tempfile.mkdtemp(prefix="roboenv-images-"))
        for key in ("file", "depth_file"):
            if data.get(key):
                name = Path(data[key]).name
                path = f"/artifacts/{identity['session']}/{identity['generation']}/{quote(name)}"
                exchange(endpoint, "GET", path, destination=directory / name)
                data[key] = str(directory / name)
    return data


def handle(action, args):
    try:
        if action in {"fk", "ik"}:
            import yaml
            from complex.kinematics import Kinematics

            profile = yaml.safe_load(Path(__file__).with_name("profile.yaml").read_text())
            solver = Kinematics(profile["intrinsics"])
            if action == "ik" and args.get("seed") is None:
                args = {**args, "seed": solver.project(remote("read", {})["positions"])}
            return {"ok": True, **getattr(solver, action)(**args)}
        data = remote(action, args)
        return {**(data if isinstance(data, dict) else {"result": data}), "ok": True}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "read"
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(handle(action, args), allow_nan=False))
