"""Persistent portable drivers reuse resources without replaying uncertain commands."""

import importlib.util
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from roboenv.devices.export import export_devices, write_driver

ROOT = Path(__file__).resolve().parents[1]


def transport_module():
    spec = importlib.util.spec_from_file_location("wire", ROOT / "src/roboenv/assets/_roboenv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_http_reuse_identity_cache_idle_reconnect_and_shutdown(
    device_service, tmp_path, monkeypatch,
):
    _, supervisor, _ = device_service
    server = supervisor._http_server
    accepted = []
    original = server.get_request

    def accept():
        result = original()
        accepted.append(result[0])
        return result

    monkeypatch.setattr(server, "get_request", accept)
    profile = {"type": "other", "access": {"connection": {
        "kind": "url", "url": supervisor.http_endpoint, "device": "controller"}}}
    file = tmp_path / "profile.yaml"
    file.write_text(yaml.safe_dump(profile))
    module = transport_module()
    wire = module.Transport(tmp_path / "driver.py")
    try:
        first = wire.remote("read", {})
        assert first["budget"]["used"] == 0
        connection = wire.connection
        for _ in range(100):
            assert wire.remote("read", {})["budget"]["used"] == 0
        assert wire.connection is connection and len(accepted) == 1
        # Config changes apply to a new process, not an established episode.
        file.write_text("invalid: profile\n")
        assert wire.settings() == profile
        assert wire.remote("read", {})["budget"]["used"] == 0
        wire.last_exchange -= 6
        wire.remote("read", {})
        assert len(accepted) == 2 and wire.connection is not connection
        server.shutdown()
        server.server_close()
        assert not server.connections
        with pytest.raises(module.WireError):
            wire.remote("read", {})
        assert wire.connection is None
    finally:
        wire.close()


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_portable_dialog_uses_real_worker_and_stdout_only_contains_replies(
    device_service, tmp_path,
):
    client, supervisor, _ = device_service
    directory = tmp_path / "controller"
    write_driver(directory, "other")
    (directory / "profile.yaml").write_text(yaml.safe_dump({
        "access": {"driver": {"actions": ["read"]}},
    }))
    requests = ([{"action": action, "args": {"unexpected": True}}
                 for action in ("load", "status", "unload")]
                + [{"action": action, "args": {}}
                   for action in ("load", "status", *["read"] * 20, "unload")])
    proc = subprocess.run([sys.executable, str(directory / "driver.py"), "--serve"],
                          input="".join(json.dumps(r) + "\n" for r in requests),
                          env={**os.environ, "ROBOENV_ENDPOINT": supervisor.http_endpoint},
                          capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(replies) == len(requests)
    assert all(not r["ok"] for r in replies[:3])
    assert all(r["ok"] for r in replies[3:])
    assert all(r["loaded"] for r in replies[3:5])
    assert all(r["budget"]["used"] == 0 for r in replies[5:-1])
    assert replies[-1]["unloaded"]
    assert client.request("observe")["budget"]["used"] == 0
    assert not proc.stderr


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
@pytest.mark.parametrize("action,args,code,budget", [
    ("stop", {}, "unsupported_action", 0),
    ("read", {"bad_read": True}, "permission_denied", 0),
    ("invoke", {"tool_name": "push", "kwargs": {"dx": .3}, "fail_after": True},
     "device_failed", 50),
])
def test_portable_error_preserves_code_and_request_id_for_lookup(
    device_service, tmp_path, action, args, code, budget,
):
    client, supervisor, _ = device_service
    directory = tmp_path / "controller"
    write_driver(directory, "other")
    result = subprocess.run(
        [sys.executable, str(directory / "driver.py"), action, json.dumps(args)],
        env={**os.environ, "ROBOENV_ENDPOINT": supervisor.http_endpoint},
        capture_output=True, text=True, timeout=15,
    )
    reply = json.loads(result.stdout)
    assert reply["ok"] is False and reply["code"] == code
    request_id = reply["request_id"]
    assert request_id.startswith("driver-")
    operation = client.operation(request_id)
    assert operation["state"] in {"failed", "indeterminate"}
    assert operation["response"]["error"]["code"] == code
    # A failed invocation may already have moved: the driver must not retry it.
    assert client.request("observe")["budget"]["used"] == budget


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_portable_lost_response_preserves_operation_id_without_retry(
    device_service, tmp_path, monkeypatch,
):
    client, supervisor, _ = device_service
    directory = tmp_path / "controller"
    write_driver(directory, "other")
    monkeypatch.setenv("ROBOENV_ENDPOINT", supervisor.http_endpoint)
    driver = runpy.run_path(str(directory / "driver.py"))
    transport = driver["transport"]
    exchange = transport.exchange
    sent = []

    def lose_response(endpoint, method, path="/", body=None):
        result = exchange(endpoint, method, path, body)
        if method == "POST":
            sent.append(json.loads(body))
            raise OSError("Response lost after the server accepted the motion")
        return result

    monkeypatch.setattr(transport, "exchange", lose_response)
    reply = driver["handle"]("invoke", {"tool_name": "push", "kwargs": {"dx": .3}})
    assert reply["ok"] is False and reply["code"] == "outcome_unknown"
    assert len(sent) == 1 and reply["request_id"] == sent[0]["request_id"]
    assert client.operation(reply["request_id"])["state"] == "succeeded"
    assert client.request("observe")["budget"]["used"] == 50
    assert transport.connection is None


DIALOG = '''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _roboenv import Transport, main
transport = Transport(__file__)
transport.profile = {"access": {"driver": {"actions": ["read", "write", "stop", "unload"]}}}
calls = 0
def handle(action, args):
    global calls
    print("SDK log")
    os.write(1, b"native SDK log\\n")
    with open("calls", "a") as output:
        output.write(action + "\\n")
    if action == "stop" and Path("hang").exists():
        time.sleep(10)
    calls += 1
    transport.used = True
    transport.unloaded = action == "unload"
    return {"ok": True, "stopped": action == "stop", "pid": os.getpid(), "calls": calls}
main(handle, transport)
'''


@pytest.mark.parametrize("ending", ["eof", "sigterm", "unload", "hang", "malformed", "one-shot"])
def test_dialog_reuses_process_and_cleans_up_once(tmp_path, ending):
    write_driver(tmp_path, "other")
    driver = tmp_path / "driver.py"
    driver.write_text(DIALOG)
    if ending == "hang":
        (tmp_path / "hang").touch()
    if ending == "one-shot":
        result = subprocess.run([sys.executable, str(driver), "write", "{}"], cwd=tmp_path,
                                capture_output=True, text=True, timeout=2)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["ok"]
        assert (tmp_path / "calls").read_text() == "write\n"
        return
    proc = subprocess.Popen([sys.executable, str(driver), "--serve"], cwd=tmp_path,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        for i in range(1000 if ending == "eof" else 2):
            proc.stdin.write('{"action":"read","args":{}}\n')
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            assert reply == {"ok": True, "stopped": False, "pid": proc.pid, "calls": i + 1}
        if ending == "unload":
            proc.stdin.write('{"action":"unload","args":{}}\n')
            proc.stdin.flush()
            assert json.loads(proc.stdout.readline())["ok"]
        if ending == "malformed":
            proc.stdin.write('{broken\n')
            proc.stdin.flush()
            assert not json.loads(proc.stdout.readline())["ok"]
        started = time.monotonic()
        if ending == "sigterm":
            proc.send_signal(signal.SIGTERM)
        proc.stdin.close()
        assert proc.wait(timeout=2) == 0
        assert time.monotonic() - started < 1.5
        calls = (tmp_path / "calls").read_text().splitlines()
        assert calls.count("stop") == (0 if ending == "unload" else 1)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


def test_local_kinematics_is_cached_and_never_connects_for_cleanup(tmp_path, monkeypatch):
    library = ROOT.parent / "kinex/packages/core/assets/python"
    if not library.is_dir():
        pytest.skip("Requires adjacent Kinex Python library")
    monkeypatch.syspath_prepend(str(library))
    source = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices/arm"
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, tmp_path / file.name)
    shutil.copy2(source.parent / "_roboenv.py", tmp_path.parent / "_roboenv.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location("arm_cache", tmp_path / "driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = module.model()
    assert module.model() is arm
    joints = [module.settings()["intrinsics"]["home"][j["name"]] for j in arm.joints]
    assert module.handle("fk", {"joints": joints})["ok"]
    assert not module.transport.used


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_artifacts_reuse_connection_and_truncation_discards_partial_file(
    device_service, tmp_path, monkeypatch,
):
    _, supervisor, _ = device_service
    module = transport_module()
    wire = module.Transport(tmp_path / "driver.py", device="controller")
    monkeypatch.setenv("ROBOENV_ENDPOINT", supervisor.http_endpoint)
    directory = supervisor.config.run_dir / "captures"
    directory.mkdir(exist_ok=True)
    name = "a" * 32 + ".png"
    (directory / name).write_bytes(b"test image")
    route = f"/artifacts/{supervisor.config.session}/{supervisor.config.generation}/{name}"
    try:
        wire.remote("read", {})
        connection = wire.connection
        wire.exchange(wire.endpoint, "GET", route, destination=tmp_path / name)
        assert (tmp_path / name).read_bytes() == b"test image"
        assert wire.connection is connection

        def truncated(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "20")
            handler.end_headers()
            handler.wfile.write(b"partial")
            handler.close_connection = True

        monkeypatch.setattr(supervisor._http_server.RequestHandlerClass, "do_GET", truncated)
        with pytest.raises(ValueError, match="Incomplete artifact"):
            wire.exchange(wire.endpoint, "GET", route, destination=tmp_path / "partial.png")
        assert not (tmp_path / "partial.png").exists()
        assert wire.connection is None
    finally:
        wire.close()


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_live_identity_change_is_not_followed(device_service, tmp_path, monkeypatch):
    _, supervisor, _ = device_service
    monkeypatch.setenv("ROBOENV_ENDPOINT", supervisor.http_endpoint)
    module = transport_module()
    wire = module.Transport(tmp_path / "driver.py", device="controller")
    try:
        wire.remote("read", {})
        generation = wire.identity["generation"]
        monkeypatch.setattr(supervisor.config, "generation", generation + 1)
        with pytest.raises(module.WireError):
            wire.remote("invoke", {"tool_name": "push", "kwargs": {"dx": .1}})
        assert wire.identity["generation"] == generation
    finally:
        wire.close()


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_current_kinex_reuses_driver_across_tools_and_reloads(
    device_service, tmp_path, monkeypatch,
):
    kinex = ROOT.parent / "kinex"
    node = shutil.which(os.environ.get("NODE", "node"))
    if not node or not (kinex / "packages/core/dist/index.js").exists():
        pytest.skip("Requires Node and built adjacent Kinex core")
    client, supervisor, _ = device_service
    directory = tmp_path / ".kinex/devices/controller"
    export_devices(client.devices()["devices"], directory.parent,
                   {"endpoint": supervisor.http_endpoint})
    profile_file = directory / "profile.yaml"
    profile = yaml.safe_load(profile_file.read_text())
    profile["access"]["driver"]["interpreter"] = sys.executable
    profile_file.write_text(yaml.safe_dump(profile))
    accepted = []
    original = supervisor._http_server.get_request

    def accept():
        value = original()
        accepted.append(value[0])
        return value

    monkeypatch.setattr(supervisor._http_server, "get_request", accept)
    result = subprocess.run([node, str(ROOT / "tests/persistent_kinex.ts"), str(kinex),
                             str(tmp_path), sys.executable], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(accepted) == 2  # first activation and explicit reload
