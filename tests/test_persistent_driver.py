"""Persistent portable drivers reuse resources without replaying uncertain commands."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from helpers import kinex_sources
from roboenv.devices.export import write_driver

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
    _, supervisor, _ = device_service
    directory = tmp_path / "controller"
    write_driver(directory, "other")
    (directory / "profile.yaml").write_text(yaml.safe_dump({
        "access": {"driver": {"actions": ["read"]}},
    }))
    requests = [{"id": str(i), "action": "read", "args": {}} for i in range(20)]
    proc = subprocess.run([sys.executable, str(directory / "driver.py"), "--serve"],
                          input="".join(json.dumps(r) + "\n" for r in requests),
                          env={**os.environ, "ROBOENV_ENDPOINT": supervisor.http_endpoint},
                          capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(replies) == len(requests) and all(r["ok"] for r in replies)
    assert all(r["budget"]["used"] == 0 for r in replies)
    assert not proc.stderr


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
            proc.stdin.write(json.dumps({'id': str(i), 'action': 'read', 'args': {}}) + '\n')
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            assert reply == {"id": str(i), "ok": True, "stopped": False,
                             "pid": proc.pid, "calls": i + 1}
        if ending == "unload":
            proc.stdin.write('{"id":"unload","action":"unload","args":{}}\n')
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
    library = ROOT.parent / "kinex/python"
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
    device_service, tmp_path,
):
    kinex, _ = kinex_sources()
    node = shutil.which("node")
    if not node or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("Requires built Kinex and Node >=22.19")
    _, supervisor, _ = device_service
    directory = tmp_path / ".kinex/devices/controller"
    write_driver(directory, "other")
    (directory / "profile.yaml").write_text(yaml.safe_dump({
        "name": "controller", "type": "other", "loaded": True,
        "meta": {"created_at": "2026-09-09T00:00:00Z", "updated_at": "2026-09-09T00:00:00Z"},
        "access": {"connection": {"kind": "url", "url": supervisor.http_endpoint,
                                  "device": "controller"},
                   "driver": {"interpreter": sys.executable,
                              "actions": ["load", "unload", "status", "read"]}},
    }))
    (directory / "connection.json").write_text(json.dumps({"endpoint": supervisor.http_endpoint}))
    driver = directory / "driver.py"
    driver.write_text(
        "import os\nfrom pathlib import Path\n"
        "with Path(__file__).with_name('boots').open('a') as log: "
        "log.write(str(os.getpid()) + '\\n')\n" + driver.read_text().replace(
            '        data = remote(action, args)',
            '        data = ({"ready": True} if action in {"load", "unload", "status"} '
            'else remote(action, args))',
        )
    )
    result = subprocess.run([node, str(ROOT / "tests/persistent_kinex.mjs"), str(kinex),
                             str(tmp_path), sys.executable], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
