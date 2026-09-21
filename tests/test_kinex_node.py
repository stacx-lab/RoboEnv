"""Relocated Node bundles exercise the real bridge against a CPU protocol worker."""

import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, load_tool
from roboenv.devices.export import export_devices
from roboenv.harbor.kinex import inspect_bundle


@pytest.fixture(scope="module")
def node_bundle(tmp_path_factory):
    kinex = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")).resolve()
    node, uv = shutil.which(os.environ.get("NODE", "node")), shutil.which("uv")
    if not node or not uv or not (kinex / "dist/cli.js").is_file():
        pytest.skip("Requires Node, uv and a built Kinex companion")
    root = tmp_path_factory.mktemp("kinex-node-bundle")
    archive = root / "agent.tar.gz"
    report = load_tool("package_kinex_agent").package(kinex, node, uv, archive)
    assert inspect_bundle(archive)["kinex_sha256"] == report["kinex_sha256"]
    destination = root / "relocated"
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    assert not (destination / "kinex/node_modules/@kinex/core").is_symlink()
    return destination


def test_relocated_bridge_and_controller_checks(node_bundle, tmp_path):
    bridge = node_bundle / "kinex/integrations/roboenv"
    node = str(node_bundle / "bin/node")
    env = {**os.environ, "KINEX_DATA_DIR": str(tmp_path / "data"),
           "KINEX_CONFIG_DIR": str(tmp_path / "config")}
    for args in ([str(bridge / "run.ts"), "--help"],
                 ["--test", str(bridge / "controller.test.ts"),
                  str(bridge / "provider-evidence.test.ts")],
                 [str(node_bundle / "kinex/bin/kinex.mjs"), "--help"]):
        result = subprocess.run([node, *args], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
@pytest.mark.parametrize("outcome", ["complete", "failed", "timeout"])
def test_bridge_device_execution_and_cleanup(node_bundle, device_service, tmp_path, outcome):
    client, supervisor, _ = device_service
    workspace = tmp_path / "workspace"
    directory = workspace / ".kinex/devices/controller"
    export_devices(client.devices()["devices"], directory.parent,
                   {"endpoint": supervisor.http_endpoint})
    profile_file = directory / "profile.yaml"
    profile = yaml.safe_load(profile_file.read_text())
    profile["access"]["driver"]["interpreter"] = sys.executable
    profile_file.write_text(yaml.safe_dump(profile))
    (workspace / ".kinex/workspace.yaml").write_text(yaml.safe_dump({
        "identity": {"id": "cpu", "name": "CPU", "platform": "sim", "kind": "other"},
        "meta": {"created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
    }))
    tool = workspace / ".kinex/tool/probe"
    tool.mkdir(parents=True)
    (tool / "tool.yaml").write_text(yaml.safe_dump({
        "name": "probe", "kind": "controller", "description": "CPU protocol probe",
        "interpreter": sys.executable, "timeout_s": 30,
        "parameters": {"type": "object", "properties": {}},
    }))
    (tool / "tool.py").write_text(
        "from kinex import device, finish\n"
        "assert device('controller')['data']['budget']['used'] == 0\n"
        "device('controller', 'invoke', tool_name='push', kwargs={'dx': 0.1})\n"
        + ("import time; time.sleep(30)\n" if outcome == "timeout"
           else "import time; time.sleep(0.5)\n")
        + f"finish({{'completed': {outcome == 'complete'}}})\n"
    )
    profile = (directory / "profile.yaml").read_bytes()
    config = tmp_path / "input.json"
    receipt = tmp_path / "execution.json"
    config.write_text(json.dumps({
        "mode": "no-model", "workspace": str(workspace), "output": str(receipt),
        "tool": "probe", "args": {}, "timeout_s": 3 if outcome == "timeout" else 15,
        "recording": {"controller": {"rate": 5}}, "require_completion": True,
    }))
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    env.update(KINEX_DATA_DIR=str(tmp_path / "data"), KINEX_CONFIG_DIR=str(tmp_path / "config"))
    result = subprocess.run([str(node_bundle / "bin/node"),
                             str(node_bundle / "kinex/integrations/roboenv/run.ts"), str(config)],
                            cwd=workspace, env=env, capture_output=True, text=True, timeout=25)
    assert receipt.exists(), result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert (result.returncode == 0) == (outcome == "complete"), result.stdout + result.stderr
    assert report["passed"] == (outcome == "complete")
    assert report["controller_completed"] == (outcome == "complete")
    assert any(event["action"] == "invoke" for event in report["events"])
    assert client.request("observe")["budget"]["used"] == 50
    assert (directory / "profile.yaml").read_bytes() == profile
    recordings = list((workspace / ".kinex/recordings").glob("*/recording.json"))
    assert len(recordings) == 1
    recording = json.loads(recordings[0].read_text())
    assert recording["status"] == "stopped"
    assert recording["devices"]["controller"]["stream"]["rows"] > 0
    assert not (workspace / ".kinex/python").exists()  # Explicit interpreters remain independent.


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_development_uses_pi_session_and_checks_provider_images(node_bundle, device_service,
                                                               tmp_path, request):
    client, supervisor, _ = device_service
    workspace = tmp_path / "workspace"
    export_devices(client.devices()["devices"], workspace / ".kinex/devices",
                   {"endpoint": supervisor.http_endpoint})
    profile_file = workspace / ".kinex/devices/controller/profile.yaml"
    profile = yaml.safe_load(profile_file.read_text())
    profile["access"]["driver"]["interpreter"] = sys.executable
    profile_file.write_text(yaml.safe_dump(profile))
    (workspace / "frame.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGMQCaggCTGMahjVMHw1AACSwNwBzrAu+gAAAABJRU5ErkJggg=="))
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            tool = len(requests) == 1
            item = ({"id": "fc_1", "call_id": "call_1", "type": "function_call", "name": "read",
                     "arguments": json.dumps({"path": "frame.png"}), "status": "completed"}
                    if tool else {"id": "msg_2", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": "done",
                                                             "annotations": []}]})
            rows = [{"type": "response.created", "response": {
                        "id": f"resp_{len(requests)}", "status": "in_progress", "output": []}},
                    {"type": "response.output_item.added", "output_index": 0,
                     "item": {**item, "status": "in_progress", "content": []}}]
            if tool:
                rows.append({"type": "response.function_call_arguments.delta", "output_index": 0,
                             "delta": item["arguments"]})
            else:
                rows.extend([
                    {"type": "response.content_part.added", "output_index": 0, "content_index": 0,
                     "part": {"type": "output_text", "text": "", "annotations": []}},
                    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                     "delta": "done"}])
            rows.extend([
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"id": f"resp_{len(requests)}",
                    "status": "completed", "output": [item], "usage": {
                        "input_tokens": 10, "output_tokens": 3, "total_tokens": 13}}}])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write("".join("data: " + json.dumps(row) + "\n\n" for row in rows).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.server_close)
    request.addfinalizer(server.shutdown)
    base = f"http://127.0.0.1:{server.server_port}/v1"
    config_dir = tmp_path / "config/agent"
    config_dir.mkdir(parents=True)
    (config_dir / "models.json").write_text(json.dumps({"providers": {"openai": {
        "baseUrl": base, "api": "openai-responses", "apiKey": "offline",
        "models": [{"id": "probe", "reasoning": True, "input": ["text", "image"],
                    "contextWindow": 64000, "maxTokens": 1024}]}}}))
    config = tmp_path / "input.json"
    receipt = tmp_path / "execution.json"
    config.write_text(json.dumps({"mode": "develop", "workspace": str(workspace),
        "output": str(receipt), "timeout_s": 15, "recording": {}, "prompt": "Read frame.png",
        "model": {"provider": "openai", "id": "probe", "effort": "low"}}))
    env = {**os.environ, "KINEX_CONFIG_DIR": str(config_dir.parent),
           "KINEX_DATA_DIR": str(tmp_path / "data"), "OPENAI_BASE_URL": base}
    result = subprocess.run([str(node_bundle / "bin/node"),
        str(node_bundle / "kinex/integrations/roboenv/run.ts"), str(config)],
        cwd=workspace, env=env, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert report["passed"] and report["mode"] == "develop"
    assert len(report["provider_requests"]) == 2
    assert report["provider_requests"][1]["images"][0]["bytes"] > 0
    assert Path(report["native_session"]).is_file()
    assert all(row["model"] == "probe" and row["reasoning"]["effort"] == "low"
               for row in requests)
    assert client.request("observe")["budget"]["used"] == 0
