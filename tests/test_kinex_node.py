"""Relocated Node bundles exercise the real bridge against a CPU protocol worker."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, load_tool
from roboenv.devices.export import write_driver
from roboenv.harbor.kinex import inspect_bundle


@pytest.fixture(scope="module")
def node_bundle(tmp_path_factory):
    kinex = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")).resolve()
    node, uv = shutil.which(os.environ.get("NODE", "node")), shutil.which("uv")
    if not node or not uv or not (kinex / "dist/node/src/index.js").is_file():
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
                 [str(node_bundle / "kinex/bin/kinex"), "--help"]):
        result = subprocess.run([node, *args], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
@pytest.mark.parametrize("outcome", ["complete", "failed", "timeout"])
def test_bridge_device_execution_and_cleanup(node_bundle, device_service, tmp_path, outcome):
    client, supervisor, _ = device_service
    workspace = tmp_path / "workspace"
    directory = workspace / ".kinex/devices/controller"
    write_driver(directory, "other")
    (directory / "profile.yaml").write_text(yaml.safe_dump({
        "name": "controller", "type": "other", "loaded": True,
        "meta": {"created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
        "access": {"connection": {"kind": "url", "url": supervisor.http_endpoint,
                                  "device": "controller"},
                   "driver": {"interpreter": sys.executable, "persistent": True,
                              "actions": ["read", "invoke"]}},
    }))
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
