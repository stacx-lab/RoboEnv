"""GPU selection rejects mismatches before a simulator can acquire devices."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.provisioning import gpu

DEVICES = [
    {"index": 0, "uuid": "GPU-00000000-0000-0000-0000-000000000000", "pci_bus_id": "0:01:00.0"},
    {"index": 2, "uuid": "GPU-22222222-2222-2222-2222-222222222222", "pci_bus_id": "0:02:00.0"},
]
VULKAN = [{"index": 3, "uuid": DEVICES[0]["uuid"]},
          {"index": 1, "uuid": DEVICES[1]["uuid"]}]


def test_physical_identity_not_equal_ordinals():
    selection = gpu.select_gpu(DEVICES, VULKAN, 2, {})
    assert (selection["index"], selection["vulkan_index"], selection["cuda_index"]) == (2, 1, 0)
    assert gpu.gpu_environment(selection)["CUDA_VISIBLE_DEVICES"] == DEVICES[1]["uuid"]
    assert gpu.select_gpu(DEVICES, VULKAN, None, {})["index"] == 0


@pytest.mark.parametrize("mask", ["2,0", DEVICES[1]["uuid"], "GPU-22222222"])
def test_inherited_visibility_and_order(mask):
    assert gpu.select_gpu(DEVICES, VULKAN, None, {"CUDA_VISIBLE_DEVICES": mask})["index"] == 2


@pytest.mark.parametrize("environment", [
    {"CUDA_VISIBLE_DEVICES": "2"}, {"NVIDIA_VISIBLE_DEVICES": "2"},
    {"NVIDIA_VISIBLE_DEVICES": "0,2", "CUDA_VISIBLE_DEVICES": "2"},
])
def test_explicit_selection_cannot_escape_inherited_restrictions(environment):
    with pytest.raises(ValueError, match="excluded"):
        gpu.select_gpu(DEVICES, VULKAN, 0, environment)


@pytest.mark.parametrize("mask", ["", "-1", "99", "2,2", "MIG-unknown"])
def test_invalid_visibility_fails_closed(mask):
    with pytest.raises(ValueError):
        gpu.select_gpu(DEVICES, VULKAN, None, {"CUDA_VISIBLE_DEVICES": mask})


@pytest.mark.parametrize("value", [-1, True, 0.5, "0"])
def test_invalid_selector_precedes_hardware_probe(monkeypatch, value):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected probe"))
    with pytest.raises(ValueError, match="nonnegative"):
        gpu.resolve_gpu(value)


def test_missing_vulkan_identity_is_rejected():
    with pytest.raises(ValueError, match="Vulkan"):
        gpu.select_gpu(DEVICES, VULKAN[:1], 2, {})


def test_probe_is_bounded_and_does_not_mutate_callers_environment(monkeypatch):
    def run(command, **kwargs):
        assert command[-2:] == ["-m", "roboenv.provisioning.gpu"]
        assert kwargs["timeout"] == 15
        return SimpleNamespace(stdout=json.dumps({"devices": DEVICES, "vulkan": VULKAN}))

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    before = dict(os.environ)
    assert gpu.resolve_gpu(2)["vulkan_index"] == 1
    assert dict(os.environ) == before
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("probe", 15)
    ))
    with pytest.raises(RuntimeError, match="before launch"):
        gpu.resolve_gpu()


def test_worker_uses_launch_selection_and_detects_drift(monkeypatch):
    selection = gpu.select_gpu(DEVICES, VULKAN, 2, {})
    for key, value in gpu.gpu_environment(selection).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(gpu, "resolve_gpu", lambda *a: pytest.fail("unexpected rediscovery"))
    assert gpu.configure_simple_gpu() == selection
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", DEVICES[0]["uuid"])
    with pytest.raises(ValueError, match="changed"):
        gpu.configure_simple_gpu()


@pytest.mark.parametrize("requested", [None, 2])
def test_launch_passes_selection_only_to_owned_process(tmp_path, monkeypatch, requested):
    from roboenv import environment

    root = Path(__file__).resolve().parents[1]
    selection = gpu.select_gpu(DEVICES, VULKAN, requested, {})
    monkeypatch.setattr(gpu, "resolve_gpu", lambda value: selection)
    monkeypatch.setattr("roboenv.provisioning.sources.status", lambda *a: {"ready": True})
    seen = []

    def capture(command, **kwargs):
        seen.append(kwargs["env"])
        raise OSError("stop before native startup")

    monkeypatch.setattr(environment.subprocess, "Popen", capture)
    before = dict(os.environ)
    with pytest.raises(OSError, match="before native startup"):
        environment.launch("simple/g1-xmove-pick", project=root, interpreter=sys.executable,
                           gpu=requested, run_dir=tmp_path)
    assert dict(os.environ) == before
    assert all(seen[0][key] == value for key, value in gpu.gpu_environment(selection).items())
    record = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert record["gpu"] == selection
    assert record["state"] == "failed"
    shutil.rmtree(seen[0]["ROBOENV_SOCKET_ROOT"])
