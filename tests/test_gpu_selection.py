"""Shared GPU identity and startup policy, without native GPU dependencies."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.provisioning import gpu
from roboenv.provisioning.acceptance import acceptance_environment


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))


DEVICE = {"uuid": "GPU-22222222-2222-2222-2222-222222222222",
          "pci_bus_id": "0000:02:00.0", "cuda_index": 0}
SELECTION = {**DEVICE, "vulkan_index": 3, "egl_index": 5}


def test_match_uses_identity_not_ordinals_and_prefers_uuid():
    candidates = [{"index": 3, "uuid": DEVICE["uuid"]},
                  {"index": 7, "pci_bus_id": DEVICE["pci_bus_id"]}]
    assert gpu.match_graphics(DEVICE, candidates, "egl") == 3
    assert gpu.match_graphics(DEVICE, candidates[1:], "egl") == 7
    for invalid in ([], [candidates[0], candidates[0]]):
        with pytest.raises(ValueError, match="no unique"):
            gpu.match_graphics(DEVICE, invalid, "egl")


@pytest.mark.parametrize("mask", [None, "6", "6,2", DEVICE["uuid"], "", "-1"])
@pytest.mark.parametrize("container_mask", [None, "7"])
def test_probe_inherits_cuda_mask_without_reinterpreting_container_indices(
    monkeypatch, mask, container_mask,
):
    for key, value in (("CUDA_VISIBLE_DEVICES", mask),
                       ("NVIDIA_VISIBLE_DEVICES", container_mask)):
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    before = dict(os.environ)

    def run(command, **kwargs):
        assert command[-3:] == ["roboenv.provisioning.gpu", "vulkan", "egl"]
        assert kwargs["timeout"] == 15
        assert "env" not in kwargs  # Driver sees precisely the caller's mask.
        assert dict(os.environ) == before
        if mask in {"", "-1"}:
            raise subprocess.CalledProcessError(1, command, stderr="No GPU is visible to CUDA")
        return SimpleNamespace(stdout=json.dumps(SELECTION))

    monkeypatch.setattr(subprocess, "run", run)
    if mask in {"", "-1"}:
        with pytest.raises(RuntimeError, match="No GPU is visible"):
            gpu.resolve_gpu(("vulkan", "egl"))
    else:
        assert gpu.resolve_gpu(("vulkan", "egl")) == SELECTION
    assert dict(os.environ) == before


def test_probe_is_bounded_and_cpu_paths_do_not_probe(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 15)

    monkeypatch.setattr(subprocess, "run", run)
    assert gpu.resolve_gpu(()) is None
    with pytest.raises(RuntimeError, match="before launch"):
        gpu.resolve_gpu(("vulkan",))


@pytest.mark.parametrize("backend,options,expected", [
    ("fakeshell", {}, ()), ("libero", {"enable_render": False}, ()),
    ("robocasa", {"enable_render": False}, ()),
    ("libero", {}, ("egl",)), ("robocasa", {}, ("egl",)),
    ("abc", {"camera_backend": "blender"}, ("egl",)),
    ("simple", {}, ("vulkan", "egl")), ("simple", {"sim_mode": "mujoco"}, ("egl",)),
    ("robotwin", {}, ("vulkan",)), ("robodojo", {}, ("vulkan",)),
    ("behavior", {}, ("vulkan",)), ("isaaclab", {}, ("vulkan",)),
])
def test_backend_graphics_requirements(backend, options, expected):
    assert gpu.graphics_apis(backend, options) == expected


def test_worker_uses_launch_selection_and_detects_drift(monkeypatch):
    for key, value in gpu.gpu_environment(SELECTION).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "99")
    monkeypatch.setattr(gpu, "resolve_gpu", lambda *a: pytest.fail("unexpected rediscovery"))
    assert gpu.configure_gpu("simple") == SELECTION
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"
    assert os.environ["SIMPLE_ISAAC_ACTIVE_GPU"] == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="changed"):
        gpu.configure_gpu("simple")


def test_direct_native_startup_resolves_current_mask(monkeypatch):
    monkeypatch.delenv("ROBOENV_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,2")
    for name in gpu.gpu_environment(SELECTION):
        if name != "CUDA_VISIBLE_DEVICES":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MUJOCO_GL", "egl")

    def resolve(apis):
        assert apis == ("egl",)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,2"
        return SELECTION

    monkeypatch.setattr(gpu, "resolve_gpu", resolve)
    assert gpu.configure_gpu("abc") == SELECTION
    assert os.environ["CUDA_VISIBLE_DEVICES"] == DEVICE["uuid"]


@pytest.mark.parametrize("task", ["simple/xmove-pick", "libero/libero-spatial-01"])
@pytest.mark.parametrize("acceptance", [None, "NO"])
def test_launch_passes_selection_and_acceptance_only_to_owned_process(
    tmp_path, monkeypatch, task, acceptance,
):
    from roboenv import environment

    monkeypatch.setattr(gpu, "resolve_gpu", lambda apis: SELECTION)
    monkeypatch.setattr("roboenv.provisioning.sources.status", lambda *a: {"ready": True})
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.delenv("ROBOENV_BEHAVIOR_ACCEPT_LICENSE", raising=False)
    monkeypatch.setenv("ROBOENV_ASSET_CACHE", str(tmp_path / "assets"))
    if acceptance is not None:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", acceptance)
        monkeypatch.setenv("ROBOENV_BEHAVIOR_ACCEPT_LICENSE", acceptance)
    seen = []

    def capture(command, **kwargs):
        seen.append(kwargs["env"])
        raise OSError("stop before native startup")

    monkeypatch.setattr(environment.subprocess, "Popen", capture)
    before = dict(os.environ)
    with pytest.raises(OSError, match="before native startup"):
        environment.launch(task, project=Path(__file__).resolve().parents[1],
                           interpreter=sys.executable, run_dir=tmp_path)
    assert dict(os.environ) == before
    assert all(seen[0][key] == value for key, value in gpu.gpu_environment(SELECTION).items())
    expected = "YES" if acceptance is None else acceptance
    assert seen[0]["OMNI_KIT_ACCEPT_EULA"] == seen[0]["ROBOENV_BEHAVIOR_ACCEPT_LICENSE"] == expected
    assert seen[0]["ROBOENV_ASSET_CACHE"] == str(tmp_path / "assets")
    record = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert record["gpu"] == SELECTION
    assert record["state"] == "failed"
    shutil.rmtree(seen[0]["ROBOENV_SOCKET_ROOT"])


@pytest.mark.parametrize("value", [None, "YES", "NO", ""])
def test_acceptance_defaults_preserve_opt_outs(value):
    source = {} if value is None else {"OMNI_KIT_ACCEPT_EULA": value,
                                      "ROBOENV_BEHAVIOR_ACCEPT_LICENSE": value}
    before = dict(source)
    result = acceptance_environment(source)
    expected = "YES" if value is None else value
    assert result["OMNI_KIT_ACCEPT_EULA"] == result["ROBOENV_BEHAVIOR_ACCEPT_LICENSE"] == expected
    assert source == before


def test_isaaclab_applies_separate_indices_after_native_resolution(monkeypatch):
    from roboenv_runtime.adapters import gpu as native

    monkeypatch.setattr(native, "configure_gpu", lambda backend: SELECTION)
    for key in ("OMNI_KIT_ACCEPT_EULA", "ROBOENV_BEHAVIOR_ACCEPT_LICENSE"):
        monkeypatch.delenv(key, raising=False)
    captured = []

    class Launcher:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "cuda:0"
            self._config_resolution(kwargs)
            captured.append(self._sim_app_config)
            self.app = object()

        def _config_resolution(self, kwargs):
            self._sim_app_config = {"active_gpu": 0, "physics_gpu": 0, "multi_gpu": True}

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=Launcher))
    assert native.launch_isaaclab("robodojo", headless=True) is not None
    assert captured == [{"active_gpu": 3, "physics_gpu": 0, "multi_gpu": False, "max_gpu_count": 1}]
    assert os.environ["OMNI_KIT_ACCEPT_EULA"] == "YES"


@pytest.mark.parametrize("mig", [False, True])
def test_cuda_probe_queries_logical_zero_without_a_context(monkeypatch, mig):
    import ctypes
    import uuid

    calls = []

    class Driver:
        def cuInit(self, flags):
            calls.append(("init", flags))
            return 0

        def cuDeviceGetCount(self, output):
            output._obj.value = 2
            return 0

        def cuDeviceGet(self, output, ordinal):
            calls.append(("device", ordinal))
            output._obj.value = 0
            return 0

        def cuDeviceGetUuid_v2(self, output, device):
            ctypes.memmove(output, uuid.UUID(DEVICE["uuid"][4:]).bytes, 16)
            return 0

        def cuDeviceGetUuid(self, output, device):
            identity = uuid.UUID(int=0) if mig else uuid.UUID(DEVICE["uuid"][4:])
            ctypes.memmove(output, identity.bytes, 16)
            return 0

        def cuDeviceGetPCIBusId(self, output, length, device):
            output.value = DEVICE["pci_bus_id"].encode()
            return 0

    monkeypatch.setattr(gpu.c, "CDLL", lambda name: Driver())
    if mig:
        with pytest.raises(ValueError, match="MIG instances"):
            gpu._cuda_device()
    else:
        assert gpu._cuda_device() == DEVICE
    assert calls == [("init", 0), ("device", 0)]


def test_single_reachable_gpu_travels_as_index_zero(monkeypatch):
    # One EGL device: the process reaches one GPU, index 0 names it, and robosuite's textual check
    # of MUJOCO_EGL_DEVICE_ID against the mask holds. Several devices, or a selection recorded
    # before the count existed: the UUID names it. The worker's drift check follows the same rule.
    single = {**SELECTION, "egl_devices": 1, "egl_index": 0}
    assert gpu.gpu_environment(single)["CUDA_VISIBLE_DEVICES"] == "0"
    assert gpu.gpu_environment(single)["MUJOCO_EGL_DEVICE_ID"] == "0"
    several = gpu.gpu_environment({**SELECTION, "egl_devices": 2})
    assert several["CUDA_VISIBLE_DEVICES"] == DEVICE["uuid"]
    assert gpu.gpu_environment(SELECTION)["CUDA_VISIBLE_DEVICES"] == DEVICE["uuid"]
    for key, value in gpu.gpu_environment(single).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(gpu, "resolve_gpu", lambda *a: pytest.fail("unexpected rediscovery"))
    assert gpu.configure_gpu("libero") == single


def test_probe_loads_only_required_graphics_library(monkeypatch):
    monkeypatch.setattr(gpu, "_cuda_device", lambda: dict(DEVICE))
    monkeypatch.setattr(gpu, "_egl_devices", lambda: [{"index": 5, "uuid": DEVICE["uuid"]}])
    monkeypatch.setattr(gpu, "_vulkan_devices", lambda: pytest.fail("unexpected Vulkan load"))
    assert gpu._probe(("egl",)) == {**DEVICE, "egl_index": 5, "egl_devices": 1}


@pytest.mark.parametrize("selected", [None, "cuda:0"])
def test_robotwin_selects_the_device_before_creating_either_renderer(monkeypatch, selected):
    import ast

    source = Path(__file__).resolve().parents[1] / "third_party/RoboTwin/envs/_base_task.py"
    if not source.is_file():
        pytest.skip("Requires managed RoboTwin source; no simulator imports")
    method = next(node for node in ast.walk(ast.parse(source.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "setup_scene")
    calls = []

    class SceneReached(Exception):
        pass

    def scene(*args):
        raise SceneReached

    render = SimpleNamespace(**{
        name: lambda *a, **k: None for name in (
            "set_global_config", "set_camera_shader_dir", "set_ray_tracing_samples_per_pixel",
            "set_ray_tracing_path_depth", "set_ray_tracing_denoiser",
        )
    })
    render.SapienRenderer = lambda device: calls.append(("renderer", device))
    render.RenderSystem = lambda device: calls.append(("scene", device))
    sapien = SimpleNamespace(
        Engine=lambda: SimpleNamespace(set_renderer=lambda renderer: None, create_scene=scene),
        Device=lambda device: device,
        SapienRenderer=lambda: calls.append(("legacy", None)),
        render=render,
        physx=SimpleNamespace(set_scene_config=lambda config: None, PhysxCpuSystem=lambda: None),
        Scene=scene, SceneConfig=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sapien.render", render)
    monkeypatch.delenv("ROBOTWIN_RENDER_DEVICE", raising=False)
    if selected is not None:
        monkeypatch.setenv("ROBOTWIN_RENDER_DEVICE", selected)
    namespace = {"sapien": sapien, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(SceneReached):
        namespace["setup_scene"](SimpleNamespace())
    assert calls == ([("legacy", None)] if selected is None else
                     [("renderer", selected), ("scene", selected)])
