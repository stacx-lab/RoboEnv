"""Lazy native creation and clock checks without substituting for GPU acceptance."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.robodojo import devices
from roboenv_runtime.adapters.robodojo.rig import camera_intrinsics, euler_pose


@pytest.mark.parametrize("budget_mode", ["practice", "official"])
def test_normal_worker_passes_only_typed_budget_arguments(monkeypatch, budget_mode):
    from roboenv_runtime.runtime import worker

    task = SimpleNamespace(
        id="robodojo.fixture", backend="robodojo", backend_kwargs={"task_name": "fixture"}
    )

    def construct(name, *, task_name):
        return name, task_name

    monkeypatch.setattr(worker, "make_adapter", construct)
    assert worker.build_adapter(task) == ("robodojo", "fixture")


@pytest.mark.parametrize("explicit", [None, "user-evaluation"])
def test_worker_output_isolation_and_explicit_run_id(monkeypatch, tmp_path, explicit):
    from roboenv_runtime.adapters.robodojo import backend

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--kit_args")

        def __init__(self, args):
            self.app = object()

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=Launcher))
    monkeypatch.setattr(backend, "ROBODOJO_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    ids = []
    for _ in range(2):
        monkeypatch.setattr(backend, "_APP", None)
        if explicit is None:
            monkeypatch.delenv("ROBODOJO_RUN_ID", raising=False)
        else:
            monkeypatch.setenv("ROBODOJO_RUN_ID", explicit)
        app = backend._boot_isaac(headless=False, enable_cameras=False)
        assert backend._boot_isaac() is app
        ids.append(backend.os.environ["ROBODOJO_RUN_ID"])
    if explicit is None:
        assert ids[0] != ids[1]
    else:
        assert ids == [explicit, explicit]


def lazy_environment(monkeypatch, native_dt=0.004):
    native = SimpleNamespace(
        sim=None,
        step_lim=100,
        dt=0.004,
        obs_manager=SimpleNamespace(collect_interval=10),
        sim_step=lambda: None,
        process_control_info=lambda control, index: [],
    )

    def construct(self, **kwargs):
        self._env = native
        self.seed = 0

    def reset(*, seed):
        native.sim = SimpleNamespace(physics_dt=native_dt)

    native.reset = reset
    monkeypatch.setattr(devices, "describe", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(devices.RoboDojoEnv, "__init__", construct)
    monkeypatch.setattr(devices.RoboDojoEnv, "_layout_id_for", lambda self, seed: seed)
    return native


def test_native_simulation_can_be_absent_until_first_reset(monkeypatch):
    native = lazy_environment(monkeypatch)
    world = devices.RoboDojoDevices("general_pickup")
    assert native.sim is None
    assert world.control_interval == pytest.approx(0.04)
    assert world.physics_steps == 0
    native.sim_step()
    assert world.physics_steps == 1


def test_reset_rejects_native_clock_drift_before_exposing_devices(monkeypatch):
    lazy_environment(monkeypatch, native_dt=0.008)
    world = devices.RoboDojoDevices("general_pickup")
    with pytest.raises(ValueError, match="Native physics timestep"):
        world.reset(seed=1)
    assert world.control is None


def test_rounded_source_mount_is_normalized_before_native_creation(monkeypatch):
    config = SimpleNamespace(
        eval_cfg=SimpleNamespace(config_name="arx_x5"),
        robot=SimpleNamespace(
            robots=[
                {"default_root_rot": [0.707, 0, 0, 0.707]},
            ]
        ),
    )
    monkeypatch.setattr(devices.RoboDojoEnv, "_assemble_config", lambda self, seed: config)
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world.layout_config = "arx_x5"
    actual = world._assemble_config(0).robot.robots[0]
    assert config.eval_cfg.config_name == "arx_x5"
    assert np.linalg.norm(actual["default_root_rot"]) == pytest.approx(1.0)
    assert actual["default_root_rot"] == pytest.approx([2**-0.5, 0, 0, 2**-0.5])
    assert actual["need_planner"] is False


def test_native_camera_uses_intrinsic_xyz_and_render_product_projection():
    # Two nonzero rotations distinguish the native convention from URDF rpy.
    np.testing.assert_allclose(
        euler_pose([1, 2, 3], [90, 90, 0]),
        [
            [0, 0, 1, 1],
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ],
        atol=1e-12,
    )
    native = camera_intrinsics(
        {
            "resolution": (640, 480),
            "focal_length": 10,
            "horizontal_aperture": 22.212,
            "vertical_aperture": 14.266,
        }
    )
    projection = native["color"]["projection"]
    assert projection["fx"] == pytest.approx(288.1325378417969, abs=1e-4)
    # camera_params reports projection[1, 1]=1.2005522598099891. The Camera
    # object's authored-aperture get_intrinsics_matrix instead reports 336.46.
    assert projection["fy"] == pytest.approx(288.1325423543974, abs=1e-4)


@pytest.mark.parametrize("invalid", [False, True])
def test_image_uses_native_tiled_capture_and_owns_its_pixels(invalid):
    camera = object()  # A native camera need not own an individual RGB annotator.
    buffer = np.full((3, 4, 4), 137, dtype=np.float32 if invalid else np.uint8)
    calls = []

    def capture(*, env_ids, cam_ids):
        calls.append((env_ids, cam_ids))
        return [{"rgb": [{"data": buffer}]}]

    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._env = SimpleNamespace(
        render=lambda: calls.append("render"),
        camera_manager=SimpleNamespace(
            cameras=[[object(), camera]], camera_names=[["cam_left_wrist", "cam_head"]]
        ),
        capture_manager=SimpleNamespace(step=capture),
    )
    world.device_facts = {"head_camera": {"intrinsics": {"color": {"width": 4, "height": 3}}}}
    if invalid:
        with pytest.raises(RuntimeError, match="valid RGB"):
            world.device_image("head_camera")
    else:
        rgb, depth = world.device_image("head_camera")
        assert depth is None
        buffer[:] = 0
        assert rgb.shape == (3, 4, 3) and np.all(rgb == 137)
    assert calls == ["render", ([0], [1])]


@pytest.mark.parametrize("mode", ["transient", "unavailable", "mismatch"])
def test_render_projection_waits_for_ready_parameters_without_hiding_mismatch(monkeypatch, mode):
    calls = []
    camera = SimpleNamespace(get_resolution=lambda: (640, 480))
    product = object()
    ready = {
        "renderProductResolution": [640, 480],
        "cameraModel": "pinhole",
        "cameraProjection": np.eye(4).reshape(-1),
    }
    first = {**ready, "renderProductResolution": [0, 0]}
    if mode == "mismatch":
        first["renderProductResolution"] = [800, 480]
    readings = iter([first, ready])

    def read():
        return first if mode != "transient" else next(readings)

    annotator = SimpleNamespace(
        attach=lambda selected: calls.append(("attach", selected)),
        detach=lambda selected: calls.append(("detach", selected)),
        get_data=read,
    )
    core = SimpleNamespace(
        AnnotatorRegistry=SimpleNamespace(get_annotator=lambda *args, **kwargs: annotator)
    )
    replicator = SimpleNamespace(core=core)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(replicator=replicator))
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._camera = lambda name: camera
    world._env = SimpleNamespace(
        camera_manager=SimpleNamespace(cameras=[[camera]]),
        capture_manager=SimpleNamespace(tiled_render_products=[product]),
        render=lambda: calls.append("render"),
    )
    if mode == "transient":
        actual = world._render_intrinsics("head_camera")
        np.testing.assert_allclose(actual, [[320, 0, 320], [0, 240, 240], [0, 0, 1]])
    elif mode == "unavailable":
        with pytest.raises(RuntimeError, match="not ready"):
            world._render_intrinsics("head_camera")
    else:
        with pytest.raises(ValueError, match="render_product=\\(800, 480\\)"):
            world._render_intrinsics("head_camera")
    assert calls[0] == ("attach", product) and calls[-1] == ("detach", product)
    assert calls.count("render") == {"transient": 4, "unavailable": 8, "mismatch": 3}[mode]
