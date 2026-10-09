"""Lazy native creation and clock checks without substituting for GPU acceptance."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.robodojo import devices
from roboenv_runtime.adapters.robodojo.rig import camera_intrinsics, euler_pose


@pytest.fixture
def owner_world():
    reward = SimpleNamespace(**{key: [] for key in (
        "score_achieved", "score_completed_count", "final_score_achieved",
        "final_score_completed_count", "score_meta", "score_trigger_meta", "final_score_meta",
    )}, func_parser=SimpleNamespace(
        pre_state={}, robot_origin_endpose={}, joint_ratio_transition_state={},
    ))
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._env = SimpleNamespace(
        sim=SimpleNamespace(scene=SimpleNamespace(get_state=lambda: {})),
        scene_manager=SimpleNamespace(
            _rigid_and_dynamic_objects=[{}], _articulation_objects=[{}],
            _geometry_objects=[{}], _garment_objects=[{}], _fluid_objects=[{}],
        ),
        reward_manager=reward,
        robot_manager=SimpleNamespace(control_manager=SimpleNamespace(prev_control={})),
        take_action_cnt=[0], end_flag=[False], success=[False], completed=[False],
    )
    return world


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_owner_state_reads_native_garment_path_without_initializing_tensor_view(
    device, owner_world,
):
    points = np.zeros((3, 3))
    velocity = np.ones((3, 3))

    def unavailable():
        raise AssertionError("Used a garment path the native environment does not initialize")

    garment = SimpleNamespace(
        _device=device,
        get_world_pose=lambda: ([0, 0, 0], [1, 0, 0, 0]),
        sample_mesh_vertices=(
            (lambda: (points, None, None, None)) if device == "cpu" else unavailable
        ),
        _prim=SimpleNamespace(GetAttribute=lambda name: SimpleNamespace(Get=lambda: velocity)),
        _cloth_prim_view=SimpleNamespace(
            is_physics_handle_valid=lambda: device != "cpu",
            get_world_positions=unavailable if device == "cpu" else lambda: points,
            get_velocities=unavailable if device == "cpu" else lambda: velocity,
        ),
    )
    world = owner_world
    world._env.scene_manager._garment_objects = [{"shirt": garment}]
    first = world.owner_state()
    assert first["state"]["task_objects"]["garment"]["shirt"]["points"] == points.tolist()
    points[0, 0] = 0.25
    assert world.owner_state()["sha256"] != first["sha256"]


def test_owner_state_reads_available_rigid_object_handles(owner_world):
    class Dynamic:
        def get_world_pose(self):
            return ([0, 0, 0], [1, 0, 0, 0])

    registry = owner_world._env.scene_manager._rigid_and_dynamic_objects[0]
    registry["conveyor"] = Dynamic()
    state = owner_world.owner_state()["state"]["task_objects"]["rigid"]["conveyor"]
    assert state == {"pose": [[0, 0, 0], [1, 0, 0, 0]]}
    registry["rigid"] = SimpleNamespace(get_world_pose=lambda: ([0, 0, 0], [1, 0, 0, 0]))
    state = owner_world.owner_state()["state"]["task_objects"]["rigid"]["rigid"]
    assert set(state) == {"pose"}
    registry["rigid"].get_linear_velocity = lambda: [1, 2, 3]
    registry["rigid"].get_angular_velocity = lambda: [4, 5, 6]
    state = owner_world.owner_state()["state"]["task_objects"]["rigid"]["rigid"]
    assert state["linear_velocity"] == [1, 2, 3] and state["angular_velocity"] == [4, 5, 6]


@pytest.mark.parametrize("requested,expected", [(None, 100), (50, 50), (7500, 7500)])
def test_explicit_frame_budget_reaches_native_eval_env(requested, expected):
    env = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    env.native_step_limit = 100
    env.requested_frame_limit = requested
    env._env = SimpleNamespace(step_lim=100, end_flag=[False])
    env.control = SimpleNamespace(frame=expected - 1)
    env._configure_step_limit()
    assert env.frame_limit == env._env.step_lim == expected
    assert not env.exhausted()
    env.control.frame = expected
    assert env.exhausted()
    env.control.frame = 0
    env._env.end_flag[0] = True
    assert env.exhausted()


@pytest.mark.parametrize("budget_mode", ["practice", "official"])
def test_normal_worker_passes_only_typed_budget_arguments(monkeypatch, budget_mode):
    from roboenv_runtime.runtime import worker

    task = SimpleNamespace(
        id="robodojo/fixture", backend="robodojo", backend_kwargs={"task_name": "fixture"},
        stop_on_success=False,
    )

    def construct(name, *, task_name, stop_on_success):
        assert stop_on_success is False
        return name, task_name

    monkeypatch.setattr(worker, "make_adapter", construct)
    assert worker.build_adapter(task) == ("robodojo", "fixture")


@pytest.mark.parametrize("stop_on_success", [False, True])
def test_adapter_passes_success_policy_and_reports_native_end(monkeypatch, stop_on_success):
    from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter

    calls = []
    native = SimpleNamespace(end_flag=[False])

    def construct(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(_env=native)

    monkeypatch.setattr(devices, "RoboDojoDevices", construct)
    adapter = RoboDojoAdapter(stop_on_success=stop_on_success)
    assert calls == [{"stop_on_success": stop_on_success}]
    assert not adapter.episode_done
    native.end_flag[0] = True
    assert adapter.episode_done


def test_completion_read_uses_evaluated_success_without_ending_episode():
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    native = world._env = SimpleNamespace(
        success=[True], completed=[False], end_flag=[False],
    )
    assert not world.task_completed()
    native.completed[0] = True
    assert world.task_completed()
    assert not native.end_flag[0]
    native.success[0] = False
    assert not world.task_completed()


@pytest.mark.parametrize("explicit", [None, "user-evaluation"])
def test_worker_output_isolation_and_explicit_run_id(monkeypatch, tmp_path, explicit):
    from roboenv_runtime.adapters.robodojo import backend

    native_args = (
        "--/app/renderer/waitIdle=true --/app/updateOrder/checkForHydraRenderComplete=1000"
    )
    monkeypatch.setitem(
        sys.modules, "env.camera_manager.capture.render_sync",
        SimpleNamespace(zero_delay_kit_args=lambda: native_args),
    )

    def launch(backend, **kwargs):
        assert backend == "robodojo"
        assert "isaacsim.asset.gen.conveyor" not in kwargs["kit_args"]
        assert "--enable isaacsim.replicator.behavior" in kwargs["kit_args"]
        assert "--enable isaacsim.sensors.camera" in kwargs["kit_args"]
        assert kwargs["kit_args"].endswith(native_args)
        return SimpleNamespace(app=object())

    monkeypatch.setattr("roboenv_runtime.adapters.gpu.launch_isaaclab", launch)
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
        reward_manager=SimpleNamespace(),
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


def test_native_layout_selection_and_initialization_provenance(monkeypatch):
    select_layout = devices.RoboDojoEnv._layout_id_for
    native = lazy_environment(monkeypatch)
    monkeypatch.setattr(devices.RoboDojoEnv, "_layout_id_for", select_layout)
    native.seed_manager = SimpleNamespace(seed_list=[9, 3], seed_info={3: {}, 9: {}})
    world = devices.RoboDojoDevices("general_pickup", max_steps=7500)
    # Stop before device construction; reset provenance must already be complete.
    class DeviceConstructionReached(Exception):
        pass

    def stop():
        raise DeviceConstructionReached

    native.robot_manager = SimpleNamespace(get_robot_by_arm_name=lambda _: stop())
    assert world._layout_id_for(3) == 3
    with pytest.raises(DeviceConstructionReached):
        world.reset(seed=1)
    assert world.initialization["layout_ids"] == [9, 3]
    assert world.initialization["effective_layout"] == 3
    assert native.env_seeds == [3]
    assert world.initialization["max_steps"] == native.step_lim == 7500
    assert world.initialization["control_frequency_hz"] == 25


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
        obs_manager=SimpleNamespace(render_for_capture=lambda: calls.append("fence")),
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
    assert calls == ["fence", ([0], [1])]


def test_first_image_after_motion_waits_for_native_camera_delivery():
    camera = object()
    state = {"physics": 2, "queued": 1, "delivered": 0, "fences": 0}

    def render():
        state["delivered"], state["queued"] = state["queued"], state["physics"]

    def fence():
        state["fences"] += 1
        render()
        render()

    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._env = SimpleNamespace(
        render=render,
        obs_manager=SimpleNamespace(render_for_capture=fence),
        camera_manager=SimpleNamespace(cameras=[[camera]], camera_names=[["cam_head"]]),
        capture_manager=SimpleNamespace(step=lambda **kwargs: [{"rgb": [{
            "data": np.full((2, 3, 3), state["delivered"], np.uint8),
        }]}]),
    )
    world.device_facts = {"head_camera": {"intrinsics": {"color": {"width": 3, "height": 2}}}}
    first, _ = world.device_image("head_camera")
    assert np.all(first == 2)
    state["physics"] = 3
    second, _ = world.device_image("head_camera")
    assert np.all(second == 3)
    assert state["fences"] == 2


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


class _View:
    def __init__(self, valid):
        self.valid = valid

    def is_physics_handle_valid(self):
        return self.valid

    def get_world_positions(self):
        return np.ones((1, 2, 3))

    def get_velocities(self):
        return np.zeros((1, 2, 3))


def _object(**fields):
    return SimpleNamespace(get_world_pose=lambda: np.zeros(7), **fields)


def test_object_state_reads_only_the_handles_an_object_has():
    from roboenv_runtime.adapters.robodojo.devices import object_state

    encode = lambda value: np.asarray(value).tolist()  # noqa: E731
    # A conveyor item is a plain prim wrapper among the rigid objects: a pose, no velocity.
    assert set(object_state("rigid", _object(), encode)) == {"pose"}
    rigid = _object(get_linear_velocity=lambda: np.ones(3), get_angular_velocity=lambda: np.ones(3))
    assert object_state("rigid", rigid, encode)["linear_velocity"] == [1.0, 1.0, 1.0]
    # A garment's cloth view is read once initialized; before that the prim's own points stand in.
    prim = SimpleNamespace(GetAttribute=lambda name: SimpleNamespace(Get=lambda: [(1.0, 2.0, 3.0)]))
    before = object_state("garment", _object(_cloth_prim_view=_View(False), prim=prim), encode)
    assert before["points"] == [[1.0, 2.0, 3.0]] and "velocities" not in before
    after = object_state("garment", _object(_cloth_prim_view=_View(True), prim=prim), encode)
    assert after["points"] == np.ones((1, 2, 3)).tolist()
    assert after["velocities"] == np.zeros((1, 2, 3)).tolist()
