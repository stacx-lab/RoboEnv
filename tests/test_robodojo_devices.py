"""Lazy native creation and clock checks without substituting for GPU acceptance."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.robodojo import devices
from roboenv_runtime.adapters.robodojo.rig import camera_intrinsics, euler_pose


@pytest.mark.parametrize("failure", [None, "checksum", "particles"])
def test_fluid_video_restoration_uses_archived_initial_state_without_steps(monkeypatch, failure):
    import copy

    from roboenv_runtime.adapters.native_record import digest

    archived = {"task_objects": {"fluid": {"bottle": {
        "positions": [[[0.1, 0.2, 0.3]], None, None], "velocities": [[0, 0, 0]],
    }}, "rigid": {"bowl": {"pose": [[1, 2, 3], [1, 0, 0, 0]],
                              "linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0]}}}}
    actual = copy.deepcopy(archived)
    actual["task_objects"]["fluid"]["bottle"]["positions"][0][0][0] += 0.013
    before = copy.deepcopy(actual)
    writes = []

    def set_field(kind, field, value):
        writes.append((kind, field))
        name = "bottle" if kind == "fluid" else "bowl"
        actual["task_objects"][kind][name][field] = np.asarray(value).tolist()

    fluid = SimpleNamespace(
        get_particle_positions=lambda: (np.asarray(actual["task_objects"]["fluid"]
                                                   ["bottle"]["positions"][0]), None, None),
        point_instancer=SimpleNamespace(GetVelocitiesAttr=lambda: SimpleNamespace(
            Set=lambda value: set_field("fluid", "velocities", value))),
    )

    def set_positions(value):
        writes.append(("fluid", "positions"))
        actual["task_objects"]["fluid"]["bottle"]["positions"] = [value.tolist(), None, None]

    fluid.set_particle_positions = set_positions
    rigid = SimpleNamespace(
        _device="cpu",
        set_linear_velocity=lambda value: set_field("rigid", "linear_velocity", value),
        set_angular_velocity=lambda value: set_field("rigid", "angular_velocity", value),
    )

    def set_pose(**kwargs):
        writes.append(("rigid", "pose"))
        actual["task_objects"]["rigid"]["bowl"]["pose"] = [
            kwargs["position"].tolist(), kwargs["orientation"].tolist()]

    rigid.set_world_pose = set_pose
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._env = SimpleNamespace(scene_manager=SimpleNamespace(
        _fluid_objects=[{"bottle": fluid}], _rigid_and_dynamic_objects=[{"bowl": rigid}]))
    world.owner_state = lambda: {"state": copy.deepcopy(actual), "sha256": digest(actual)}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32=np.float32, as_tensor=lambda value, **kwargs: np.asarray(value, dtype=np.float32)))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Vt=SimpleNamespace(Vec3fArray=SimpleNamespace(FromNumpy=lambda value: value))))
    if failure == "particles":
        archived["task_objects"]["fluid"]["bottle"]["positions"][0].append([0, 0, 0])
        archived["task_objects"]["fluid"]["bottle"]["velocities"].append([0, 0, 0])
    record = {"state": archived, "sha256": "invalid" if failure == "checksum" else digest(archived)}
    if failure:
        with pytest.raises(ValueError, match="checksum" if failure == "checksum" else "dimensions"):
            world.restore_replay_initial_state(record)
        assert writes == [] and actual == before
    else:
        report = world.restore_replay_initial_state(record)
        assert report["source"] == "archived-native-initial-state"
        assert report["sha256"] == record["sha256"] and report["final_state_restored"] is False
        assert len(writes) == 5
        assert actual["task_objects"]["fluid"]["bottle"]["positions"][0][0][0] == pytest.approx(0.1)
        assert world.initial_state["sha256"] == digest(actual)


@pytest.mark.parametrize("tampered", [False, True])
def test_video_materials_use_pinned_originals_in_session_layer(monkeypatch, tmp_path, tampered):
    import hashlib
    from contextlib import contextmanager

    from roboenv.provisioning import robodojo_assets

    original = b"original shader"
    relative = "Metals/native.mdl"
    path = tmp_path / ".roboenv-materials" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed shader" if tampered else original)
    pin = {"base_url": "https://materials.example/", "files": {
        relative: {"size": len(original), "sha256": hashlib.sha256(original).hexdigest()}}}
    monkeypatch.setattr(robodojo_assets, "material_manifest", lambda: pin)
    writes, editing = [], []

    def set_asset(value):
        assert editing == ["session"]
        writes.append(value.path)

    attribute = SimpleNamespace(
        Get=lambda: SimpleNamespace(path=pin["base_url"] + relative), Set=set_asset,
        GetPath=lambda: "/World/robot/material/info:mdl:sourceAsset")
    stage = SimpleNamespace(Traverse=lambda: [SimpleNamespace(GetAttribute=lambda _: attribute)],
                            GetSessionLayer=lambda: "session")

    @contextmanager
    def context(current_stage, layer):
        assert current_stage is stage
        editing.append(layer)
        yield
        editing.clear()

    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage",
                        SimpleNamespace(get_current_stage=lambda: stage))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Sdf=SimpleNamespace(AssetPath=lambda value: SimpleNamespace(path=value)),
        Usd=SimpleNamespace(EditContext=context)))
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world.asset_roots = {"robodojo": tmp_path}
    if tampered:
        with pytest.raises(ValueError, match="missing or changed"):
            world.prepare_replay_materials()
        assert writes == []
    else:
        report = world.prepare_replay_materials()
        assert writes == [str(path)]
        assert report[0]["source_url"] == pin["base_url"] + relative
        assert report[0]["sha256"] == hashlib.sha256(original).hexdigest()
        assert not editing


@pytest.mark.parametrize("invalid", [False, True])
def test_fluid_surface_updates_use_the_strong_session_layer_before_render(monkeypatch, invalid):
    from contextlib import contextmanager

    root = {"bottle/points": np.zeros((1, 3), dtype=np.float32),
            "bottle/faceVertexIndices": np.zeros(3, dtype=np.int32)}
    session, editing = {}, []

    def write(field, value):
        assert editing == ["session"], "Root edits cannot override a session-layer replay mesh"
        session["bottle/" + field] = value.copy()

    mesh = SimpleNamespace(GetPrim=lambda: SimpleNamespace(
        GetAttribute=lambda field: SimpleNamespace(Set=lambda value: write(field, value))))
    stage = SimpleNamespace(GetSessionLayer=lambda: "session")

    @contextmanager
    def context(current, layer):
        assert current is stage
        editing.append(layer)
        yield
        editing.clear()

    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage",
                        SimpleNamespace(get_current_stage=lambda: stage))
    converter = SimpleNamespace(FromNumpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Usd=SimpleNamespace(EditContext=context),
        Vt=SimpleNamespace(IntArray=converter, Vec3fArray=converter)))
    world = devices.RoboDojoDevices.__new__(devices.RoboDojoDevices)
    world._fluid_surface_meshes = lambda: {"bottle": mesh}
    world.fluid_surface_state = lambda: {
        key: session.get(key, value) for key, value in root.items()}
    restored = {key: value.copy() for key, value in root.items()}
    restored["bottle/points"][:] = float("nan") if invalid else 0.25
    if invalid:
        with pytest.raises(ValueError, match="Invalid recorded fluid surface"):
            world.restore_fluid_surfaces(restored)
        assert session == {} and editing == []
    else:
        world.restore_fluid_surfaces(restored)
        np.testing.assert_array_equal(world.fluid_surface_state()["bottle/points"], [[0.25] * 3])
        np.testing.assert_array_equal(root["bottle/points"], [[0] * 3])
        assert not editing


def test_garment_evidence_uses_native_cpu_points_without_an_initialized_tensor_view():
    points = np.array([[1, 2, 3], [4, 5, 6]])
    velocity = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    obj = SimpleNamespace(_device="cpu", _cloth_prim_view=object(),
                          sample_mesh_vertices=lambda: (points, None, None, None),
                          _prim=SimpleNamespace(GetAttribute=lambda key:
                              SimpleNamespace(Get=lambda: velocity
                                              if key == "velocities" else None)))
    actual = devices.garment_particles(obj)
    np.testing.assert_array_equal(actual["points"], points[None])
    np.testing.assert_array_equal(actual["velocities"], velocity[None])
    obj._prim = SimpleNamespace(GetAttribute=lambda _: SimpleNamespace(Get=lambda: None))
    with pytest.raises(ValueError, match="missing particle velocities"):
        devices.garment_particles(obj)


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


@pytest.mark.parametrize("requested,terminal_frame", [
    (None, 300), (200, 200), (5000, 5000), (5000, 2),
])
def test_explicit_frame_budget_reaches_native_eval_and_preserves_early_stop(
    monkeypatch, requested, terminal_frame,
):
    from roboenv_runtime.adapters import native_record

    native = lazy_environment(monkeypatch)
    native.step_lim, native.end_flag, native.take_action_cnt = 300, [False], [0]
    native.seed_manager = SimpleNamespace(seed_list=[0, 1, 2])
    robots = [SimpleNamespace(type="target", arm_joints_name=[side],
                              arm_joint_indices=[0], gripper_joint_indices=[1])
              for side in ("left", "right")]
    keys = [SimpleNamespace(data=SimpleNamespace(joint_pos_limits=[[[-1, 1]]],
                                                 joint_vel_limits=[[1]],
                                                 joint_effort_limits=[[1]])) for _ in robots]
    native.robot_manager = SimpleNamespace(
        robot_list=robots, robot_key=keys,
        get_robot_by_arm_name=lambda name: robots[0 if name == "left_arm" else 1],
        get_link_pose=lambda *args: [[0, 0, 0, 1, 0, 0, 0]],
    )
    facts = {side + "_arm": {
        "intrinsics": {"model": {"joints": [{"name": side, "limits": {"min": -1, "max": 1},
                                             "max_velocity": 1, "max_effort": 1}]}},
        "extrinsics": {"mount": {"translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}},
    } for side in ("left", "right")}
    monkeypatch.setattr(devices, "describe", lambda *args: (
        facts, {side: {"base": side} for side in ("left", "right")}, {},
    ))
    recorded = []
    monkeypatch.setattr(native_record, "initialize_record",
                        lambda world, config, backend: recorded.append(config))
    world = devices.RoboDojoDevices("fixture", max_steps=requested)
    world.task_name = "fixture"
    world._measured_target = lambda name: [0]
    world._verify_cameras = lambda: None
    world.owner_state = lambda: {"sha256": "fixture"}
    world._apply = lambda targets: None
    world.native_action = {}

    def take_action(action):
        # EvalEnv refuses frames beyond its own horizon, independently of the owner.
        if native.take_action_cnt[0] >= native.step_lim or native.end_flag[0]:
            return
        for _ in range(native.obs_manager.collect_interval):
            native.sim_step()
        native.take_action_cnt[0] += 1
        native.end_flag[0] = native.take_action_cnt[0] == terminal_frame

    native.take_action = take_action
    world.reset(seed=1)
    expected_limit = 300 if requested is None else requested
    assert native.step_lim == world.frame_limit == recorded[0]["max_steps"] == expected_limit
    for _ in range(terminal_frame):
        world.device_write("left_arm", [0])
        world.control.tick()
    assert world.control.frame == native.take_action_cnt[0] == terminal_frame
    assert world.physics_steps == terminal_frame * native.obs_manager.collect_interval
    with pytest.raises(RuntimeError, match="budget exhausted"):
        world.device_write("left_arm", [0])
        world.control.tick()
    assert world.control.frame == native.take_action_cnt[0] == terminal_frame


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_frame_budget_fails_before_native_creation(monkeypatch, invalid):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid budgets must not initialize the native environment")

    monkeypatch.setattr(devices.RoboDojoEnv, "__init__", unexpected)
    with pytest.raises(ValueError, match="positive action-frame count"):
        devices.RoboDojoDevices("fixture", max_steps=invalid)


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
