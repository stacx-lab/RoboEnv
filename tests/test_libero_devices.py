"""Official LIBERO contract gates; native physics stays in its isolated interpreter."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import ROOT, kinex_sources, load_tool
from roboenv.tasks.libero import ROBOTS, entries
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.backend import LiberoEnv
from roboenv_runtime.runtime.worker import SimWorker


def require(condition, message):
    if not condition:
        if os.environ.get("ROBOENV_LIBERO_ACCEPTANCE") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.mark.parametrize("size", [(128, 128), (256, 128), (512, 512)])
def test_camera_size_must_match_catalog(size):
    from roboenv_runtime.adapters.libero.backend import LiberoEnv

    with pytest.raises(ValueError, match="catalogs require camera_size"):
        LiberoEnv(camera_size=size)


def interpreter():
    value = os.environ.get("ROBOENV_LIBERO_TEST_PYTHON")
    require(value and Path(value).is_file(), "Requires prepared official LIBERO interpreter")
    return value


def run_native(script, *args, timeout=180, render=False):
    result = subprocess.run(
        [interpreter(), str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        env={
            **os.environ,
            **({} if render else {"CUDA_VISIBLE_DEVICES": ""}),
            "MUJOCO_GL": "egl",
            "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src")),
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_official_registry_and_assets_are_complete():
    base = ROOT / "third_party/LIBERO/libero/libero"
    require(base.exists(), "Requires pinned LIBERO checkout, but no simulator imports")
    tasks = list(entries(base))
    assert len(tasks) == 130
    assert {t["robot"] for t in tasks} == set(ROBOTS.values())
    assert {t["problem"] for t in tasks} == set(ROBOTS)
    lookup = {(t["suite_name"], t["task_id"]): t["robot"] for t in tasks}
    for suite, task_id, scene in [
        ("libero_spatial", 0, "tabletop"),
        ("libero_object", 0, "floor"),
        ("libero_10", 0, "living-room"),
        ("libero_10", 2, "kitchen"),
        ("libero_10", 5, "study"),
    ]:
        assert lookup[suite, task_id] == f"libero/libero-panda-{scene}"
    assert [
        sum(t["suite_name"] == suite for t in tasks)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    ] == [10, 10, 10, 10, 90]
    assert {p.parent.name for p in (ROOT / "catalog/tasks/libero").glob("*/task.yaml")} == {
        "bowl-onto-plate",
        "close-top-drawer",
        "bowl-into-bottom-drawer",
        "mug-into-microwave",
        "drawer-and-bowl",
    }
    for task in (
        "bowl-onto-plate",
        "close-top-drawer",
        "bowl-into-bottom-drawer",
        "mug-into-microwave",
        "drawer-and-bowl",
    ):
        resolved = resolve_task("libero/" + task, project=ROOT)
        assert resolved.task.backend == "libero"
        assert resolved.task.budget_mode == "practice"
        assert resolved.task.backend_kwargs["init_state_index"] == 0
        assert "realtime" not in resolved.task.backend_kwargs
        kwargs = resolved.task.backend_kwargs
        assert resolved.task.robot == lookup[kwargs["suite_name"], kwargs["task_id"]]
        assert resolved.task.device_bindings is None
        assert not resolved.path.with_name("solution.py").exists()


def test_discovered_scene_robots_reach_prepared_kinex_workspaces(tmp_path):
    from roboenv.tasks.catalog import discover

    require((ROOT / "third_party/LIBERO/libero/libero").exists(), "Requires pinned LIBERO")
    destination = tmp_path / "discovered"
    assert discover(ROOT, destination, backends=["libero"])["tasks"] == 130
    native = {
        (e["suite_name"], e["task_id"]): e
        for e in entries(ROOT / "third_party/LIBERO/libero/libero")
    }
    checked = set()
    for path in (destination / "tasks/libero").glob("*/task.yaml"):
        resolved = resolve_task(path, catalog=ROOT / "catalog")
        kwargs = resolved.task.backend_kwargs
        problem = native[kwargs["suite_name"], kwargs["task_id"]]["problem"]
        assert resolved.task.robot == native[kwargs["suite_name"], kwargs["task_id"]]["robot"]
        assert resolved.task.device_bindings is None
        assert all(not spec.overrides for spec in resolved.bindings.devices)
        if problem in checked:
            continue
        workspace = tmp_path / problem
        load_tool("prepare_kinex_workspace").prepare(
            path, workspace, project=ROOT, catalog=ROOT / "catalog"
        )
        for spec in resolved.bindings.devices:
            profile = workspace / ".kinex/devices" / spec.name / "profile.yaml"
            assert yaml.safe_load(profile.read_text()) == yaml.safe_load(
                (resolved.robot / spec.profile).read_text()
            )
        assert (workspace / ".kinex/workspace.yaml").read_bytes() == (
            resolved.robot / "workspace.yaml"
        ).read_bytes()
        checked.add(problem)
    assert checked == set(ROBOTS)


def test_catalogs_are_complete_portable_and_use_the_shared_driver(tmp_path):
    from roboenv.devices.export import DRIVERS

    transport = (ROOT / "src/roboenv/assets/_roboenv.py").read_bytes()
    profiles = []
    assert {p.name for p in (ROOT / "catalog/robots/libero").iterdir()} == {
        Path(rig).name for rig in ROBOTS.values()
    }
    for rig in ROBOTS.values():
        root = ROOT / "catalog/robots" / rig
        assert {p.name for p in root.iterdir()} == {"workspace.yaml", "devices"}
        assert {p.name for p in (root / "devices").iterdir()} == {
            "arm",
            "gripper",
            "camera",
            "camera_wrist",
            "world",
            "_roboenv.py",
        }
        for file in root.rglob("*"):
            if file.is_file():
                assert b"???" not in file.read_bytes()
                assert file.name != "connection.json"
        for path in root.glob("devices/*/profile.yaml"):
            profile = yaml.safe_load(path.read_text())
            profiles.append(profile)
            assert profile["access"]["driver"]["persistent"] is True
            assert (
                path.with_name("driver.py").read_bytes()
                == (ROOT / "src/roboenv/assets" / DRIVERS[profile["type"]]).read_bytes()
            )
            assert not path.with_name("_roboenv.py").exists()
            assert (root / "devices/_roboenv.py").read_bytes() == transport
            if profile["type"] == "joint":
                assert len(profile["intrinsics"]["model"]["joints"]) == 7
                assert profile["intrinsics"]["control"] == {
                    "mode": "position",
                    "limit_handling": "native",
                }
            if profile["type"] == "rgb":
                color = profile["intrinsics"]["color"]
                assert (color["width"], color["height"]) == (256, 256)
                focal = {
                    "camera": 309.019335983756,
                    "camera_wrist": 166.812847723674,
                }[profile["name"]]
                assert color["projection"]["fx"] == pytest.approx(focal)
                assert color["projection"]["fy"] == pytest.approx(focal)
                assert color["projection"]["cx"] == 127.5
                assert color["projection"]["cy"] == 127.5
        assert "simulator" not in str(
            yaml.safe_load((root / "devices/arm/profile.yaml").read_text())["access"]
        )
    kinex_root, _ = kinex_sources()
    node = shutil.which(os.environ.get("NODE", "node"))
    require(
        node and (kinex_root / "packages/core/dist/index.js").is_file(),
        "Kinex schema integration requires built Kinex and Node >=22.19",
    )
    workspace = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare("libero/bowl-onto-plate", workspace, project=ROOT)
    checked = subprocess.run(
        [node, str(ROOT / "tests/native_rigs_kinex.mjs"), str(kinex_root), str(workspace)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "KINEX_CONFIG_DIR": str(tmp_path / "config")},
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["results"][0]["profiles"] == 5


def test_static_generation_is_deterministic(tmp_path):
    for name in ("first", "second"):
        destination = tmp_path / name
        run_native("tools/generate_libero_config.py", "--out", destination)
        expected = {
            path.relative_to(ROOT / "catalog")
            for subtree in ("robots/libero",)
            for path in (ROOT / "catalog" / subtree).rglob("*")
            if path.is_file()
        }
        actual = {p.relative_to(destination) for p in destination.rglob("*") if p.is_file()}
        assert actual == expected
        for path in destination.rglob("*"):
            if path.is_file():
                relative = path.relative_to(destination)
                catalog_path = ROOT / "catalog" / relative
                if path.suffix == ".yaml":
                    # Mapping order is not a device fact; keep catalog diffs semantic.
                    assert yaml.safe_load(path.read_text()) == yaml.safe_load(
                        catalog_path.read_text()
                    )
                else:
                    assert path.read_bytes() == catalog_path.read_bytes()
                if name == "second":
                    assert path.read_bytes() == (tmp_path / "first" / relative).read_bytes()


@pytest.mark.parametrize("case", range(8))
@pytest.mark.parametrize("init", [0, 1])
def test_native_geometry_control_and_initialization(tmp_path, case, init):
    run_native("tests/libero_native.py", "--out", tmp_path, "--case", case, "--init", init)
    report = json.loads((tmp_path / "report.json").read_text())[0]
    assert report["passed"] and not report["full_task_attempted"]
    assert report["fk_error_m"] < 1e-9


def test_clock_budget_and_replay(tmp_path):
    run_native("tests/libero_clock.py", "--out", tmp_path)


def test_native_contact_stop_and_release(tmp_path):
    run_native("tests/libero_contact.py", "--out", tmp_path / "contact.json")
    assert json.loads((tmp_path / "contact.json").read_text())["passed"]


def test_native_camera_pixels_survive_concurrent_rpc(tmp_path):
    require(
        os.environ.get("ROBOENV_LIBERO_RENDER_TESTS") == "1",
        "Requires explicitly enabled native EGL camera tests",
    )
    run_native(
        "tests/libero_camera_rpc.py",
        "--out",
        tmp_path,
        render=True,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] and report["camera_reads"] == 10


def libero_world(*, limit=None, stop=True, success=False):
    native = LiberoEnv.__new__(LiberoEnv)
    native.configured_max_steps = limit
    native._max_steps = limit
    native._step_count = 5
    native.stop_on_success = stop
    native.terminal = False
    native.on_step = None
    native.owner_recording = None
    native._success = False
    native.initialization = {"init_state_index": 0}
    native.command_log = []
    native.control_protocol = "absolute_joint_position"
    sim = SimpleNamespace(forward=lambda: None, get_state=lambda: np.zeros(2))
    native.wrapper = SimpleNamespace(check_success=lambda: success,
                                    env=SimpleNamespace(sim=sim, control_freq=20,
                                                        step=lambda _: None, close=lambda: None))
    native.env = native.wrapper.env
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    adapter._env = native
    adapter._set_stop_on_success(stop)
    return adapter, native


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_device_stop_scope_and_persistent_gripper_target(direction):
    _, native = libero_world()
    measured = np.linspace(-0.3, 0.3, 7)
    updates = []
    controller = SimpleNamespace(
        name="JOINT_POSITION", goal_qpos=np.ones(7),
        update=lambda **kwargs: updates.append(kwargs),
    )
    controller.reset_goal = lambda: setattr(controller, "goal_qpos", measured.copy())
    controller.set_goal = lambda _action, set_qpos: setattr(controller, "goal_qpos", set_qpos)
    gripper = SimpleNamespace(current_action=np.array([1.0, -1.0]))
    native.env.robots = [SimpleNamespace(
        controller=controller, gripper=gripper, action_dim=8,
        _ref_joint_gripper_actuator_indexes=np.array([0, 1]),
        _ref_gripper_joint_pos_indexes=np.array([0, 1]),
    )]
    native.sim.model = SimpleNamespace(actuator_ctrlrange=np.array([[0, .04], [-.04, 0]]))
    native.sim.data = SimpleNamespace(qpos=np.array([.03, -.01]))
    native._gripper_command = direction
    initial_target = gripper.current_action.copy()

    native.stop_part("arm")
    assert updates == [{"force": True}]
    np.testing.assert_array_equal(controller.goal_qpos, measured)
    assert native._gripper_command == direction
    np.testing.assert_array_equal(gripper.current_action, initial_target)
    assert native.get_step_budget()["used"] == 5

    # Exercise the real step/tick path: arm-only writes retain the last direction.
    actions = []
    native.env.step = lambda action: actions.append(action.copy())
    native.step({"arm": np.zeros(7)})
    assert not actions
    native.tick()
    assert actions[-1][-1] == direction
    assert native.get_step_budget()["used"] == 6

    native.step({"gripper": [0]})
    native.tick()
    np.testing.assert_array_equal(gripper.current_action, initial_target)
    assert actions[-1][-1] == 0
    assert native.get_step_budget()["used"] == 7
    arm_target = controller.goal_qpos.copy()
    native.stop_part("gripper")
    assert native._gripper_command == 0
    # Measured asymmetric fingers map to [0.5, 0.5], replacing the old preload.
    np.testing.assert_allclose(gripper.current_action, [.5, .5])
    np.testing.assert_array_equal(controller.goal_qpos, arm_target)
    assert native.get_step_budget()["used"] == 7
    with pytest.raises(ValueError, match="Unknown native actuator"):
        native.stop_part("unknown")
    assert native.get_step_budget()["used"] == 7


@pytest.mark.parametrize("limit,stop,success,reason", [
    (None, True, True, "success"), (6, False, False, "step_limit"),
    (None, False, True, None),
])
def test_control_interval_seals_native_verdict(tmp_path, monkeypatch, limit, stop, success, reason):
    monkeypatch.delenv("ROBOENV_RECORD_EPISODE", raising=False)
    adapter, native = libero_world(limit=limit, stop=stop, success=success)
    worker = SimWorker(adapter, task_id="libero/10_3", seed=0, session="test",
                       run_dir=tmp_path, reset_adapter=False, execution_mode="stepped")
    try:
        native._advance(np.zeros(8))
        status = worker.simulation_status(include_verdict=True)
        assert status["terminal"] == (reason is not None)
        assert status["budget"]["used"] == 6
        assert native._success == success
        assert status["reason"] == reason
        if reason is not None:
            receipt = json.loads((tmp_path / "terminal.json").read_text())
            assert receipt["status"]["success"] == success
            assert native.terminal
            native.tick()
            assert native._step_count == 6
            with pytest.raises(RuntimeError):
                native.step({"arm": [0] * 7})
        else:
            worker.submit()
            assert native.terminal
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_recording_starts_after_initialization_and_seals_without_waiting(
    tmp_path, monkeypatch, mode
):
    import roboenv_runtime.recording as recording
    from roboenv_runtime.runtime import worker as worker_module

    calls, clock = [], [10.0]
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Recorder:
        def __init__(self, directory, identity, **kwargs):
            calls.append(("start", directory.name, identity["execution_mode"]))
            clock[0] += 5

        def capture(self, frame, simulation_seconds, **kwargs):
            calls.append(("capture", frame))

        def seal(self, frame, simulation_seconds, success):
            calls.append(("seal", frame, success))

        def finish(self):
            calls.append("finish")

    monkeypatch.setattr(recording, "Recording", Recorder)
    monkeypatch.setattr(recording, "MujocoSource", lambda native: object())
    monkeypatch.setenv("ROBOENV_RECORD_EPISODE", "1")
    adapter, native = libero_world(success=True)
    worker = SimWorker(adapter, task_id="libero/10_0", seed=0, run_dir=tmp_path,
                       reset_adapter=False, execution_mode=mode)
    try:
        if mode == "realtime":
            assert worker.world.next_tick == 15 and worker.world.time_s == 0
        native._advance(np.zeros(8))
        assert calls == [("start", "recording", mode), ("capture", 5),
                         ("capture", 6), ("seal", 6, True)]
        assert worker.simulation_status(include_verdict=True)["success"]
        assert "finish" not in calls
    finally:
        worker.close()
    assert calls[-1] == "finish"


def test_finalization_io_failure_still_releases_native_world(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("ROBOENV_RECORD_EPISODE", raising=False)
    adapter, native = libero_world()
    closed = []
    native.wrapper.close = lambda: closed.append(True)
    worker = SimWorker(adapter, task_id="libero/10_0", seed=0, run_dir=tmp_path,
                       reset_adapter=False, execution_mode="stepped")

    def fail():
        raise OSError("fixture disk failure")

    worker.recording = SimpleNamespace(finish=fail)
    worker.close()
    assert closed and worker.state == "closed"
    assert "Owner recording finalization failed" in caplog.text
