"""Public transport contracts on instrumented worlds; not native physics evidence."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.model import DeviceError
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]


def test_robodojo_public_instruction_uses_cached_native_description():
    from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter

    class Native:
        obs_manager = SimpleNamespace(instruction=["Pick up the blue cup by 10 cm."])

        def gen_instruction(self, *_args):
            raise AssertionError("Public reads must not regenerate or resample descriptions")

    adapter = RoboDojoAdapter.__new__(RoboDojoAdapter)
    adapter._env = SimpleNamespace(_env=Native())
    assert adapter.goal() == "Pick up the blue cup by 10 cm."
    assert adapter.goal() == "Pick up the blue cup by 10 cm."


@pytest.mark.parametrize("override", [None, "Seat all four eggs and fully close the lid."])
def test_public_instruction_override_preserves_native_goal(override):
    from roboenv_runtime.runtime.worker import SimWorker

    native_goal = "Place the eggs into the holder, then close the lid."
    adapter = SimpleNamespace(goal=lambda: native_goal)
    worker = SimWorker.__new__(SimWorker)
    worker.state, worker.request_count = "ready", 0
    worker.adapter, worker.task_id = adapter, "fill_egg_holder"
    worker.session, worker.generation = "test", 0
    worker.devices = SimpleNamespace(environment=lambda: {
        "task": {"id": "robodojo/fill_egg_holder", "instruction": "Catalog goal"},
        "instruction_override": override, "devices": [],
    })
    description = worker._dispatch("environment.describe", {})
    assert description["task"]["instruction"] == (override or native_goal)
    assert adapter.goal() == native_goal
    assert description["task"]["id"] == "robodojo/fill_egg_holder"


def service(tmp_path, *, embodiment="piper"):
    root = ROOT / "catalog/robots/robotwin/robotwin-dual-piper"
    plan = BindingPlan.load(root, backend="robotwin")
    facts = {}
    for spec in plan.devices:
        definition = spec.definition(root)
        facts[spec.entity] = {
            "type": definition.type,
            "intrinsics": definition.intrinsics,
            "extrinsics": definition.extrinsics,
        }
    world = SimpleNamespace(embodiment=embodiment, device_facts=facts, steps=0, sample_time_s=0.0)
    actuator_names = [name for name, fact in facts.items() if fact["type"] in {"joint", "gripper"}]
    targets = {name: [0.0] * (6 if name.endswith("arm") else 1) for name in actuator_names}

    def advance():
        world.steps += 1

    world.control = DeviceControl(
        {name: [(-1.0, 1.0)] * len(value) for name, value in targets.items()},
        targets,
        lambda target: None,
        advance,
        exhausted=lambda: world.steps >= 3,
        interval_s=0.04,
    )
    world.device_write = world.control.write
    world.device_stop = lambda name: world.control.stop(name, targets[name])
    world.device_read = lambda name: {
        "positions": targets[name],
        "velocities": [0.0] * len(targets[name]),
    }
    world.device_image = lambda name: (np.zeros((240, 320, 3), dtype=np.uint8), np.ones((240, 320)))
    from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
    from roboenv_runtime.runtime.world import WorldSession

    owner = RoboTwinAdapter.__new__(RoboTwinAdapter)
    owner._env, owner.run_dir = world, tmp_path
    world.control_interval, world.frame_limit = 0.04, 3
    session = WorldSession(owner, mode="stepped")
    context = BindingContext("robotwin", "piper", owner, world=session)
    return (
        default_registry().assemble(
            plan, root, context, motion_scheduler=session.schedule, resolve=session.resolve
        ),
        world,
        plan,
        root,
        context,
    )


def test_public_world_has_no_truth_or_mutation_shortcuts(tmp_path):
    wire, world, *_ = service(tmp_path)
    try:
        assert {"read", "wait"} <= wire.describe("world")["actions"].keys()
        assert wire.call({"device": "world", "action": "read", "args": {}})["data"] == {
            "time_s": 0, "budget": {"elapsed_s": 0.0, "max_s": 3 * 0.04},
        }
        assert wire.call({"device": "world", "action": "status", "args": {}})["data"] == {
            "loaded": True,
        }
        for action in ("state", "invoke", "score", "reset", "restore"):
            with pytest.raises(DeviceError):
                wire.call({"device": "world", "action": action, "args": {}})
        assert world.steps == 0
    finally:
        wire.close()


def test_world_budget_read_remains_available_at_the_stepped_horizon(tmp_path):
    wire, world, *_ = service(tmp_path)
    try:
        with pytest.raises(DeviceError) as ended:
            wire.call({"device": "world", "action": "wait", "args": {"duration_s": 0.2}})
        assert ended.value.code == "episode_finished"
        for _ in range(2):
            assert wire.call({"device": "world", "action": "read", "args": {}})["data"] == {
                "time_s": pytest.approx(0.12),
                "budget": {"elapsed_s": pytest.approx(0.12), "max_s": pytest.approx(0.12)},
            }
        assert world.steps == 3
    finally:
        wire.close()


def test_independent_writes_and_read_stop_invariance(tmp_path):
    wire, world, *_ = service(tmp_path)

    def call(device, action, args=None):
        return wire.call({"device": device, "action": action, "args": args or {}})

    try:
        call("left_arm", "write", {"actions": [[0.1] * 6], "duration_s": len([[0.1] * 6]) / 25})
        assert world.steps == 0
        with pytest.raises(DeviceError):
            call(
                "right_arm",
                "write",
                {"actions": [[float("nan")] * 6], "duration_s": len([[float("nan")] * 6]) / 25},
            )
        assert world.steps == 0
        call("right_arm", "write", {"actions": [[0.2] * 6], "duration_s": len([[0.2] * 6]) / 25})
        call("left_gripper", "write", {"actions": [0.3, 0.7], "duration_s": 0.04})
        with pytest.raises(DeviceError):
            call("left_gripper", "write", {"actions": [[0.2]], "duration_s": 0.04})
        call("world", "wait", {"duration_s": 0.04})
        assert world.steps == 1
        assert world.control.targets["left_gripper"] == [0.7]
        for action in ("load", "read", "status", "stop", "unload"):
            call("left_arm", action)
        call("head_camera", "read")
        assert world.steps == 1
        assert len(list((tmp_path / "captures").glob("*.png"))) == 1
    finally:
        wire.close()


def test_native_aliases_are_allowed_and_rig_mismatches_fail_before_effects(tmp_path):
    with pytest.raises(ValueError, match="embodiment"):
        service(tmp_path, embodiment="aloha_agilex")
    wire, world, plan, root, context = service(tmp_path)
    try:
        alias = replace(next(s for s in plan.devices if s.entity == "left_arm"), name="alias")
        duplicate = replace(plan, devices=(*plan.devices, alias))
        aliases = default_registry().assemble(duplicate, root, context)
        try:
            assert aliases.describe("alias")["intrinsics"]["stop"]["affected_devices"] == ["alias"]
        finally:
            aliases.close()
        world.device_facts = copy.deepcopy(world.device_facts)
        world.device_facts["left_arm"]["intrinsics"]["home"]["joint1"] = 0.123
        with pytest.raises(ValueError, match="Catalog intrinsics"):
            default_registry().assemble(plan, root, context)
        assert world.steps == 0
    finally:
        wire.close()


def test_behavior_velocity_binding_accepts_named_commands_without_joint_feedback(tmp_path):
    root = ROOT / "catalog/robots/behavior/behavior-r1pro"
    plan = BindingPlan.load(root, backend="behavior")
    spec = next(s for s in plan.devices if s.name == "base")
    definition = spec.definition(root)
    world = SimpleNamespace(
        embodiment="r1pro",
        device_facts={
            "base": {
                "type": "velocity",
                "intrinsics": definition.intrinsics,
                "extrinsics": definition.extrinsics,
            }
        },
        control=SimpleNamespace(limits={"base": [(-0.5, 0.5), (-0.5, 0.5), (-1, 1)]}),
        device_read=lambda name: {"velocities": [0, 0, 0]},
    )
    writes = []

    def write(name, target):
        writes.append((name, target))
        return {"accepted": True}

    world.device_write = write
    world.device_stop = lambda name: {"stopped": True}
    from roboenv_runtime.core.types import Budget
    from roboenv_runtime.runtime.world import WorldSession

    owner = SimpleNamespace(
        _env=world,
        run_dir=tmp_path,
        control_interval_s=1 / 30,
        name="behavior",
        embodiment="r1pro",
        budget=lambda: Budget(used=0, max=None),
        advance_control=lambda: None,
        configure_execution_mode=lambda mode: None,
    )
    session = WorldSession(owner, mode="stepped")
    wire = default_registry().assemble(
        replace(plan, devices=(spec,)),
        root,
        BindingContext("behavior", "r1pro", owner),
        motion_scheduler=session.schedule,
        resolve=session.resolve,
    )
    try:
        assert wire.call({"device": "base", "action": "read", "args": {}})["data"] == {
            "velocities": [0, 0, 0]
        }
        command = {"duration_s": 0.1, "mode": "drive", "velocity": {"vx": 0.1, "vy": 0, "wz": -0.2}}
        wire.call({"device": "base", "action": "write", "args": command})
        assert writes == [("base", [0.1, 0, -0.2])]
        for args in (
            {"target": [0, 0, 0]},
            {**command, "posture": {"height": 0.7}},
            {**command, "step": {"id": 0}},
        ):
            with pytest.raises(DeviceError):
                wire.call({"device": "base", "action": "write", "args": args})
        assert len(writes) == 1
    finally:
        wire.close()


@pytest.mark.parametrize("limit,outcomes,expected", [
    (1, [False, True], False),
    (4, [False, False, False, True], True),
    (4, [True], True),
    (4, [False] * 4, False),
])
def test_robotwin_settling_preserves_native_check_and_bounds(
    monkeypatch, tmp_path, limit, outcomes, expected
):
    import importlib.util
    import sys

    monkeypatch.setenv("ROBOTWIN_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transforms3d", SimpleNamespace())
    source = ROOT / "runtime/src/roboenv_runtime/adapters/robotwin/backend.py"
    spec = importlib.util.spec_from_file_location("robotwin_settling_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worlds, remaining = [], iter(outcomes)

    class Task:
        def setup_scene(self):
            self.scene = SimpleNamespace(step=lambda: None)
            worlds.append(self.scene)

        def check_stable(self):
            for _ in range(5):
                self.scene.step()
            stable = next(remaining)
            return stable, [] if stable else ["garbage"]

    world = module.RoboTwinEnv.__new__(module.RoboTwinEnv)
    world._task_cls = Task
    world.max_settling_checks = limit
    world._initialization_steps_total = 0
    world._stability_checks = []
    task = world._new_task()
    task.setup_scene()
    assert task.check_stable() == (expected, [] if expected else ["garbage"])
    calls = outcomes.index(True) + 1 if expected else limit
    assert len(worlds) == 1
    assert world._initialization_steps_total == 5 * calls
    assert world._stability_checks == [[
        {"stable": stable, "unstable_objects": [] if stable else ["garbage"],
         "physics_steps": 5} for stable in outcomes[:calls]
    ]]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_settling_checks"):
            module.RoboTwinEnv(max_settling_checks=invalid)


@pytest.mark.parametrize("activity,rooms", [
    ("fixture_task", ["corridor_0", "living_room_0"]), ("unlisted_task", None),
])
def test_behavior_scene_uses_native_transit_rooms_instead_of_display_categories(
    monkeypatch, activity, rooms
):
    import sys

    from roboenv_runtime.adapters.behavior.backend import BehaviorBackend

    monkeypatch.setitem(sys.modules, "omnigibson.eval.utils.eval_utils", SimpleNamespace(
        TASK_NAMES_TO_ROOMS={"fixture_task": ["corridor_0", "living_room_0"]},
    ))
    world = BehaviorBackend(activity, "house_double_floor_lower", max_steps=50000)
    world.challenge = {"rooms": ["living_room"]}
    config = world.build_config()
    assert config["scene"]["load_room_types"] is None
    assert config["scene"]["load_room_instances"] == rooms
    assert config["scene"]["scene_model"] == world.scene_model
    assert config["task"]["activity_name"] == activity
    assert config["task"]["termination_config"]["max_steps"] == 50000


@pytest.mark.parametrize("cached,completed,raw", [
    ([0.0], False, 0.0), ([25.0], False, 25.0), ([100.0], True, 100.0),
    ([100.0], False, 100.0), (None, False, 0.0), (None, True, 100.0),
])
def test_robodojo_score_normalizes_points_without_changing_completion(cached, completed, raw):
    adapter = RoboDojoAdapter.__new__(RoboDojoAdapter)
    adapter._env = SimpleNamespace(
        task_completed=lambda: completed,
        _env=SimpleNamespace(reward_manager=SimpleNamespace(_gated_score_lst=cached)),
    )
    score = adapter.score()
    assert score.reward == raw / 100.0
    assert score.raw_score == raw
    assert score.completed is completed


@pytest.mark.parametrize("ending", ["success", "step_limit", "native_done", "stopped"])
@pytest.mark.parametrize("recording_failed", [False, True])
def test_native_device_terminal_callback_freezes_and_retains_evidence(
    tmp_path, monkeypatch, ending, recording_failed
):
    import json

    from roboenv.recording import inspect_recording
    from roboenv_runtime.adapters.native import NativeDeviceAdapter
    from roboenv_runtime.core.types import ScoreReport
    from roboenv_runtime.runtime.worker import SimWorker

    monkeypatch.setenv("ROBOENV_RECORD_EPISODE", "1")

    class NativeFixture(NativeDeviceAdapter):
        name = "robotwin"

        def __init__(self):
            self._set_stop_on_success(True)
            self._env = SimpleNamespace(
                realtime=False, frame_limit=1 if ending == "step_limit" else 3,
                                        control_interval=0.04, _native_done=False, completed=False)
            self._env.control = DeviceControl(
                {"arm": [(-1, 1)]}, {"arm": [0]}, lambda _: None, self.advance,
                interval_s=0.04, exhausted=lambda: self.budget().exhausted,
            )

        def episode_initialization(self):
            return {"schema": "roboenv/native-device-episode/1", "fixture": "initial"}

        def advance(self):
            self._env.completed = ending == "success"
            self._env._native_done = ending == "native_done"

        def score(self):
            return ScoreReport(reward=1.0 if self._env.completed else 0.5,
                               completed=self._env.completed)

        def observe(self):
            return {"schema": "roboenv/native-device-episode/1", "backend": self.name,
                    "native_success": self._env.completed,
                    "commands": list(self._env.control.commands)}

        def close(self):
            pass

    adapter = NativeFixture()
    worker = SimWorker(adapter, task_id="native-fixture", seed=0, session="s",
                       run_dir=tmp_path, reset_adapter=False, execution_mode="stepped")
    try:
        if recording_failed:
            worker.recording.writer.kill()
            worker.recording.writer.join()
        if ending == "stopped":
            worker.dispatch("simulation.stop", {})
        else:
            adapter._env.control.write("arm", [0.2])
            worker.world.resolve(worker.world.wait(adapter.control_interval_s))
        initial = json.loads((tmp_path / "native-initialization.json").read_text())
        assert initial["task"] == "native-fixture" and initial["seed"] == 0
        terminal = json.loads((tmp_path / "terminal.json").read_text())
        observation = json.loads((tmp_path / "native-episode.json").read_text())
        assert terminal["status"]["reason"] == ending
        assert observation["budget"] == terminal["status"]["budget"]
        assert observation["native_success"] == (ending == "success")
        assert observation["submitted"] is True
        # The sealed writer completes without waiting for native adapter close.
        worker.recording.writer.join(10)
        assert not worker.recording.writer.is_alive()
        recording = inspect_recording(tmp_path / "recording")
        assert recording["complete"] is (not recording_failed)
        assert worker._closed is False
        before = adapter.budget()
        with pytest.raises(RuntimeError):
            adapter._env.control.write("arm", [0.1])
        with pytest.raises(RuntimeError):
            adapter._env.control.stop("arm", [0.1])
        assert adapter.budget() == before
        assert worker.dispatch("simulation.stop", {})["budget"] == before.as_dict()
    finally:
        worker.close()
