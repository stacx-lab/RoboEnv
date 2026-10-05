"""Catalog coverage and native SIMPLE verdicts without simulator imports."""

import ast
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from roboenv.tasks.catalog import load_catalog
from roboenv.tasks.simple import discover_tasks, entries
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.backend import SimpleEnv
from roboenv_runtime.adapters.simple.control import Control
from roboenv_runtime.runtime.worker import SimWorker

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/SIMPLE"
NAMES = [
    "g1-xmove-pick", "g1-transfer-between-tables", "g1-bend-pick",
    "bend-pick-and-place", "bend-handover", "handover",
    "pick-and-place-and-hug-container", "close-door", "open-oven",
    "open-faucet", "push-office-chair", "open-trash-can",
]


def source_entries():
    if not (SOURCE / "src/simple/envs/__init__.py").exists():
        pytest.skip("requires pinned SIMPLE source")
    return list(entries(SOURCE))


def test_all_decoupled_tasks_have_matching_catalog_and_native_instructions():
    native = {entry["name"]: entry for entry in source_entries()}
    # Bend-handover's actual language differs from its class description.
    assert native["bend-handover"]["instruction"] == (
        "bend the robot and pick up the cracker box, then place it on the container."
    )
    # Constructor defaults override the class DR config (hug uses apple, not
    # its class-level cracker box). Fixture tasks must not inherit that apple.
    targets = {
        "g1-xmove-pick": "apple", "pick-and-place-and-hug-container": "apple",
        "g1-transfer-between-tables": "cracker box", "g1-bend-pick": "cracker box",
        "bend-pick-and-place": "cracker box", "bend-handover": "cracker box",
        "handover": "cracker box", "close-door": "door", "open-oven": "oven",
        "open-faucet": "faucet", "push-office-chair": "office chair",
        "open-trash-can": "trash can",
    }
    for name, target in targets.items():
        assert target in native[name]["instruction"]
        assert "target object" not in native[name]["instruction"]
        if target != "apple":
            assert "apple" not in native[name]["instruction"]
    catalog = load_catalog(ROOT / "catalog/tasks/simple")
    assert set(native) == set(NAMES)
    assert set(catalog) == {"simple/" + name for name in NAMES}
    for task, _ in discover_tasks(ROOT):
        assert asdict(task) == asdict(catalog[task.id][0])
        entry = native[task.id.split("/")[1]]
        assert task.instruction == entry["instruction"]
        assert "{}" not in task.instruction
        assert task.execution_mode == "realtime"
        assert task.backend_kwargs == {
            "env_id": entry["env_id"], "sim_mode": "mujoco_isaac", "dr_level": 0,
            "arm_gravity_compensation": True, "stop_on_success": False,
        }
        assert task.robot == "simple/simple-g1" and task.budget_mode == "practice"


def test_asset_pin_covers_every_default_task_scene_and_asset_family():
    pin = json.loads((ROOT / "src/roboenv/assets/simple-g1.json").read_text())
    archives = {entry["file"] for entry in pin["archives"]}
    tasks = source_entries()
    scenes = {s["uid"]: s["name"] for s in yaml.safe_load(
        (SOURCE / "src/simple/resources/hssd-scenes/config.yaml").read_text()
    )}
    families = {"graspnet1b": "graspnet", "objects": "objects", "articulated": "articulated"}
    for entry in tasks:
        for node in ast.walk(ast.parse(entry["source"].read_text())):
            if not isinstance(node, ast.keyword):
                continue
            if node.arg == "room_choices":
                for room in ast.literal_eval(node.value):
                    assert f"scenes_hssd_{scenes[room.split(':')[1]]}.zip" in archives
            if node.arg == "asset_id":
                family = ast.literal_eval(node.value).split(":")[0]
                assert f"assets_{families[family]}.zip" in archives


def native_task(entry):
    """Execute the pinned predicate methods with synthetic native observations."""
    tree = ast.parse(entry["source"].read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and (n.name.startswith("check_") or n.name == "compute_reward")]
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(methods)
    namespace = {"np": np, "_LIFT_HEIGHT": 0.1}
    exec(compile(module, str(entry["source"]), "exec"), namespace)
    task = type("NativePredicate", (), {n.name: namespace[n.name] for n in methods})()
    task.metadata = dict(entry["metadata"])
    constructor = next(n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in constructor.args.args][-len(constructor.args.defaults):],
                        constructor.args.defaults, strict=True))
    threshold = (ast.literal_eval(defaults["success_criteria"])
                 if "success_criteria" in defaults else None)
    task.success_criteria = (threshold if threshold is not None
                             else task.metadata["success_criteria"])
    task.reward, task._init_target_height, task.chair_init_pos = 0.0, None, None
    task.target = SimpleNamespace(asset=SimpleNamespace(label="target"))
    task.container = SimpleNamespace(asset=SimpleNamespace(label="basket"))
    task._distractors = [SimpleNamespace(asset=SimpleNamespace(label="distractor"))]
    task.layout = SimpleNamespace(actors={
        name: SimpleNamespace(pose=SimpleNamespace(position=[0, 0, 0]))
        for name in ("target", "container")
    })
    return task


def observation(success):
    pairs = [("target", "basket"), ("basket", "table"), ("distractor", "basket")] if success else []
    contacts = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
    joints = {"articulate_joint_0": [0.8 if success else 0],
              "articulate_joint_1": [-0.2 if success else 0],
              "articulate_joint_2": [1.5 if success else 0],
              "articulate_floating_base": [0.9 if success else 0, 0, 0]}
    native = SimpleNamespace(
        mjData=SimpleNamespace(ncon=len(contacts), contact=contacts,
                               joint=lambda name: SimpleNamespace(qpos=np.array(joints[name]))),
        mjModel=SimpleNamespace(geom=lambda name: SimpleNamespace(bodyid=name),
                                body=lambda name: SimpleNamespace(name=name)),
    )
    return {"target": [0, 0, 0.11 if success else 0]}, native


@pytest.mark.parametrize("name", NAMES)
def test_pinned_native_success_failure_and_accumulator_reset(name):
    entry = next(e for e in source_entries() if e["name"] == name)
    task = native_task(entry)

    def tick(success):
        info, native = observation(success)
        # Preserve SonicLocoManipEnv's reward then check_success call sequence.
        reward = task.compute_reward(info, mujoco_env=native)
        return reward, bool(task.check_success(info, mujoco_env=native))

    assert not tick(False)[1]
    assert tick(True)[0] > 0
    assert tick(False) == (0.0, False)
    assert any(tick(True)[1] for _ in range(100)), name
    assert tick(False) == (0.0, False)


@pytest.mark.parametrize("name", ["g1-transfer-between-tables", "bend-pick-and-place"])
def test_native_placement_requires_releasing_the_object(name):
    task = native_task(next(e for e in source_entries() if e["name"] == name))
    info, native = observation(True)
    native.mjData.contact.append(SimpleNamespace(geom1="target", geom2="right_hand"))
    native.mjData.ncon += 1
    for _ in range(100):
        assert not task.check_success(info, mujoco_env=native)
    assert task.reward == 0


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_native_hug_requires_object_container_table_and_distractor_contacts(missing):
    task = native_task(next(e for e in source_entries()
                            if e["name"] == "pick-and-place-and-hug-container"))
    info, native = observation(True)
    native.mjData.contact.pop(missing)
    native.mjData.ncon -= 1
    for _ in range(100):
        assert not task.check_success(info, mujoco_env=native)
    assert task.reward == 0


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("policy", [{}, {"stop_on_success": False}, {"stop_on_success": True}])
def test_simple_worker_success_policy_and_latched_verdict(tmp_path, monkeypatch, mode, policy):
    native = SimpleEnv.__new__(SimpleEnv)
    native.configured_max_steps = 6
    facts = ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json"
    native.model = json.loads(facts.read_text())
    native.control = Control(native, native.model)
    native.arm_gravity_compensation = True
    native.env = None

    def reset(seed):
        native.steps, native.reward, native.completed, native.done = 0, 0.0, False, False
        native.frames, native.images = [], {}

    native.reset = reset
    native.record = lambda: {"native_success": native.completed}
    native._step = lambda targets: ({}, 0.4, native.steps == 1, False, {})
    monkeypatch.setattr("roboenv_runtime.adapters.simple.backend.SimpleEnv", lambda **kw: native)
    adapter = SimpleAdapter(**policy)
    now = [0.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    worker = SimWorker(adapter, task_id="simple/test", seed=0,
                       run_dir=tmp_path, execution_mode=mode)
    try:
        pending = worker.world.wait(0.08)
        for _ in range(7):
            worker.world.advance()
            now[0] += 0.021
            if pending.done():
                break
        assert pending.done() and worker.world.failure is None
        stop = policy.get("stop_on_success", False)
        assert native.steps == (2 if stop else 4)
        assert not native.done and native.completed
        stamp = native.steps
        for _ in range(3):
            assert adapter.score().completed
        assert native.steps == stamp
        status = worker.simulation_status(include_verdict=True)
        assert status["terminal"] is stop
        if stop:
            assert status["reason"] == "success"
            worker.world.advance()
            assert native.steps == stamp
        else:
            if mode == "stepped":
                pending = worker.world.wait(0.1)
                for _ in range(5):
                    worker.world.advance()
                    if pending.done():
                        break
                assert pending.done() and native.steps == 6 and native.done
                assert worker.simulation_status(include_verdict=True)["reason"] == "success"
            result = worker.submit()
            assert result["completed"]
        assert worker.submit()["completed"]
    finally:
        worker.close()
    native.reset(1)
    assert not native.completed and native.steps == 0


def test_native_truncation_remains_terminal():
    native = SimpleEnv.__new__(SimpleEnv)
    native.steps, native.max_steps, native.done, native.completed = 0, None, False, False
    native.frames, native.images = [], {}
    native._step = lambda targets: ({}, 0, False, True, {})
    native.advance({})
    assert native.done and not native.completed
    with pytest.raises(ValueError, match="terminated"):
        native.advance({})
