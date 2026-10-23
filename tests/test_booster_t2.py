"""T2 task integrity, scoring and portable geometry without a native simulator."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import load_tool
from roboenv.devices.export import driver_resources
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.registry import get_task
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.environment import T2Env
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.scenes import load_scene
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.task import Box
from roboenv_runtime.adapters.isaaclab.tasks.t2_household.scoring import State

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"


def task(name):
    return yaml.safe_load((CATALOG / "tasks/isaaclab" / name / "task.yaml").read_text())


def test_t2_catalogs_and_portable_resources_are_closed():
    paths = sorted((CATALOG / "tasks/isaaclab").glob("t2-*/task.yaml"))
    assert {p.parent.name for p in paths} == {"t2-reach-table", "t2-can-in-bowl"}
    for path in paths:
        resolved = resolve_task(path, catalog=CATALOG)
        spec = resolved.task.backend_kwargs
        factory = get_task(spec["task_name"])
        assert factory.embodiment == "booster_t2"
        assert factory.name == "t2-household"
        assert factory.definition(spec["task_spec"]) == spec["task_spec"]
        assert resolved.task.execution_mode == "realtime"
        assert len(resolved.bindings.devices) == 8
    rig = CATALOG / "robots/isaaclab/isaaclab-booster-t2/devices"
    for path in rig.glob("*/profile.yaml"):
        profile = yaml.safe_load(path.read_text())
        facts = profile["intrinsics"]
        resources = driver_resources(path.parent)
        if profile["type"] in {"joint", "dexterous_hand"}:
            assert [joint["name"] for joint in facts["model"]["joints"]] == list(facts["home"])
        if facts.get("model", {}).get("format") == "urdf":
            assert Path(facts["model"]["path"]) in resources
        for resource in resources:
            text = (path.parent / resource).read_text()
            assert "locomanip_agentic" not in text
            assert "<mesh" not in text


@pytest.mark.parametrize("value", [None, {}, {"version": 1}, []])
def test_task_validation_rejects_incomplete_definitions(value):
    with pytest.raises(ValueError):
        get_task("t2-household").definition(value)


@pytest.mark.parametrize("mode,maximum", [("realtime", None), ("stepped", 100)])
def test_t2_mode_retains_native_physics_but_switches_budget(mode, maximum):
    env = T2Env.__new__(T2Env)
    env._stepped_max_steps, env.steps = 100, 7
    env.configure_execution_mode(mode)
    assert env.budget().as_dict() == {"used": 7, "max": maximum, "unit": "action_frames"}
    with pytest.raises(ValueError):
        env.configure_execution_mode("fast")


def test_nonfinite_and_out_of_range_base_commands_are_rejected_before_effects():
    env = T2Env.__new__(T2Env)
    env.policy_config = SimpleNamespace(command_range=np.array([[-1, 1]] * 3))
    env.velocity, env.events = np.zeros(3), []
    for bad in (True, float("nan"), float("inf"), 2, "0"):
        with pytest.raises(ValueError):
            env.set_target(
                "base", {"mode": "locomotion", "velocity": {"vx": bad, "vy": 0, "wz": 0}}
            )
        assert env.events == []
        assert not env.velocity.any()


def test_placement_requires_lift_release_and_continuous_dwell():
    # Preserve the version-2 predicate contract for historical replay inputs.
    spec = task("t2-reach-table")["backend_kwargs"]["task_spec"]
    spec.update(scene="kitchen", hold_s=0.5)
    spec["items"] = {"box": {"model": 12, "xy": [-0.3, -1.15],
                             "surface": "kitchen_island", "yaw": 0.0}}
    spec["boxes"] = [{"name": "zone", "pos": [-.3, 1.26, .761],
                      "size": [.36, .36, .002], "color": [0, 1, 0]}]
    spec["stages"] = [[{"kind": "place", "item": "box", "surface": "dining_table",
                        "zone": "zone", "upright": False}]]
    state = State(spec)
    pos = np.array([-0.30, -1.15, 1.0])
    gap = [0.0]
    corners = np.array([[-0.32, 1.24, 0.761], [-0.28, 1.28, 0.95]])
    world = SimpleNamespace(
        time=0.0,
        task=SimpleNamespace(scene=load_scene("kitchen"), boxes=[Box(**spec["boxes"][0])]),
        item_pose=lambda name: (pos.copy(), np.array([1, 0, 0, 0])),
        item_speed=lambda name: 0.0,
        item_velocity=lambda name: np.zeros(6),
        item_gap=lambda name, hands: gap[0],
        item_corners=lambda name: corners,
        link_poses=lambda names: (np.zeros((12, 3)), None),
    )
    state.initialize(world)
    pos[:] = [-0.30, 1.26, 1.0]
    gap[0] = 0.1
    for world.time in (0.0, 0.6, 1.2):
        state.update(world)
    assert not state.completed  # Sliding or teleporting to the destination is insufficient.
    pos[2], gap[0] = 1.1, 0.0
    for world.time in (1.3, 1.5):
        state.update(world)
    assert state.lifted["box"]
    pos[2], gap[0] = 0.85, 0.1
    world.time = 1.6
    state.update(world)
    gap[0], world.time = 0.0, 2.0
    state.update(world)
    assert state.since is None  # Recontact breaks the dwell.
    gap[0] = 0.1
    for world.time in (2.1, 2.59):
        state.update(world)
    assert not state.completed
    world.time = 2.6
    state.update(world)
    assert state.completed


def test_replay_comparison_rejects_state_and_outcome_tampering():
    replay = load_tool("replay_booster_t2")
    original = {"root": [0.0, 0.0, 0.8], "task": {"completed": False}, "steps": 10}
    assert replay.compare(copy.deepcopy(original), original)
    for bad in (
        {**original, "steps": 9},
        {**original, "task": {"completed": True}},
        {**original, "root": [0, 0, 0.81]},
    ):
        assert not replay.compare(bad, original)
    assert not replay.compare(json.loads('{"value": NaN}'), {"value": 0.0})
    native = {"joints": {"hinge": np.float64(0.7)}, "velocity": np.float64(0.0)}
    assert replay.compare(native, json.loads(json.dumps(native)))
    assert not replay.compare(native, {"joints": {"hinge": 0.70002}, "velocity": 0.0})
    assert not replay.compare(True, 1)


def test_replay_retains_task_actor_order_and_rejects_substitution(tmp_path):
    replay = load_tool("replay_booster_t2")
    original = task("t2-can-in-bowl")
    record = {"task": original["id"], "task_name": "t2-household", "seed": original["seed"],
              "execution_mode": original["execution_mode"],
              "task_definition": original["backend_kwargs"]["task_spec"]}
    record = json.loads(json.dumps(record, sort_keys=True))
    assert list(record["task_definition"]["items"]) == ["bowl", "can"]
    with pytest.raises(ValueError, match="original task YAML"):
        replay.load_definition(record)
    source = tmp_path / "task.yaml"
    source.write_text(yaml.safe_dump(original, sort_keys=False))
    assert list(replay.load_definition(record, source)["items"]) == ["can", "bowl"]
    for key, value in (("seed", 1), ("execution_mode", "stepped"), ("id", "isaaclab/other")):
        changed = {**original, key: value}
        source.write_text(yaml.safe_dump(changed, sort_keys=False))
        with pytest.raises(ValueError, match="differs"):
            replay.load_definition(record, source)
    changed = copy.deepcopy(original)
    changed["backend_kwargs"]["task_name"] = "go2-d1-control"
    source.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="differs"):
        replay.load_definition(record, source)
    changed = copy.deepcopy(original)
    changed["backend_kwargs"]["task_spec"]["items"]["can"]["pos"][0] += 0.1
    source.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="differs"):
        replay.load_definition(record, source)
