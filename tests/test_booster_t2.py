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

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"


def task(name):
    return yaml.safe_load((CATALOG / "tasks/isaaclab" / name / "task.yaml").read_text())


def test_t2_catalogs_and_portable_resources_are_closed():
    paths = sorted((CATALOG / "tasks/isaaclab").glob("t2-*/task.yaml"))
    assert {p.parent.name for p in paths} == {"t2-can-in-bowl", "t2-ordered-reach"}
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
