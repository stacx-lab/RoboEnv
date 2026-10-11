"""Keep one RoboEnv world on OmniGibson's batched environment API."""

import os
import sys
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.behavior import backend
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices


@pytest.mark.parametrize("rooms", [["living_room", "garage"], ["kitchen"], []])
def test_task_room_labels_do_not_filter_native_scene(monkeypatch, rooms):
    monkeypatch.setattr(backend, "load_challenge_index", lambda: {
        "activity": {"scene_model": "house", "rooms": rooms},
    })
    world = backend.BehaviorBackend("activity", activity_instance_id=2, max_steps=9000)
    config = world.build_config()
    assert config["scene"]["load_room_types"] is None
    assert config["scene"]["load_room_instances"] is None
    assert config["scene"]["scene_model"] == "house"
    assert config["task"]["activity_name"] == "activity"
    assert config["task"]["activity_instance_id"] == 2
    assert config["task"]["termination_config"]["max_steps"] == 9000


@pytest.mark.parametrize("count", [1, 2])
def test_boot_selects_the_only_scene_and_robot(monkeypatch, count):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.setattr(
        "roboenv.provisioning.gpu.configure_gpu", lambda backend: {"vulkan_index": 3},
    )
    world = backend.BehaviorBackend("turning_on_radio", scene_model="house")
    robot = object()
    configs = []
    env = SimpleNamespace(num_envs=count, robots=[[robot]] * count)
    monkeypatch.setattr(backend, "fix_cache_env", lambda: None)
    launcher = SimpleNamespace(DEFAULT_LAUNCHER_CONFIG={"extra_args": []})
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=launcher))
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(
        Environment=lambda configs: (configs_seen(configs) or env)
    ))

    def configs_seen(config):
        assert os.environ["OMNIGIBSON_GPU_ID"] == "0"
        assert os.environ["OMNIGIBSON_RENDER_GPU"] == "3"
        assert os.environ["OMNI_KIT_ACCEPT_EULA"] == "YES"
        assert "--/app/hydraEngine/waitIdle=true" in launcher.DEFAULT_LAUNCHER_CONFIG["extra_args"]
        configs.append(config)

    if count == 1:
        world.boot()
        assert world.robot is robot
        world.boot()
    else:
        with pytest.raises(ValueError, match="one scene with one robot"):
            world.boot()
    assert len(configs) == 1
    assert configs[0]["env"]["num_envs"] == 1
    assert launcher.DEFAULT_LAUNCHER_CONFIG == {"extra_args": []}


@pytest.mark.parametrize("predicate", [True, False])
def test_goal_report_uses_scene_zero_and_preserves_native_verdict(predicate):
    calls = []

    def check(index):
        calls.append(index)
        return False, {"satisfied": [0], "unsatisfied": [1]}

    world = backend.BehaviorBackend("turning_on_radio", scene_model="house")
    world.env = SimpleNamespace(task=SimpleNamespace(
        _termination_conditions={"predicate": SimpleNamespace(_check_goal_fn=check)}
        if predicate else {},
        success=[False],
        activity_natural_language_goal_conditions=["turn on radio"],
    ))
    report = world.goal_report()
    assert report["success"] is False  # bool([False]) would incorrectly be True
    assert report["conditions"] == ["turn on radio"]
    assert calls == ([0] if predicate else [])
    assert report["unsatisfied"] == ([1] if predicate else [])


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True)])
def test_step_reads_single_scene_termination(terminated, truncated):
    world = BehaviorDevices.__new__(BehaviorDevices)
    actions = []
    world.action = [0.25]
    world._native_success = False
    world.env = SimpleNamespace(step=lambda action: (
        actions.append(action), 0, [terminated], [truncated], {}
    ))
    world.goal_report = lambda: {"success": False}
    world._advance()
    assert actions == [[0.25]]
    assert world._native_done is (terminated or truncated)
    assert world._native_success is False
