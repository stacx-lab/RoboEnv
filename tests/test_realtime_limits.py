"""Realtime removes native step caps; stepped retains them and faults still stop."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.behavior import devices as behavior_devices
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.backend import LiberoEnv
from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robodojo.devices import RoboDojoDevices
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.backend import SimpleEnv


@pytest.mark.parametrize("mode", ["realtime", "stepped"])
@pytest.mark.parametrize("adapter_type,env_type", [
    (RoboTwinAdapter, None),
    (RoboDojoAdapter, RoboDojoDevices),
    (BehaviorAdapter, BehaviorDevices),
])
def test_native_device_caps_follow_execution_mode(adapter_type, env_type, mode, monkeypatch):
    if adapter_type is RoboTwinAdapter:
        # Exercise the real device policy without importing SAPIEN or creating a GPU world.
        package = "roboenv_runtime.adapters.robotwin"
        path = (Path(__file__).resolve().parents[1] / "runtime/src"
                / "roboenv_runtime/adapters/robotwin/devices.py")
        with monkeypatch.context() as imports:
            imports.setitem(sys.modules, package + ".backend", SimpleNamespace(
                RoboTwinEnv=object, ROBOTWIN_ROOT="unused",
            ))
            spec = importlib.util.spec_from_file_location(package + "._limits_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            env_type = module.RoboTwinDevices
    env = env_type.__new__(env_type)
    env.frame_limit = env.requested_frame_limit = env.max_steps = 2
    env.task = SimpleNamespace(step_lim=2)
    env._env = SimpleNamespace(step_lim=2, end_flag=[False])
    env.native_step_limit = 2
    env._native_done = False
    adapter = adapter_type.__new__(adapter_type)
    adapter._env = env
    adapter.configure_execution_mode("realtime")
    env._configure_step_limit()
    adapter.configure_execution_mode(mode)
    env._configure_step_limit()
    exhausted = env._exhausted if adapter_type is RoboTwinAdapter else env.exhausted
    env.control_interval = 0.04
    assert adapter.control_interval_s == 0.04
    assert adapter.owner_thread_required
    env.control = DeviceControl(
        {"arm": [(-1, 1)]}, {"arm": [0]}, lambda _: None, lambda: None,
        exhausted=exhausted, interval_s=0.04,
    )
    for _ in range(2):
        adapter.advance_control()
    if mode == "stepped":
        assert adapter.budget().exhausted and adapter.budget().max == 2
        with pytest.raises(Exception, match="exhausted"):
            adapter.advance_control()
    else:
        for _ in range(3):
            adapter.advance_control()
        assert adapter.budget().used == 5 and adapter.budget().max is None
        assert not exhausted()
        if env_type is RoboDojoDevices:
            assert env._env.step_lim == float("inf")
            env._env.end_flag[0] = True
            assert exhausted()
        if env_type is BehaviorDevices:
            env._native_done = True
            assert exhausted()


@pytest.mark.parametrize("mode", ["realtime", "stepped"])
@pytest.mark.parametrize("adapter_type,env_type", [
    (LiberoAdapter, LiberoEnv), (RoboCasaAdapter, RoboCasaEnv),
])
def test_mujoco_ticks_pass_configured_cap_only_in_realtime(adapter_type, env_type, mode):
    env = env_type.__new__(env_type)
    controller = SimpleNamespace(set_goal=lambda *args: None)
    robot = SimpleNamespace(action_dim=1, composite_controller=controller)
    env.env = SimpleNamespace(
        robots=[robot], sim=SimpleNamespace(forward=lambda: None),
        step=lambda action: ({}, 0, False, {}),
    )
    env.wrapper = SimpleNamespace(check_success=lambda: False)
    env._joint_controller = lambda: controller
    env.configured_max_steps = 2
    env._step_count, env._max_steps, env._success = 0, 2, False
    env._gripper_command = 0
    env._stream_commands = {}
    adapter = adapter_type.__new__(adapter_type)
    adapter._env = env
    adapter.configure_execution_mode("realtime")
    adapter.configure_execution_mode(mode)
    env.tick()
    env.tick()
    if mode == "realtime":
        env.tick()
        assert adapter.budget().used == 3 and adapter.budget().max is None
    else:
        assert adapter.budget().exhausted
        if env_type is LiberoEnv:
            with pytest.raises(RuntimeError, match="budget exhausted"):
                env.tick()
        else:
            env.tick()
            assert env._step_count == 2


@pytest.mark.parametrize("mode", ["realtime", "stepped"])
def test_simple_keeps_native_termination_but_removes_realtime_horizon(mode):
    env = SimpleEnv.__new__(SimpleEnv)
    env.configured_max_steps = 2
    env.max_steps, env.steps, env.done = 2, 0, False
    env.frames, env.images = [], {}
    env._step = lambda _: (None, 0, False, False, {})
    adapter = SimpleAdapter.__new__(SimpleAdapter)
    adapter._env = env
    adapter.configure_execution_mode("realtime")
    adapter.configure_execution_mode(mode)
    env.advance({})
    env.advance({})
    if mode == "stepped":
        assert env.done and adapter.budget().exhausted
    else:
        env.advance({})
        assert not env.done and adapter.budget().max is None
        env._step = lambda _: (None, 1, True, False, {})
        env.advance({})
        assert env.done and env.completed


@pytest.mark.parametrize("mode", ["realtime", "stepped"])
def test_behavior_native_timeout_matches_execution_mode(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "omnigibson.tasks.behavior_task", SimpleNamespace(
        BehaviorTask=SimpleNamespace(get_cached_activity_scene_filename=lambda *args: "scene"),
    ))
    monkeypatch.setattr(behavior_devices, "robot_config", lambda: {})
    env = BehaviorDevices.__new__(BehaviorDevices)
    env.activity_name, env.scene_model, env.activity_instance_id = "test", "scene", 0
    env.challenge, env.max_steps = {}, 2
    adapter = BehaviorAdapter.__new__(BehaviorAdapter)
    adapter._env = env
    adapter.configure_execution_mode(mode)
    limit = env.build_config()["task"]["termination_config"]["max_steps"]
    assert (3 >= limit) is (mode == "stepped")
    assert np.isinf(limit) if mode == "realtime" else limit == 2
