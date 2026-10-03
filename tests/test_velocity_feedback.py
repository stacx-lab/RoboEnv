"""Public base feedback keeps body measurements without world localization."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env
from roboenv_runtime.adapters.simple.backend import SimpleEnv
from roboenv_runtime.bindings.backends.isaaclab.go2_d1 import read_robot_state


def test_behavior_base_read_has_body_twist_without_global_pose():
    world = BehaviorDevices.__new__(BehaviorDevices)
    world.control = SimpleNamespace(frame=3)
    world.robot = SimpleNamespace(
        get_position_orientation=lambda: ([8, 9, 0], [0, 0, 0, 1]),
        get_linear_velocity=lambda: [1, 2, 3],
        get_angular_velocity=lambda: [4, 5, 6],
    )
    assert world.device_read("base") == {
        "velocities": [1, 2, 6], "faults": [], "timestamp": .1,
    }


@pytest.mark.parametrize("failure", [None, "robot_fallen", "task_failed"])
def test_go2_base_keeps_height_and_physical_faults(failure):
    world = Go2Env.__new__(Go2Env)
    world.native = SimpleNamespace(
        steps=5, task=SimpleNamespace(failure=failure),
        root_pose=lambda: np.array([8, 9, .3, 0, 0, 0, 1]), array=np.asarray,
        robot=SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=[1, 2, 3], root_ang_vel_b=[4, 5, 6],
        )),
    )
    assert read_robot_state(world, "base") == {
        "height_m": .3, "velocities": [1, 2, 3, 4, 5, 6], "timestamp": .1,
        "faults": [{"code": "fallen"}] if failure == "robot_fallen" else [],
    }


def test_simple_base_keeps_pelvis_height_and_parent_relative_torso(monkeypatch):
    def velocity(model, data, kind, body, output, local):
        assert local == 1
        output[:] = [4, 5, 6, 1, 2, 3]

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(
        mj_objectVelocity=velocity, mjtObj=SimpleNamespace(mjOBJ_XBODY=1),
    ))
    world = SimpleEnv.__new__(SimpleEnv)
    world.steps = 5
    world.mjmodel = SimpleNamespace(body=lambda name: SimpleNamespace(id=0))
    world.mjdata = SimpleNamespace()
    world.model = {"parts": {"waist": {"joints": []}}}
    world.ids = {"waist": []}
    world._pose = lambda name: [8, 9, .7 if name == "pelvis" else .9, 0, 0, 0, 1]
    state = world.read_device("base")
    assert "pose" not in state
    assert state["height_m"] == .7
    assert state["velocities"] == [1, 2, 3, 4, 5, 6]
    assert state["timestamp"] == .1
    np.testing.assert_allclose(state["frame_poses"]["torso"], [0, 0, .2, 0, 0, 0, 1])
