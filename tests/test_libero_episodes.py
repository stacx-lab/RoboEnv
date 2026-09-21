"""Owner termination with the real LIBERO adapter and a small CPU world double."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.backend import LiberoEnv
from roboenv_runtime.runtime.worker import SimWorker


def world(*, limit=None, stop=True, success=False):
    native = LiberoEnv.__new__(LiberoEnv)
    native._max_steps = limit
    native._step_count = 5
    native.stop_on_success = stop
    native.terminal = False
    native.on_step = None
    native.owner_recording = None
    native._success = False
    native.realtime = False
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
    return adapter, native


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_device_stop_scope_and_persistent_gripper_target(direction):
    _, native = world()
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
    assert actions[-1][-1] == direction
    assert native.get_step_budget()["used"] == 6

    native.step({"gripper": [0]})
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
    adapter, native = world(limit=limit, stop=stop, success=success)
    worker = SimWorker(adapter, task_id="libero/10_3", seed=0, session="test",
                       run_dir=tmp_path, reset_adapter=False)
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


def test_recording_starts_after_initialization_and_closes_at_terminal(tmp_path, monkeypatch):
    from roboenv_runtime.adapters.libero import recording

    calls = []

    class Recorder:
        def __init__(self, native, directory):
            calls.append(("start", native._step_count, directory.name))

        def capture(self):
            calls.append("capture")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(recording, "OwnerRecording", Recorder)
    monkeypatch.setenv("ROBOENV_RECORD_EPISODE", "1")
    adapter, native = world(success=True)
    worker = SimWorker(adapter, task_id="libero/10_0", seed=0, run_dir=tmp_path,
                       reset_adapter=False)
    try:
        native._advance(np.zeros(8))
        assert calls[:3] == [("start", 5, "recording"), "capture", "close"]
        assert worker.simulation_status(include_verdict=True)["success"]
    finally:
        worker.close()
