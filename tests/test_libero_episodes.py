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
