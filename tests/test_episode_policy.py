"""A task's episode policy on the adapters whose world advances one control interval at a time.

stop_on_success is taken by the adapter and never reaches the native simulator; a finite budget
(the task's max_steps or the simulator's own horizon) and a latched success end the episode with
a sealed verdict after the interval, through the real worker, instead of failing the world.
"""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.libero import backend as libero_backend
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
from roboenv_runtime.adapters.simple import backend as simple_backend
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.runtime.worker import SimWorker


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def test_libero_and_simple_take_the_policy_and_never_pass_it_to_the_native(monkeypatch):
    monkeypatch.setattr(libero_backend, "LiberoEnv", Recorder)
    monkeypatch.setattr(simple_backend, "SimpleEnv", Recorder)
    libero = LiberoAdapter("libero_10", 3, stop_on_success=True, init_state_index=0, max_steps=50)
    assert libero._env.args == ("libero_10", 3)
    assert libero._env.kwargs == {"init_state_index": 0, "max_steps": 50}
    simple = SimpleAdapter(stop_on_success=True, sim_mode="mujoco")
    assert simple._env.kwargs == {"sim_mode": "mujoco"}
    assert libero.stop_on_success and simple.stop_on_success
    for adapter in (LiberoAdapter, SimpleAdapter):
        with pytest.raises(ValueError, match="stop_on_success must be a boolean"):
            adapter(stop_on_success="yes")


@pytest.mark.parametrize(
    "adapter_type,module,native",
    [
        (RoboTwinAdapter, "robotwin", "RoboTwinDevices"),
        (BehaviorAdapter, "behavior", "BehaviorDevices"),
        (RoboDojoAdapter, "robodojo", "RoboDojoDevices"),
    ],
)
def test_native_adapters_take_the_policy(monkeypatch, adapter_type, module, native):
    monkeypatch.setitem(
        sys.modules,
        f"roboenv_runtime.adapters.{module}.devices",
        SimpleNamespace(**{native: Recorder}),
    )
    adapter = adapter_type("task", stop_on_success=True, max_steps=10000)
    assert "stop_on_success" not in adapter._env.kwargs
    assert adapter._env.kwargs["max_steps"] == 10000
    assert adapter.stop_on_success
    assert not adapter_type("task").stop_on_success
    with pytest.raises(ValueError, match="stop_on_success must be a boolean"):
        adapter_type("task", stop_on_success="yes")


class Native:
    """A CPU double of the LIBERO native: action frames and a success latched from one frame."""

    control_protocol = "absolute_joint_position"
    env = SimpleNamespace(control_freq=20)
    sim = SimpleNamespace(get_state=lambda: np.zeros(2))

    def __init__(self, limit, success_at):
        self.frame, self.limit, self.success_at = 0, limit, success_at
        self.configured_max_steps = self._max_steps = limit
        self.initialization, self.command_log = {"init_state_index": 0}, []

    def tick(self):
        if self._max_steps is not None and self.frame >= self._max_steps:
            raise RuntimeError("LIBERO action frame budget exhausted")
        self.frame += 1

    def get_step_budget(self):
        return {"used": self.frame, "max": self._max_steps}

    def task_completed(self):
        return self.success_at is not None and self.frame >= self.success_at

    def goal_text(self):
        return "toy"

    def render(self, camera=None):
        return None


def libero(limit, success_at, stop):
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    adapter._env = Native(limit, success_at)
    adapter._take_policy(stop)
    return adapter


@pytest.mark.parametrize(
    "stop,limit,success_at,bounded",
    [(True, None, None, True), (False, 50, None, True), (False, None, 2, False)],
)
def test_a_finite_budget_or_stop_on_success_bounds_the_episode(stop, limit, success_at, bounded):
    assert libero(limit, success_at, stop).bounded_episodes is bounded


def worker(tmp_path, adapter):
    return SimWorker(adapter, task_id="libero/10_3", seed=0, session="test", run_dir=tmp_path,
                     reset_adapter=False, execution_mode="stepped")


@pytest.mark.parametrize(
    "stop,limit,success_at,frames,reason",
    [
        (True, None, 3, 3, "success"),  # ends at the first successful interval
        (False, 4, None, 4, "step_limit"),  # a finite budget ends it without success
        (False, 10, 2, 5, None),  # success alone does not end it without the policy
        (False, None, None, 5, None),  # nothing bounds it
    ],
)
def test_the_world_ends_the_episode_after_the_interval(tmp_path, stop, limit, success_at, frames,
                                                       reason):
    adapter = libero(limit, success_at, stop)
    owner = worker(tmp_path, adapter)
    try:
        # A wait of five intervals, cut short when the episode ends inside it. It used to fail
        # the world at a budget: every later request, the final submit included, returned
        # world_failed and the trial had no verdict.
        wait = owner.world.wait(5 / 20)
        if reason is None:
            assert owner.world.resolve(wait)["elapsed_s"] == pytest.approx(5 / 20)
        else:
            with pytest.raises(Exception, match="world execution ended"):
                owner.world.resolve(wait)
        assert owner.world.failure is None and not owner.world.pending
        status = owner.simulation_status(include_verdict=True)
        assert status["terminal"] is (reason is not None)
        assert status["reason"] == reason and adapter._env.frame == frames
        if reason is not None:
            assert status["success"] is (reason == "success")
            stopped = owner.dispatch("simulation.stop", {})
            assert stopped["reason"] == reason and stopped["success"] is (reason == "success")
            assert owner.dispatch("submit", {})["completed"] is (reason == "success")
            with pytest.raises(Exception, match="frozen"):
                owner.world.schedule(iter(()), threading.Event())
    finally:
        owner.close()


def test_a_wait_that_reaches_the_last_frame_returns_before_the_episode_ends(tmp_path):
    adapter = libero(4, None, False)
    owner = worker(tmp_path, adapter)
    try:
        assert owner.world.resolve(owner.world.wait(4 / 20))["elapsed_s"] == pytest.approx(4 / 20)
        status = owner.simulation_status(include_verdict=True)
        assert status["terminal"] and status["reason"] == "step_limit" and adapter._env.frame == 4
    finally:
        owner.close()
