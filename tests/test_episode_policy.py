"""Owner policy on real adapters without importing simulator dependencies."""

import sys
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.runtime.worker import SimWorker

ADAPTERS = [
    (SimpleAdapter, "simple.backend", "SimpleEnv"),
    (LiberoAdapter, "libero.backend", "LiberoEnv"),
    (RoboTwinAdapter, "robotwin.devices", "RoboTwinDevices"),
    (BehaviorAdapter, "behavior.devices", "BehaviorDevices"),
    (RoboDojoAdapter, "robodojo.devices", "RoboDojoDevices"),
]


class Constructor:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class PolicyToy(FakeShell):
    control_interval_s = 0.05
    success_at = None

    def advance_control(self):
        self.used += 1
        if self.success_at is not None and self.used >= self.success_at:
            self.pos[0] = self.goal_x


@pytest.fixture(params=ADAPTERS, ids=lambda item: item[0].name)
def adapter_type(request, monkeypatch):
    adapter, module, name = request.param
    monkeypatch.setitem(sys.modules, f"roboenv_runtime.adapters.{module}",
                        SimpleNamespace(**{name: Constructor}))
    return adapter


@pytest.mark.parametrize("policy", [{}, {"stop_on_success": False}, {"stop_on_success": True}])
def test_policy_is_consumed_and_native_arguments_preserved(adapter_type, policy):
    args = () if adapter_type is SimpleAdapter else ("task",)
    kwargs = {"sim_mode": "mujoco"} if adapter_type is SimpleAdapter else {"max_steps": 20}
    adapter = adapter_type(*args, **kwargs, **policy)
    assert adapter.stop_on_success is policy.get("stop_on_success", False)
    assert adapter._env.args[:len(args)] == args
    assert adapter._env.kwargs == kwargs


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_invalid_policy_is_rejected_before_native_construction(adapter_type, value, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached native construction")

    monkeypatch.setattr(Constructor, "__init__", forbidden)
    with pytest.raises(ValueError, match="stop_on_success must be a boolean"):
        adapter_type(stop_on_success=value)


@pytest.mark.parametrize("mode,stop,success_at,frames,reason", [
    ("stepped", True, 3, 3, "success"),
    ("realtime", True, 7, 7, "success"),
    ("stepped", False, 3, 5, None),
    ("stepped", False, None, 6, "step_limit"),
    ("stepped", False, 6, 6, "success"),
    ("realtime", False, 3, 8, None),
    ("realtime", False, None, 8, None),
])
def test_worker_owns_termination(tmp_path, monkeypatch, mode, stop, success_at, frames, reason):
    now = [0.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    adapter = PolicyToy(stop_on_success=stop, max_budget=6)
    adapter.success_at = success_at
    worker = SimWorker(adapter, task_id="audit", seed=0, run_dir=tmp_path, execution_mode=mode)
    try:
        wait = worker.world.wait(frames * adapter.control_interval_s)
        for _ in range(frames + 2):
            worker.world.advance()
            now[0] += 0.051
            if wait.done():
                break
        assert wait.done()
        assert adapter.budget().used == frames
        status = worker.simulation_status(include_verdict=True)
        assert status["terminal"] is (reason is not None)
        assert status["reason"] == reason
        assert status["capabilities"]["bounded_episodes"] is (stop or mode == "stepped")
        assert adapter.budget().max == (None if mode == "realtime" else 6)
        assert worker.world.failure is None
        if reason is not None:
            assert status["success"] is (reason == "success")
    finally:
        worker.close()
