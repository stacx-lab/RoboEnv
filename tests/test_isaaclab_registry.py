"""Task selection and lifecycle are independent of the Go2 robot implementation."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from roboenv.tasks.model import Task
from roboenv_runtime.adapters.isaaclab import registry
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.core.types import Budget, ScoreReport
from roboenv_runtime.runtime.worker import build_adapter
from roboenv_runtime.runtime.world import WorldSession


def test_registered_other_robot_uses_shared_adapter_and_clock(monkeypatch):
    calls = []
    state = {"ticks": 0, "mode": None}
    score = ScoreReport(reward=0.25, raw_score=17, detail={"objects": 2})

    def tick():
        state["ticks"] += 1

    environment = SimpleNamespace(
        control_interval=0.1,
        tick=tick,
        configure_execution_mode=lambda mode: state.update(mode=mode),
        reset=lambda seed: calls.append(("reset", seed)),
        close=lambda: calls.append(("close",)),
        score=lambda: score,
        budget=lambda: Budget(
            used=state["ticks"], max=10 if state["mode"] == "stepped" else None,
        ),
        record=lambda: {"ticks": state["ticks"], "robot": "other_robot"},
    )

    class OtherTask:
        name = "other-robot-task"
        embodiment = "other_robot"

        @classmethod
        def create_environment(cls, **kwargs):
            calls.append(("create", kwargs))
            return environment

    module = ModuleType("isaac_registry_fixture")
    module.Task = OtherTask
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(registry.TASKS, OtherTask.name, module.__name__)
    task = Task(
        id="isaaclab/other-layout", robot="isaaclab/other-robot", instruction="Move the object",
        backend_kwargs={"task_name": OtherTask.name, "task_spec": {"target": 2}},
        stop_on_success=True,
    )
    adapter = build_adapter(task)
    try:
        assert adapter.embodiment == "other_robot"
        assert adapter.environment is environment
        assert adapter.stop_on_success is True
        assert calls == [("create", {"task_spec": {"target": 2}})]
        adapter.reset(7)
        world = WorldSession(adapter, "stepped")
        assert world.embodiment == "other_robot"
        world.advance()
        assert state["ticks"] == 0
        waiting = world.wait(0.2)
        world.advance()
        world.advance()
        assert waiting.result()["time_s"] == pytest.approx(0.2)
        assert adapter.observe() == {"ticks": 2, "robot": "other_robot"}
        assert adapter.score() is score
        assert adapter.budget().used == 2
        assert adapter.budget().max == 10
        adapter.configure_execution_mode("realtime")
        assert adapter.budget().max is None
    finally:
        adapter.close()
    assert calls[-2:] == [("reset", 7), ("close",)]


@pytest.mark.parametrize("name", ["missing-task", "go2-d1-missing", None, []])
def test_unknown_task_is_rejected_before_import_or_native_start(monkeypatch, name):
    def forbidden(*args):
        pytest.fail("Unknown task attempted to import an implementation")

    monkeypatch.setattr(registry, "import_module", forbidden)
    with pytest.raises(ValueError, match="Unknown Isaac Lab task"):
        IsaacLabAdapter(task_name=name)


def test_builtin_registration_is_cpu_importable_and_selects_distinct_scenes():
    tasks = [registry.get_task(name) for name in registry.TASKS]
    assert {task.embodiment for task in tasks} == {"go2_d1", "yam", "roboballet_eight_panda"}
    assert len({task.scene_type for task in tasks if task.embodiment == "go2_d1"}) == 3


def test_new_go2_task_owns_asset_configuration_and_reuses_robot(monkeypatch):
    from roboenv_runtime.adapters.isaaclab.robots.go2_d1 import native
    from roboenv_runtime.adapters.isaaclab.robots.go2_d1.task import Go2Task

    class AssetTask(Go2Task):
        name = "go2-asset-task"
        fields = {"asset", "enabled"}

        @staticmethod
        def validate(spec):
            if not isinstance(spec["asset"], str) or type(spec["enabled"]) is not bool:
                raise ValueError("Invalid asset configuration")

    module = ModuleType("isaac_asset_fixture")
    module.Task = AssetTask
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(registry.TASKS, AssetTask.name, module.__name__)
    captured = []

    class Configured(Exception):
        pass

    def construct(*args, **kwargs):
        captured.append(kwargs)
        raise Configured

    monkeypatch.setattr(native, "Native", construct)
    spec = {"asset": "objects/fixture.usd", "enabled": True}
    with pytest.raises(Configured):
        IsaacLabAdapter(task_name=AssetTask.name, task_spec=spec)
    assert captured[0]["task_name"] == AssetTask.name
    assert captured[0]["task_spec"]["asset"] == spec["asset"]
    assert captured[0]["task_spec"]["enabled"] is True
    assert "tcp_in_flange_m" in captured[0]["task_spec"]
    assert "arm" in captured[0]["device_profiles"]
    with pytest.raises(ValueError, match="Invalid asset configuration"):
        IsaacLabAdapter(task_name=AssetTask.name, task_spec={**spec, "enabled": 1})
    assert len(captured) == 1
