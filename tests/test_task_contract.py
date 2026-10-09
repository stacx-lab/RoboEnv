"""One task schema and effective clock policy across every simulator."""

from pathlib import Path

import pytest
import yaml
from roboenv.tasks.catalog import BACKENDS, load_catalog
from roboenv.tasks.model import Task
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.worker import build_adapter
from roboenv_runtime.runtime.world import WorldSession

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", [None, "stepped"])
def test_catalog_resolution_preserves_task_and_clock(tmp_path, backend, mode):
    task = next(
        task for task, _ in load_catalog(ROOT / "catalog/tasks").values() if task.backend == backend
    )
    value = task.as_dict()
    if mode is None:
        value.pop("execution_mode")
    else:
        value["execution_mode"] = mode
    source = tmp_path / "task.yaml"
    source.write_text(yaml.safe_dump(value))
    resolved = resolve_task(source, catalog=ROOT / "catalog")
    assert resolved.task.execution_mode == (mode or "realtime")
    serialized = tmp_path / "resolved.yaml"
    serialized.write_text(yaml.safe_dump(resolved.task.as_dict()))
    assert Task.load(serialized) == resolved.task
    assert resolved.bindings.devices
    assert resolved.task.id == task.id


def test_task_rejects_unknown_fields():
    task = resolve_task("libero/bowl-onto-plate", project=ROOT).task.as_dict()
    with pytest.raises(ValueError, match="environment task fields"):
        Task.from_dict({**task, "agent_options": {"max_turns": 5}})


def test_world_applies_mode_before_reset_and_preserves_configured_horizon():
    task = Task.from_dict(
        {
            "id": "fakeshell/clock",
            "robot": "toy",
            "instruction": "Test clock policy.",
            "backend_kwargs": {"max_steps": 1},
        }
    )
    adapter = build_adapter(task)
    assert isinstance(adapter, FakeShell)
    world = WorldSession(adapter)
    world.reset(0)
    adapter.push(0.1)
    adapter.push(0.1)
    assert adapter.budget().used == 100 and adapter.budget().max is None
    world.close()
    stepped = WorldSession(adapter, mode="stepped")
    stepped.reset(0)
    assert adapter.budget().max == 1
    adapter.push(0.1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        adapter.push(0.1)
    stepped.close()


@pytest.mark.parametrize(
    "kind,actions",
    [
        ("joint", [[0.1, 0.2]]),
        ("dexterous_hand", [[0.1, 0.2]]),
        ("gripper", [1.0]),
        ("suction", [1.0]),
    ],
)
def test_sdk_buffered_writes_use_actions_for_every_actuator_type(kind, actions):
    from types import SimpleNamespace

    from roboenv.devices.client import Device

    calls = []

    def send(action, args):
        calls.append((action, args))
        return {"data": {"execution_id": "accepted"}}

    env = SimpleNamespace(
        _closed=False,
        _transport=SimpleNamespace(device=lambda name: SimpleNamespace(call=send)),
    )
    device = Device(
        env,
        {
            "name": "actuator",
            "type": kind,
            "actions": {"write": {"supported": True}},
        },
    )
    assert device.write(actions, duration_s=0.5) == {"execution_id": "accepted"}
    assert calls == [("write", {"actions": actions, "duration_s": 0.5})]


def test_entire_catalog_uses_canonical_format_and_robot_bindings():
    catalog = load_catalog(ROOT / "catalog/tasks")
    assert len(catalog) == 64
    assert {task.backend for task, _ in catalog.values()} == set(BACKENDS)
    fields = ["id", "robot", "instruction", "seed", "budget_mode", "execution_mode",
              "stop_on_success", "backend_kwargs"]
    for task, path in catalog.values():
        serialized = task.as_dict()
        assert list(serialized) == fields
        assert path.parent.relative_to(ROOT / "catalog/tasks").as_posix() == task.id
        assert path.read_text() == yaml.safe_dump(serialized, sort_keys=False)
        assert Task.from_dict(serialized) == task
        assert task.stop_on_success is (task.id == "isaaclab/roboballet-eight-arm")
        assert "stop_on_success" not in task.backend_kwargs
        resolved = resolve_task(path, project=ROOT)
        assert resolved.bindings.backend == task.backend
        assert resolved.bindings.devices
        assert Task.from_dict(resolved.task.as_dict()) == resolved.task


def test_minimal_task_exposes_defaults_and_serialization_is_independent():
    task = Task.from_dict({"id": "fakeshell/example", "robot": "toy", "instruction": "Test."})
    assert task.backend == "fakeshell"
    assert task.as_dict() == {
        "id": "fakeshell/example", "robot": "toy", "instruction": "Test.",
        "seed": 0, "budget_mode": "official", "execution_mode": "realtime",
        "stop_on_success": False, "backend_kwargs": {},
    }
    task.backend_kwargs["nested"] = {"value": [1]}
    task.as_dict()["backend_kwargs"]["nested"]["value"].append(2)
    assert task.backend_kwargs["nested"]["value"] == [1]


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_task_success_policy_requires_boolean(value):
    with pytest.raises(ValueError, match="stop_on_success must be a boolean"):
        Task.from_dict({"id": "fakeshell/example", "robot": "toy", "instruction": "Test.",
                        "stop_on_success": value})


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stop", [False, True])
def test_worker_routes_shared_policy_separately_from_native_settings(monkeypatch, backend, stop):
    from dataclasses import replace

    from roboenv_runtime.runtime import worker

    original = next(t for t, _ in load_catalog(ROOT / "catalog/tasks").values()
                    if t.backend == backend)
    task = replace(original, stop_on_success=stop)
    received = []
    monkeypatch.setattr(
        worker, "make_adapter", lambda name, **kwargs: received.append((name, kwargs))
    )
    worker.build_adapter(task)
    assert received == [(backend, {**original.backend_kwargs, "stop_on_success": stop})]
    assert "stop_on_success" not in original.backend_kwargs
