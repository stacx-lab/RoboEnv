"""One task schema and effective clock policy across every simulator."""

from dataclasses import asdict
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


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "isaaclab"])
@pytest.mark.parametrize("mode", [None, "stepped"])
def test_catalog_resolution_preserves_task_and_clock(tmp_path, backend, mode):
    task = next(
        task for task, _ in load_catalog(ROOT / "catalog/tasks").values() if task.backend == backend
    )
    value = asdict(task)
    if mode is None:
        value.pop("execution_mode")
    else:
        value["execution_mode"] = mode
    source = tmp_path / "task.yaml"
    source.write_text(yaml.safe_dump(value))
    resolved = resolve_task(source, catalog=ROOT / "catalog")
    assert resolved.task.execution_mode == (mode or "realtime")
    serialized = tmp_path / "resolved.yaml"
    serialized.write_text(yaml.safe_dump(asdict(resolved.task)))
    assert Task.load(serialized) == resolved.task
    assert resolved.bindings.devices
    assert resolved.task.id == task.id


def test_task_rejects_unknown_fields():
    task = asdict(resolve_task("libero/bowl-onto-plate", project=ROOT).task)
    with pytest.raises(ValueError, match="environment task fields"):
        Task.from_dict({**task, "agent_options": {"max_turns": 5}})


def test_world_applies_mode_before_reset_and_preserves_configured_horizon():
    task = Task.from_dict(
        {
            "id": "fakeshell/clock",
            "backend": "fakeshell",
            "backend_task": "clock",
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
