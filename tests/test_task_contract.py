"""One task schema and effective clock policy across every simulator and export."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from roboenv.harbor.export import export_task
from roboenv.harbor.taskset import validate_harbor_task
from roboenv.tasks.catalog import BACKENDS, load_catalog
from roboenv.tasks.model import Task
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.worker import build_adapter
from roboenv_runtime.runtime.world import WorldSession

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "isaaclab"])
@pytest.mark.parametrize("mode", [None, "stepped"])
def test_catalog_launch_and_export_share_task_and_clock(tmp_path, backend, mode):
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
    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(b"test-only build input")
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": backend,
                "image": "sha256:" + "a" * 64,
                "legacy_tree_required": False,
            }
        )
    )
    out = tmp_path / "export"
    export_task(
        source, ROOT / "catalog", out, wheel, seed=17, practice=True, deployment_path=deployment
    )
    assert validate_harbor_task(out)["schema_version"] == "1.4"
    exported = Task.load(next((out / "environment/sim/tasks").glob("*.yaml")))
    expected = {**asdict(resolved.task), "seed": 17, "budget_mode": "practice"}
    assert asdict(exported) == expected
    owner = json.loads((out / "environment/sim/environment.json").read_text())
    assert owner["task"] == expected
    verifier = json.loads((out / "tests/task.json").read_text())
    assert (verifier["task_id"], verifier["seed"]) == (task.id, 17)
    public = json.loads((out / "environment/definition/environment.json").read_text())
    assert set(public["task"]) == {"id", "instruction", "context"}
    assert public["task"]["context"] == {}
    instruction = (out / "instruction.md").read_text()
    assert ("no episode step cap" in instruction) is (mode is None)
    assert ("Stepped mode" in instruction) is (mode == "stepped")


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
