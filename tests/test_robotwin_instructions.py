"""Source discovery must retain public task semantics without starting a simulator."""

import json
import sys
from types import SimpleNamespace

import pytest
from roboenv.tasks.catalog import _robotwin
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter


def test_robotwin_defaults_to_aloha_without_starting_a_simulator(monkeypatch):
    calls = []

    def devices(task_name, embodiment, **kwargs):
        calls.append((task_name, embodiment))

    monkeypatch.setitem(
        sys.modules, "roboenv_runtime.adapters.robotwin.devices",
        SimpleNamespace(RoboTwinDevices=devices),
    )
    adapter = RoboTwinAdapter()
    assert adapter.embodiment == "aloha_agilex"
    assert calls == [("handover_block", "aloha_agilex")]


def test_piper_is_rejected_before_loading_a_simulator(tmp_path, monkeypatch):
    from roboenv_runtime.adapters.robotwin.rig import describe

    monkeypatch.setitem(sys.modules, "roboenv_runtime.adapters.robotwin.devices", None)
    with pytest.raises(ValueError, match="only the ALOHA AgileX"):
        RoboTwinAdapter(embodiment="piper")
    with pytest.raises(ValueError, match="only the ALOHA AgileX"):
        describe(tmp_path, "piper")


@pytest.mark.parametrize("description", [
    {"full_description": "Pick the microphone and pass it to the other arm."},
    {"full_description": "  Stack red, green and blue in that order.\n"},
    {},
    {"description": "This obsolete field must not replace the pinned source schema."},
    {"full_description": " "},
    {"full_description": ["not a string"]},
    None,
])
def test_robotwin_discovery_uses_explicit_upstream_instruction(tmp_path, description):
    root = tmp_path / "third_party/RoboTwin"
    source = root / "envs/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('Discovery must not execute simulator code')\n")
    tasks = root / "env_cfg/eval/all_tasks.yml"
    tasks.parent.mkdir(parents=True)
    tasks.write_text("tasks: [example]\n")
    if description is not None:
        path = root / "description/task_instruction/example.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(description))
    issues = []
    entries = list(_robotwin(tmp_path, issues))
    text = (description or {}).get("full_description")
    if isinstance(text, str) and text.strip():
        assert len(entries) == 1
        task, _ = entries[0]
        assert task.robot == "robotwin/robotwin-aloha-agilex"
        assert task.id == "robotwin/example-aloha-agilex"
        assert task.backend_kwargs == {"task_name": "example", "embodiment": "aloha_agilex"}
        assert all(task.instruction == text.strip() for task, _ in entries)
        assert all(task.backend == "robotwin" for task, _ in entries)
        assert not issues
    else:
        assert not entries
        assert issues == [{
            "backend": "robotwin", "task": "example",
            "reason": "task lacks the upstream full_description instruction",
        }]


@pytest.mark.parametrize("tag,side", [(0, "left"), (1, "right")])
def test_bottle_context_uses_sampled_assignment_after_reset(tmp_path, monkeypatch, tag, side):
    monkeypatch.setattr("roboenv_runtime.adapters.robotwin.adapter.robotwin_root", lambda: tmp_path)
    for model, color in [(16, "green"), (7, "blue")]:
        path = tmp_path / "description/objects_description/001_bottle" / f"base{model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"raw_description": "bottle", "seen": [f"{color} bottle"]}))
    adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
    task = SimpleNamespace(qpose_tag=tag, model_id=16, info={"info": {}},
                           private_position=[1, 2, 3])
    adapter._env = SimpleNamespace(task_name="adjust_bottle", task=task)
    assert adapter.task_context() == {"assigned_arm": side, "target_side": side,
                                      "bottle": {"name": "bottle", "description": "green bottle"}}
    task.qpose_tag = 1 - tag
    task.model_id = 7
    other = "right" if side == "left" else "left"
    assert adapter.task_context() == {"assigned_arm": other, "target_side": other,
                                      "bottle": {"name": "bottle", "description": "blue bottle"}}


@pytest.mark.parametrize("task_name", ["place_a2b_left", "place_a2b_right"])
def test_object_roles_use_sampled_model_descriptions(tmp_path, monkeypatch, task_name):
    monkeypatch.setattr("roboenv_runtime.adapters.robotwin.adapter.robotwin_root", lambda: tmp_path)
    for name, model, text in [("086_woodenblock", 4, "yellow cylinder"),
                               ("057_toycar", 2, "teal toy car")]:
        path = tmp_path / "description/objects_description" / name / f"base{model}.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"raw_description": name, "seen": [text]}))
    task = SimpleNamespace(selected_modelname_A="086_woodenblock", selected_model_id_A=4,
                           selected_modelname_B="057_toycar", selected_model_id_B=2,
                           info={"info": {}}, private_position=[1, 2, 3])
    adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
    adapter._env = SimpleNamespace(task_name=task_name, task=task)
    assert adapter.task_context() == {"objects": {
        "A": {"name": "086_woodenblock", "description": "yellow cylinder"},
        "B": {"name": "057_toycar", "description": "teal toy car"},
    }}
    task.selected_modelname_A, task.selected_modelname_B = (
        task.selected_modelname_B, task.selected_modelname_A)
    task.selected_model_id_A, task.selected_model_id_B = (
        task.selected_model_id_B, task.selected_model_id_A)
    assert adapter.task_context()["objects"]["A"]["description"] == "teal toy car"


def test_unrelated_task_does_not_expose_native_info():
    adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
    adapter._env = SimpleNamespace(task_name="handover_mic",
                                   task=SimpleNamespace(info={"private_position": [1, 2, 3]}))
    assert adapter.task_context() == {}
