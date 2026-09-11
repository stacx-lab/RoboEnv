"""Source discovery must retain public task semantics without starting a simulator."""

import json

import pytest
from roboenv.tasks.catalog import _robotwin


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
        assert len(entries) == 2
        assert {task.robot for task, _ in entries} == {
            "robotwin/robotwin-dual-piper", "robotwin/robotwin-aloha-agilex",
        }
        assert all(task.instruction == text.strip() for task, _ in entries)
        assert all(task.action_abstraction == "devices" for task, _ in entries)
        assert not issues
    else:
        assert not entries
        assert issues == [{
            "backend": "robotwin", "task": "example",
            "reason": "task lacks the upstream full_description instruction",
        }]
