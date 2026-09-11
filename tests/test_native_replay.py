"""Native shutdown cannot erase a replay verdict or hide a failed comparison."""

import json
from types import SimpleNamespace

import pytest
from helpers import load_tool


@pytest.mark.parametrize("mismatch", [False, True])
def test_replay_persists_result_before_native_shutdown_exits(monkeypatch, tmp_path, mismatch):
    tool = load_tool("replay_native_devices")
    state = {"pose": [0.0]}
    budget = {"used": 0, "max": 1}
    native = SimpleNamespace(
        implementation_identity={"version": "fixture"},
        initialization={"requested_seed": 0},
        initial_state=state,
        owner_state=lambda: {"pose": [1.0]} if mismatch else state,
        control=SimpleNamespace(frame=0, targets={"arm": [0.0]}, realtime=False),
        task_completed=lambda: False,
    )

    def close():
        raise SystemExit(0)

    adapter = SimpleNamespace(
        _env=native, embodiment="fixture", reset=lambda seed: None, close=close,
        budget=lambda: SimpleNamespace(as_dict=lambda: budget),
    )
    monkeypatch.setattr(tool, "make_adapter", lambda *args, **kwargs: adapter)
    record = {
        "schema": "roboenv/native-device-episode/1", "backend": "robodojo",
        "budget": budget, "environment": {}, "initialization": native.initialization,
        "implementation": native.implementation_identity, "embodiment": "fixture",
        "initial_state": state, "commands": [], "state": state,
        "control_state": {"targets": native.control.targets, "gripper_holds": {}},
        "native_success": False,
    }
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exit_info:
        tool.replay(record, out=out)
    assert exit_info.value.code == 0
    report = json.loads(out.read_text())
    assert report["passed"] is not mismatch
    if mismatch:
        assert "Replay state differs" in report["error"]
    else:
        assert report["record_sha256"] == tool.digest(record)
