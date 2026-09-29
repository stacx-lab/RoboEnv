"""Native shutdown cannot erase a replay verdict or hide a failed comparison."""

import json
from types import SimpleNamespace

import pytest
from helpers import load_tool


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_replay_persists_result_before_native_shutdown_exits(monkeypatch, tmp_path, mismatch, mode):
    tool = load_tool("replay_native_devices")
    state = {"pose": [0.0]}
    budget = {"used": 0, "max": 1 if mode == "stepped" else None}
    native = SimpleNamespace(
        implementation_identity={"version": "fixture"},
        initialization={"requested_seed": 0},
        initial_state=state,
        owner_state=lambda: {"pose": [1.0]} if mismatch else state,
        control=SimpleNamespace(frame=0, targets={"arm": [0.0]}),
        task_completed=lambda: False,
    )

    def close():
        raise SystemExit(0)

    configured = []
    adapter = SimpleNamespace(
        _env=native, embodiment="fixture", reset=lambda seed: None, close=close,
        configure_execution_mode=configured.append,
        budget=lambda: SimpleNamespace(as_dict=lambda: budget),
    )
    monkeypatch.setattr(tool, "make_adapter", lambda *args, **kwargs: adapter)
    record = {
        "schema": "roboenv/native-device-episode/1", "backend": "robodojo",
        "budget": budget, "environment": {}, "initialization": native.initialization,
        "implementation": native.implementation_identity, "embodiment": "fixture",
        "initial_state": state, "commands": [], "state": state,
        "control_state": {"targets": native.control.targets, "gripper_holds": {}},
        "native_success": False, "execution_mode": mode,
    }
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exit_info:
        tool.replay(record, out=out)
    assert exit_info.value.code == 0
    assert configured == [mode]
    report = json.loads(out.read_text())
    assert report["passed"] is not mismatch
    if mismatch:
        assert "Replay state differs" in report["error"]
    else:
        assert report["record_sha256"] == tool.digest(record)


@pytest.mark.parametrize("name", ["native", "simple"])
@pytest.mark.parametrize("mode", [None, "invalid", True])
def test_replay_rejects_missing_or_invalid_mode_before_creating_simulator(monkeypatch, name, mode):
    tool = load_tool(f"replay_{name}_devices")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid mode must be rejected before simulator creation")

    monkeypatch.setattr(tool, "make_adapter" if name == "native" else "SimpleEnv", unexpected)
    schema = "roboenv/native-device-episode/1" if name == "native" else "roboenv/simple-record/3"
    record = {"schema": schema, "backend": "robodojo"}
    if mode is not None:
        record["execution_mode"] = mode
    with pytest.raises(ValueError, match="execution_mode"):
        tool.replay(record)
