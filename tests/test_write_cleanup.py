"""Accepted writes cannot outlive failed feedback unless a replacement owns motion."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from helpers import kinex_python
from motion_fixtures import fixture, submit
from roboenv.devices.execution import request_identity
from test_persistent_driver import transport_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wire(tmp_path):
    module = transport_module()
    transport = module.Transport(tmp_path / "driver.py", device="left")
    transport.config = {"endpoint": "http://127.0.0.1:8123", "session": "fixture", "generation": 0}
    state = SimpleNamespace(calls=[], cancel={"stopped": True},
                            result=lambda identity: {"execution_id": identity, "state": {}})

    def exchange(endpoint, method, path="/", body=None, destination=None):
        request = json.loads(body)
        action = "cancel" if request["op"] == "devices.cancel" else request["args"]["action"]
        state.calls.append((action, request))
        if action == "write":
            data = state.result(request["request_id"])
        elif action == "cancel":
            if isinstance(state.cancel, Exception):
                raise state.cancel
            data = state.cancel
        else:
            data = {"loaded": True}
        return (json.dumps({**request, "ok": True, "result": {"data": data}}) + "\n").encode()

    transport.exchange = exchange
    return transport, state, module


@pytest.mark.parametrize("result", [None, {}, {"execution_id": "wrong", "state": {}},
                                    {"state": {}}, {"state": []}])
def test_malformed_receipt_cancels_outgoing_identity(wire, result):
    transport, state, _ = wire
    state.result = lambda identity: result
    with pytest.raises(ValueError):
        transport.write({}, lambda value: value)
    assert [action for action, _ in state.calls] == ["write", "cancel"]
    assert state.calls[1][1]["args"] == {"execution_id": state.calls[0][1]["request_id"]}
    assert not transport._motions


def bad_reading(value):
    raise ValueError("bad measured state")


@pytest.mark.parametrize("state_value", [None, [], "invalid"])
def test_receipt_requires_measured_state_object(wire, state_value):
    transport, state, _ = wire
    state.result = lambda identity: {"execution_id": identity, "state": state_value}
    with pytest.raises(ValueError, match="measured state"):
        transport.write({}, lambda value: value)
    assert [action for action, _ in state.calls] == ["write", "cancel"]


def test_remote_rejection_does_not_trigger_stop(wire):
    transport, state, module = wire

    def reject(action, args, *, request_id=None):
        state.calls.append(action)
        raise module.WireError("invalid command", "invalid_args", request_id,
                               {"effects": "none"})

    transport.remote = reject
    with pytest.raises(module.WireError) as caught:
        transport.write({}, lambda value: value)
    assert caught.value.code == "invalid_args"
    assert state.calls == ["write"]


def test_failed_cleanup_latches_and_preserves_both_causes_until_load(wire):
    transport, state, module = wire
    state.cancel = OSError("connection lost")
    with pytest.raises(module.WireError) as caught:
        transport.write({}, bad_reading)
    error = caught.value
    assert error.code == "stop_failed"
    assert error.details["effects"] == "unknown"
    assert error.details["validation_error"] == "bad measured state"
    assert "connection lost" in error.details["cleanup_error"]
    with pytest.raises(module.WireError):
        transport.write({}, lambda value: value)
    assert len(state.calls) == 2
    transport.remote("load", {})
    assert transport.write({}, lambda value: value)["execution_id"]


@pytest.mark.parametrize("cancel", [{}, {"stopped": 1}])
def test_cancel_requires_boolean_acknowledgment(wire, cancel):
    transport, state, module = wire
    state.cancel = cancel
    with pytest.raises(module.WireError) as caught:
        transport.write({}, bad_reading)
    assert caught.value.code == "stop_failed"


def test_uncertain_write_is_cancelled_without_replay(wire):
    transport, state, module = wire

    def broken(identity):
        raise OSError("write response lost")

    state.result = broken
    with pytest.raises(module.WireError) as caught:
        transport.write({}, lambda value: value)
    assert caught.value.code == "outcome_unknown"
    assert [action for action, _ in state.calls] == ["write", "cancel"]


@pytest.mark.parametrize("replacement", [False, True])
def test_failed_feedback_uses_service_ownership_even_after_position_hold(wire, replacement):
    transport, state, _ = wire
    native, world, service = fixture()

    def exchange(endpoint, method, path="/", body=None, destination=None):
        request = json.loads(body)
        token = request_identity.set(request["request_id"])
        try:
            result = (service.cancel(request["args"]) if request["op"] == "devices.cancel"
                      else service.call(request["args"]))
        finally:
            request_identity.reset(token)
        return (json.dumps({**request, "ok": True, "result": result}) + "\n").encode()

    transport.exchange = exchange

    def reading(value):
        world.resolve(world.wait(.01))
        if replacement:
            token = request_identity.set("replacement")
            try:
                submit(service, "left", actions=[[-1]], duration_s=.1)
            finally:
                request_identity.reset(token)
        raise ValueError("bad measured state")

    with pytest.raises(ValueError, match="bad measured state"):
        transport.write({"actions": [[1]], "duration_s": .01}, reading)
    stops = [event for event in native.events if event[2] == "stop"]
    assert bool(stops) is not replacement
    if replacement:
        world.resolve(world.wait(.05))
        assert native.positions["left"] < 0
    else:
        assert native.targets["left"] == native.positions["left"]


@pytest.mark.parametrize("robot,name", [
    ("robocasa/robocasa-panda-omron", "base"),
    ("robocasa/robocasa-panda-omron", "arm"),
    ("robocasa/robocasa-panda-omron", "gripper"),
    ("simple/simple-g1", "left_hand"),
])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_exported_actuator_validators_cancel_bad_feedback(robot, name, stop_fails, monkeypatch):
    library = kinex_python()
    if not (library / "kinex").is_dir():
        pytest.skip("Requires Kinex's Python library for exported drivers")
    monkeypatch.syspath_prepend(str(library))
    directory = ROOT / "catalog/robots" / robot / "devices" / name
    profile = yaml.safe_load((directory / "profile.yaml").read_text())
    module = runpy.run_path(str(directory / "driver.py"))
    calls = []

    def remote(action, args, *, request_id=None):
        calls.append((action, args, request_id))
        if action == "cancel_motion":
            if stop_fails:
                raise OSError("cannot stop")
            return {"stopped": True}
        return {"execution_id": request_id, "state": {}}

    monkeypatch.setattr(module["transport"], "remote", remote)
    kind = profile["type"]
    if kind == "velocity":
        args = {"mode": "drive", "velocity": {"vx": .1, "vy": 0, "wz": 0}}
    elif kind == "gripper":
        args = {"actions": [1]}
    else:
        joints = profile["intrinsics"]["model"]["joints"]
        args = {"actions": [[profile["intrinsics"]["home"][joint["name"]] for joint in joints]]}
    result = module["handle"]("write", {**args, "duration_s": .5})
    assert not result["ok"]
    assert [action for action, _, _ in calls] == ["write", "cancel_motion"], result
    assert calls[1][1] == {"execution_id": calls[0][2]}
    if stop_fails:
        assert result["code"] == "stop_failed"
        assert result["details"]["validation_error"]
        assert "cannot stop" in result["details"]["cleanup_error"]
