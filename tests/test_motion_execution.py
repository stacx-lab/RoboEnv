"""Clock, admission and terminal-reference contracts without a simulator engine."""

import pytest
from motion_fixtures import fixture, submit
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService


@pytest.mark.parametrize("kind,control,actions,expected,bad", [
    ("gripper", {"mode": "position", "bounds": {"min": 0, "max": 1}, "resolution": 0.1},
     [0.24, 0.76], 0.8, 2),
    ("gripper", {"mode": "direction"}, [-0.3, 0.2], 1, 2),
    ("suction", {"mode": "binary", "off_target": 0, "on_target": 1}, [0, 1], 1, 0.5),
])
def test_flat_end_effector_actions_validate_before_replacing_buffers(
    kind, control, actions, expected, bad
):
    native, world, service = fixture(kind=kind, control=control)
    receipt = submit(service, "left", actions=actions, duration_s=0.1)
    assert receipt.done() and native.frame == 0
    for args in ({"target": [0]}, {"actions": [[0]]}, {"actions": []},
                 {"actions": [True]}, {"actions": [float("nan")]},
                 {"actions": [float("inf")]}, {"actions": [actions[0], bad]}):
        with pytest.raises(DeviceError):
            submit(service, "left", **args, duration_s=0.1)
    world.resolve(world.wait(0.1))
    assert native.targets["left"] == expected
    assert not any(event[2] == "stop" for event in native.events)
    submit(service, "left", actions=[actions[0]], duration_s=0.1)
    world.resolve(world.wait(0.1))
    target = native.targets["left"]
    world.resolve(world.wait(0.1))
    assert native.targets["left"] == target
    assert not any(event[2] == "stop" for event in native.events)


def test_completed_reference_remains_while_a_peer_moves():
    native, world, service = fixture()
    first = submit(service, "left", actions=[[1]], duration_s=0.01)
    assert first.done() and native.frame == 0
    world.resolve(world.wait(0.01))
    submit(service, "right", actions=[[1]] * 10, duration_s=0.1)
    world.resolve(world.wait(0.1))
    assert native.positions["left"] == pytest.approx(0.11)
    assert not any(event[2] == "stop" for event in native.events)


def test_concurrent_writes_and_invalid_replacement():
    native, world, service = fixture()
    submit(service, "left", actions=[[1]] * 5, duration_s=0.05)
    with pytest.raises(DeviceError):
        submit(service, "left", actions=[[0.1], [99]], duration_s=0.05)
    submit(service, "right", actions=[[0.3]] * 3, duration_s=0.03)
    world.resolve(world.wait(0.03))
    assert native.frame == 3
    assert native.positions == {"left": 0.03, "right": 0.03}
    assert len(service._active) == 1


def test_replacement_preserves_peers_without_halting():
    native, world, service = fixture()
    submit(service, "left", actions=[[1]], duration_s=0.1)
    submit(service, "right", actions=[[1]], duration_s=0.1)
    world.resolve(world.wait(0.01))
    submit(service, "left", actions=[[-1]], duration_s=0.03)
    world.resolve(world.wait(0.03))
    assert native.positions["left"] < 0 < native.positions["right"]
    assert not any(event[2] == "stop" for event in native.events)


def test_faulted_background_motion_stops_and_reports_fault():
    native, world, service = fixture()
    submit(service, "right", actions=[[1]], duration_s=0.1)
    native.fault = "right"
    world.resolve(world.wait(0.01))
    assert native.targets == native.positions
    assert service.call({"device": "right", "action": "read", "args": {}})["data"]["faults"]
    with pytest.raises(DeviceError):
        submit(service, "right", actions=[[1]], duration_s=0.1)


def test_late_cleanup_cannot_stop_replacement():
    from roboenv.devices.execution import request_identity

    native, world, service = fixture()
    for identity, goal in (("old", 1), ("new", -1)):
        token = request_identity.set(identity)
        try:
            submit(service, "left", actions=[[goal]], duration_s=0.1)
        finally:
            request_identity.reset(token)
    assert service.cancel({"execution_id": "old"})["data"] == {"stopped": False}
    world.resolve(world.wait(0.1))
    assert native.positions["left"] < 0
    assert service.cancel({"execution_id": "new"})["data"] == {"stopped": True}


def test_cancel_before_admission():
    from roboenv.devices.execution import request_identity

    native, _, service = fixture()
    service.cancel({"execution_id": "late"})
    token = request_identity.set("late")
    try:
        with pytest.raises(DeviceError, match="cancelled before admission"):
            submit(service, "left", actions=[[1]], duration_s=0.1)
    finally:
        request_identity.reset(token)
    assert native.events == []


@pytest.mark.parametrize("duration,starts", [(0.3, [0, 2, 4]), (1, [0, 7, 14]), (0.375, [0, 3, 5])])
def test_fractional_periods_have_no_accumulated_drift(duration, starts):
    native, world, service = fixture(period=0.05)
    for name in ("left", "right"):
        submit(service, name, actions=[[0.1], [0.2], [0.3]], duration_s=duration)
    world.resolve(world.wait(duration))
    for name in ("left", "right"):
        assert [e[0] for e in native.events if e[1] == name] == starts
    assert native.targets == {"left": 0.3, "right": 0.3}
    assert duration - 1e-9 <= world.time_s < duration + 0.05


def test_sub_tick_duration_keeps_final_reference():
    native, world, service = fixture(period=0.05)
    submit(service, "left", actions=[[0.1], [0.2], [0.3]], duration_s=0.001)
    world.resolve(world.wait(0.001))
    assert native.frame == 1
    assert native.targets["left"] == 0.3


def test_world_wait_cancellation_leaves_peer_wait_and_motion_running():
    from roboenv.devices.execution import request_identity
    from roboenv_runtime.bindings.world import clock_actions

    native, world, service = fixture()
    clock = BoundDevice(
        DeviceDefinition("world", "world", {}),
        clock_actions(BindingContext("fixture", "fixture", native, world=world)),
        implementation="clock",
    )
    clocks = DeviceService([clock], execution_lock=world.lock, resolve=world.resolve)
    waits = []
    for identity in ("first", "second"):
        token = request_identity.set(identity)
        try:
            waits.append(
                clocks.submit({"device": "world", "action": "wait", "args": {"duration_s": 0.1}})
            )
        finally:
            request_identity.reset(token)
    submit(service, "left", actions=[[0.2]], duration_s=0.1)
    clocks.cancel({"execution_id": "first"})
    assert waits[0].cancelled() and not waits[1].done()
    world.resolve(waits[1])
    assert native.frame == 10 and native.positions["left"] > 0


def test_write_returns_immediately_and_wait_keeps_background_ownership():
    native, world, service = fixture()
    result = submit(service, "left", actions=[[1]], duration_s=0.1)
    assert result.done() and native.frame == 0
    assert result.result()["data"]["elapsed_s"] == 0
    world.resolve(world.wait(0.03))
    assert service._active
    assert not any(event[2] == "stop" for event in native.events)
    world.resolve(world.wait(0.07))
    assert not service._active


def test_write_rejects_blocking_argument_before_replacing_motion():
    native, world, service = fixture()
    submit(service, "left", actions=[[1]], duration_s=0.1)
    with pytest.raises(DeviceError):
        submit(service, "left", actions=[[-1]], duration_s=0.1, block_s=0)
    world.resolve(world.wait(0.1))
    assert native.targets["left"] == 1
    assert not any(event[2] == "stop" for event in native.events)


@pytest.mark.parametrize("stop_failure", [False, True])
def test_budget_cut_preserves_readings_and_distinguishes_cleanup_failure(stop_failure):
    native, world, service = fixture()
    native.maximum = 3
    native.stop_failure = stop_failure
    submit(service, "left", actions=[[0.1], [0.2], [0.3], [0.4]], duration_s=0.04)
    submit(service, "right", actions=[[0.2]], duration_s=0.1)
    with pytest.raises(DeviceError) as ended:
        world.resolve(world.wait(0.1))
    assert native.frame == 3
    assert ended.value.code == ("stop_failed" if stop_failure else "episode_finished")
    assert (world.failure is not None) == stop_failure
    if not stop_failure:
        assert native.targets == native.positions
        assert not service.call({"device": "left", "action": "read", "args": {}})["data"]["faults"]
        assert not service._active and not world.pending
        with pytest.raises(DeviceError) as rejected:
            world.resolve(submit(service, "left", actions=[[1]], duration_s=0.1))
        assert rejected.value.code == "episode_finished"
        assert rejected.value.details["effects"] == "none"
        assert native.frame == 3


def test_exact_horizon_buffer_and_wait_complete_without_extra_reference():
    native, world, service = fixture()
    native.maximum = 3
    submit(service, "left", actions=[[0.1], [0.2], [0.3]], duration_s=0.03)
    assert world.resolve(world.wait(0.03))["elapsed_s"] == pytest.approx(0.03)
    assert native.frame == 3 and world.failure is None and not world.pending
    assert native.targets["left"] == 0.3
    assert not service._faults
    with pytest.raises(DeviceError, match="frozen"):
        world.wait(0.01)
    world.advance()
    assert native.frame == 3


def test_unknown_native_episode_effects_are_not_certified_as_budget_termination():
    native, world, service = fixture()
    native.write_error = DeviceError("episode_finished", "unknown native effects",
                                     details={"effects": "unknown"})
    with pytest.raises(DeviceError) as failed:
        world.resolve(submit(service, "left", actions=[[0.1]], duration_s=0.1))
    assert failed.value.details["effects"] == "unknown"
    assert world.failure is not None


def test_exhausted_budget_rejects_before_staging_any_reference():
    native, world, service = fixture()
    native.maximum = native.frame = 3
    with pytest.raises(DeviceError) as rejected:
        world.resolve(submit(service, "left", actions=[[0.1]], duration_s=0.1))
    assert rejected.value.code == "episode_finished"
    assert rejected.value.details["effects"] == "none"
    assert not native.events and world.failure is None
    assert native.frame == 3


def test_native_rejection_with_acknowledged_cleanup_records_partial_effects():
    native, world, service = fixture()
    native.targets["left"] = 1  # A previously retained reference.
    native.write_error = DeviceError("episode_finished", "native budget guard",
                                     details={"effects": "none"})
    with pytest.raises(DeviceError) as ended:
        world.resolve(submit(service, "left", actions=[[0.1]], duration_s=0.1))
    # Stopping changed the retained reference even though the write itself was rejected.
    assert ended.value.details["effects"] == "partial"
    assert native.targets["left"] == native.positions["left"]
    assert native.frame == 0 and world.failure is None
