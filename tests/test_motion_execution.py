"""Clock, admission and terminal-reference contracts without a simulator engine."""

from types import SimpleNamespace

import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService
from roboenv_runtime.bindings.motion import control_binding
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession


class Engine:
    name = embodiment = "fixture"
    control_interval_s = 0.01
    owner_thread_required = True

    def __init__(self):
        self.frame = 0
        self.positions = {name: 0.0 for name in ("left", "right")}
        self.targets = dict(self.positions)
        self.events = []
        self.fault = None

    def configure_execution_mode(self, mode):
        self.execution_mode = mode

    def budget(self):
        return Budget(used=self.frame, max=1000)

    def advance_control(self):
        self.frame += 1
        for name, target in self.targets.items():
            self.positions[name] += max(-0.01, min(0.01, target - self.positions[name]))

    def close(self):
        pass


def fixture(mode="stepped", *, period=0.01, kind="joint", control=None):
    native = Engine()
    native.control_interval_s = period
    world = WorldSession(native, mode)

    @control_binding
    def bind(spec, root, context):
        name = spec.entity
        facts = {
            "model": {
                "joints": [{"name": "j", "type": "revolute", "limits": {"min": -2, "max": 2}}]
            },
            "control": control or {"mode": "position"},
            "feedback": {"velocities": True},
            "stop": {"behavior": "hold", "affected_devices": [name]},
        }

        def write(args, _):
            native.targets[name] = args["target"][0]
            native.events.append((native.frame, name, "write", native.targets[name]))

        def read(args, _):
            return {
                "positions": [native.positions[name]],
                "velocities": [0],
                "efforts": [],
                "faults": [{"code": "fault"}] if native.fault == name else [],
                "sample_time_s": native.frame * period,
            }

        def stop(args, _):
            native.targets[name] = native.positions[name]
            native.events.append((native.frame, name, "stop", native.targets[name]))
            return {"stopped": True, **facts["stop"]}

        return BoundDevice(
            DeviceDefinition(name, kind, facts),
            {
                "write": Action(write),
                "read": Action(read),
                "stop": Action(stop),
            },
            implementation="fixture",
            binding={"entity": name},
        )

    devices = [
        bind(
            SimpleNamespace(entity=name),
            None,
            BindingContext("fixture", "fixture", native, world=world),
        )
        for name in native.positions
    ]
    service = DeviceService(
        devices,
        execution_lock=world.lock,
        motion_scheduler=world.schedule,
        resolve=world.resolve,
        flush=world.poll_motions,
    )
    return native, world, service


def submit(service, name, action="write", **args):
    return service.submit({"device": name, "action": action, "args": args})


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
