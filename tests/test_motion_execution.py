"""Clock, admission and terminal-reference contracts without a simulator engine."""

import threading
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

    def budget(self):
        return Budget(used=self.frame, max=1000)

    def advance_control(self):
        self.frame += 1
        for name, target in self.targets.items():
            self.positions[name] += max(-0.01, min(0.01, target - self.positions[name]))

    def close(self):
        pass


def fixture(mode="stepped", *, period=0.01):
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
            "control": {"mode": "position"},
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
            DeviceDefinition(name, "joint", facts),
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
            BindingContext("fixture", "fixture", native, execution_mode=mode),
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


def test_stepped_completion_revokes_reference_before_peer_advances_world():
    native, world, service = fixture()
    result = world.resolve(submit(service, "left", actions=[[1]], freq_hz=100))["data"]
    assert result["completed_actions"] == 1 and result["reference_state"] == "stopped"
    assert native.positions["left"] == 0.01
    world.resolve(submit(service, "right", actions=[[1]] * 10, freq_hz=100))
    assert native.positions["left"] == 0.01
    assert native.frame == 11


def test_realtime_advances_idle_and_keeps_completed_reference(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: clock[0])
    native, world, service = fixture("realtime")
    future = submit(service, "left", actions=[[1]], freq_hz=100)
    world.advance()
    assert future.result()["data"]["reference_state"] == "target"
    for _ in range(10):
        clock[0] += 0.011
        world.advance()
    assert native.positions["left"] == pytest.approx(0.11)
    before = native.frame
    submit(service, "right", "read")
    assert native.frame == before
    world.freeze()
    clock[0] += 1
    world.advance()
    assert native.frame == before


def test_concurrent_writes_share_ticks_and_validate_before_replacement():
    native, world, service = fixture()
    old = submit(service, "left", actions=[[1]] * 5, freq_hz=100)
    with pytest.raises(DeviceError):
        submit(service, "left", actions=[[0.1], [99]], freq_hz=100)
    assert not old.done()
    left = submit(service, "left", actions=[[0.2]] * 3, freq_hz=100)
    right = submit(service, "right", actions=[[0.3]] * 3, freq_hz=100)
    with pytest.raises(DeviceError, match="interrupted"):
        old.result()
    result = world.resolve(left)["data"]
    assert right.result()["data"]["elapsed_s"] == pytest.approx(0.03)
    assert result["completed_actions"] == 3 and native.frame == 3


def test_replacing_one_device_leaves_independent_peer_running():
    native, world, service = fixture()
    left = submit(service, "left", actions=[[1]] * 10, freq_hz=100)
    right = submit(service, "right", actions=[[1]] * 10, freq_hz=100)
    world.advance()
    new = submit(service, "left", actions=[[-1]] * 3, freq_hz=100)
    with pytest.raises(DeviceError):
        left.result()
    world.resolve(new)
    assert not right.done() and native.positions["right"] > 0.01
    assert native.positions["left"] < 0
    world.resolve(right)


def test_bounded_writes_stop_at_their_own_deadlines():
    native, world, service = fixture()
    futures = [
        submit(service, name, actions=[[1]] * 4, freq_hz=50, for_s=0.035)
        for name in ("left", "right")
    ]
    for future in futures:
        result = world.resolve(future)["data"]
        assert result["completed_actions"] == 1 and result["terminated_by"] == "timeout"
    assert native.frame == 3 and native.targets == native.positions




def test_faulted_write_stops_its_device_without_an_extra_tick():
    native, world, service = fixture()
    future = submit(service, "right", actions=[[1]] * 10, freq_hz=100)
    native.fault = "right"
    world.advance()
    with pytest.raises(DeviceError):
        future.result()
    assert not world.pending and native.frame == 1
    assert native.targets == native.positions


def test_owner_thread_required_for_idle_realtime_tick():
    _, world, _ = fixture("realtime")
    errors = []

    def other():
        try:
            world.advance()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert len(errors) == 1


def test_late_cleanup_cannot_stop_another_clients_replacement():
    from roboenv.devices.execution import request_identity

    native, world, service = fixture()
    token = request_identity.set("old-client")
    try:
        old = submit(service, "left", actions=[[1]] * 10, freq_hz=100)
    finally:
        request_identity.reset(token)
    world.advance()
    token = request_identity.set("new-client")
    try:
        new = submit(service, "left", actions=[[-1]] * 10, freq_hz=100)
    finally:
        request_identity.reset(token)
    assert old.done()
    assert service.cancel({"execution_id": "old-client"})["data"] == {"stopped": False}
    result = world.resolve(new)["data"]
    assert result["completed_actions"] == 10
    assert native.positions["left"] < 0
    assert service.cancel({"execution_id": "new-client"})["data"] == {"stopped": True}


def test_cancellation_overtaking_motion_prevents_late_admission():
    from roboenv.devices.execution import request_identity

    native, _, service = fixture()
    service.cancel({"execution_id": "late"})
    token = request_identity.set("late")
    try:
        with pytest.raises(DeviceError, match="cancelled before admission"):
            submit(service, "left", actions=[[1]], freq_hz=100)
    finally:
        request_identity.reset(token)
    assert native.events == []




@pytest.mark.parametrize("frequency,starts", [(10, [0, 2, 4]), (3, [0, 7, 13]), (8, [0, 3, 5])])
@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_arbitrary_rates_round_cumulative_boundaries(frequency, starts, mode, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: clock[0])
    native, world, service = fixture(mode, period=0.05)
    future = submit(service, "left", actions=[[0.1], [0.2], [0.3]], freq_hz=frequency)
    while not future.done():
        world.advance()
        clock[0] += 0.05
    result = future.result()["data"]
    assert [event[0] for event in native.events if event[2] == "write"] == starts
    assert native.frame == {10: 6, 3: 20, 8: 8}[frequency]
    assert result["elapsed_s"] == pytest.approx(native.frame * 0.05)
    assert result["completed_actions"] == 3


def test_fractional_rate_has_no_drift_and_devices_share_one_world_clock():
    native, world, service = fixture(period=0.05)
    actions = [[0.1], [0.2], [0.3]] * 10
    futures = [
        submit(service, name, actions=actions, freq_hz=3, for_s=10) for name in ("left", "right")
    ]
    result = world.resolve(futures[0])["data"]
    assert futures[1].result()["data"]["completed_actions"] == 30
    left = [event[0] for event in native.events if event[1:3] == ("left", "write")]
    right = [event[0] for event in native.events if event[1:3] == ("right", "write")]
    assert left == right
    assert left[:6] == [0, 7, 13, 20, 27, 33]
    assert result["elapsed_s"] == 10 and native.frame == 200


def test_faster_inputs_skip_samples_but_retain_the_final_reference(monkeypatch):
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: 100.0)
    native, world, service = fixture("realtime", period=0.05)
    future = submit(service, "left", actions=[[0.1], [0.2], [0.3], [0.4]], freq_hz=100)
    world.advance()
    result = future.result()["data"]
    writes = [event[3] for event in native.events if event[2] == "write"]
    assert writes == [0.3, 0.4]
    assert native.targets["left"] == 0.4
    assert result["completed_actions"] == 4 and result["elapsed_s"] == 0.05


def test_sub_tick_chunk_executes_once_and_bounded_chunk_never_exceeds_budget():
    native, world, service = fixture(period=0.05)
    result = world.resolve(submit(service, "left", actions=[[0.1]], freq_hz=1000))["data"]
    assert native.frame == 1 and result["completed_actions"] == 1
    result = world.resolve(submit(service, "left", actions=[[0.2]], freq_hz=3, for_s=1 / 3))["data"]
    assert result["terminated_by"] == "timeout"
    assert result["elapsed_s"] == pytest.approx(0.3)
    assert result["completed_actions"] == 0
    assert native.targets == native.positions


@pytest.mark.parametrize("frequency", [0, -3])
def test_invalid_frequency_does_not_replace_an_active_chunk(frequency):
    native, world, service = fixture(period=0.05)
    old = submit(service, "left", actions=[[0.1], [0.2]], freq_hz=3)
    before = list(native.events)
    with pytest.raises(DeviceError):
        submit(service, "left", actions=[[0.3]], freq_hz=frequency)
    assert not old.done() and native.events == before
    world.resolve(old)
