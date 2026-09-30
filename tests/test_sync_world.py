"""Only active commands advance the shared world; freeze and budgets end them."""

import threading

import pytest
from roboenv.devices.model import DeviceError
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession


class Native:
    name = embodiment = "test"
    control_interval_s = 0.02

    def __init__(self, maximum=3, fail=False):
        self.frames = 0
        self.maximum = maximum
        self.fail = fail
        self.closed = False

    def configure_execution_mode(self, mode):
        self.execution_mode = mode

    def budget(self):
        return Budget(used=self.frames, max=self.maximum)

    def advance_control(self):
        assert not self.closed
        if self.fail:
            raise RuntimeError("physics failed")
        self.frames += 1

    def close(self):
        self.closed = True


def motion(count):
    for _ in range(count):
        yield
    return count


@pytest.mark.parametrize("fail", [False, True])
def test_pending_commands_end_at_budget_or_failure(fail):
    native = Native(fail=fail)
    world = WorldSession(native, mode="stepped")
    for _ in range(5):
        world.advance()
    assert native.frames == 0
    future = world.schedule(motion(10), threading.Event())
    with pytest.raises(Exception, match="physics failed|budget exhausted"):
        world.resolve(world.wait(1))
    assert future.done()
    assert native.frames == (0 if fail else 3)
    assert (world.failure is not None) == fail
    assert not world.pending
    world.close()
    world.close()
    assert native.closed


def test_independent_commands_share_ticks_and_freeze_cancels_without_ticks():
    native = Native(maximum=None)
    world = WorldSession(native, mode="stepped")
    left = world.schedule(motion(3), threading.Event())
    right = world.schedule(motion(2), threading.Event())
    world.resolve(world.wait(0.04))
    assert right.result() == 2
    assert native.frames == 2 and not left.done()
    world.freeze()
    world.poll_motions()
    with pytest.raises(Exception, match="ended"):
        left.result()
    world.advance()
    assert native.frames == 2
    world.close()


@pytest.mark.parametrize("cancel,fail,code", [
    (False, False, "episode_finished"),
    (True, False, "operation_interrupted"),
    (False, True, "world_failed"),
    (True, True, "world_failed"),
])
def test_natural_finish_preserves_failure_and_explicit_cancellation(cancel, fail, code):
    native = Native(maximum=None)
    world = WorldSession(native, mode="stepped")
    event = threading.Event()
    pending = world.schedule(motion(3), event)
    if cancel:
        event.set()
    if fail:
        world.failure = DeviceError(
            "world_failed", "native failure", details={"effects": "unknown"}
        )
    world.finish_episode()
    assert pending.exception().code == code
    assert native.frames == 0 and not world.pending
    assert (world.failure is not None) is fail
    world.close()


def test_unbounded_budget_roundtrip():
    budget = Budget(used=100_000, max=None, unit="action_frames")
    assert budget.max is None and budget.remaining is None and not budget.exhausted
    assert Budget(**budget.as_dict()) == budget
    finite = Budget(used=3, max=3)
    assert finite.exhausted and finite.remaining == 0


def test_concurrent_waits_overlap_and_empty_world_advances():
    native = Native(maximum=None)
    world = WorldSession(native, mode="stepped")
    first, second = world.wait(0.04), world.wait(0.10)
    world.resolve(first)
    assert native.frames == 2 and not second.done()
    world.resolve(second)
    assert native.frames == 5
    world.advance()
    assert native.frames == 5


def test_cancelled_wait_does_not_advance_and_freeze_ends_waiters():
    world = WorldSession(Native(maximum=None), mode="stepped")
    pending = world.wait(1)
    pending.cancel()
    world.advance()
    assert world.time_s == 0
    pending = world.wait(1)
    world.freeze()
    with pytest.raises(Exception, match="ended"):
        pending.result()


def test_realtime_advances_without_wait(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    world = WorldSession(Native(maximum=1), mode="realtime")
    world.advance()
    assert world.time_s == 0.02
    world.advance()
    assert world.time_s == 0.02
    now[0] += 0.02
    world.advance()
    assert world.time_s == 0.04
    assert world.native.frames == 2 and world.failure is None


def test_native_budget_guard_is_distinct_from_a_fault_at_the_last_frame():
    from roboenv.devices.model import DeviceError

    class Guarded(Native):
        def advance_control(self):
            raise DeviceError("episode_finished", "native budget exhausted",
                              details={"effects": "none"})

    world = WorldSession(Guarded(), mode="stepped")
    with pytest.raises(DeviceError, match="budget exhausted") as ended:
        world.resolve(world.wait(1))
    assert ended.value.details["effects"] == "none"
    assert world.failure is None and world.time_s == 0

    class Broken(Native):
        def advance_control(self):
            self.frames = self.maximum
            raise RuntimeError("physics failed at horizon")

    world = WorldSession(Broken(), mode="stepped")
    with pytest.raises(DeviceError, match="physics failed") as failed:
        world.resolve(world.wait(1))
    assert failed.value.details["effects"] == "unknown"
    assert world.failure is not None
