"""Clock lifecycle, exact budget boundaries and fail-closed ticking."""

import threading
import time

import pytest
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession


class Native:
    name = "test"
    embodiment = "test"
    realtime_interval = 0.005

    def __init__(self, fail=False, maximum=3):
        self.frames = 0
        self.maximum = maximum
        self.fail = fail
        self.closed = False
        self.ticked = threading.Event()

    def budget(self):
        return Budget(used=self.frames, max=self.maximum)

    def realtime_tick(self):
        assert not self.closed
        if self.fail:
            raise RuntimeError("physics failed")
        self.frames += 1
        self.ticked.set()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_autonomous_clock_ends_at_horizon_or_failure(fail):
    native = Native(fail)
    world = WorldSession(native)
    world.start()
    try:
        assert world._halt.wait(2)
        assert native.frames == (0 if fail else 3)
        assert isinstance(world.failure, RuntimeError) if fail else world.failure is None
    finally:
        world.close()
    assert native.closed and not world._thread.is_alive()
    world.close()


def test_freeze_excludes_ticks_and_close_joins_before_destroying_native():
    native = Native()
    world = WorldSession(native)
    world.start()
    assert native.ticked.wait(2)
    with world.lock:
        world.freeze()
        frames = native.frames
    time.sleep(0.03)
    assert native.frames == frames
    world.close()
    assert not world._thread.is_alive()


def test_unbounded_clock_keeps_advancing_until_closed():
    native = Native(maximum=None)
    world = WorldSession(native)
    world.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with world.lock:
                if native.frames > 6:
                    break
            time.sleep(0.005)
        assert native.frames > 6
        assert not world._halt.is_set()
        assert world.failure is None
    finally:
        world.close()
    assert not world._thread.is_alive()


def test_unbounded_budget_roundtrip():
    budget = Budget.coerce({"used": 100_000, "max": None, "unit": "action_frames"})
    assert budget.max is None and budget.remaining is None and not budget.exhausted
    assert Budget.coerce(budget.as_dict()) == budget
    finite = Budget(used=3, max=3)
    assert finite.exhausted and finite.remaining == 0
