"""Only active commands advance the shared world; freeze and budgets end them."""

import threading

import pytest
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession


class Native:
    name = embodiment = "test"

    def __init__(self, maximum=3, fail=False):
        self.frames = 0
        self.maximum = maximum
        self.fail = fail
        self.closed = False

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
    world = WorldSession(native)
    for _ in range(5):
        world.advance()
    assert native.frames == 0
    future = world.schedule(motion(10), threading.Event())
    with pytest.raises(Exception, match="physics failed|budget exhausted"):
        world.resolve(future)
    assert native.frames == (0 if fail else 3)
    assert world.failure is not None and not world.pending
    world.close()
    world.close()
    assert native.closed


def test_independent_commands_share_ticks_and_freeze_cancels_without_ticks():
    native = Native(maximum=None)
    world = WorldSession(native)
    left = world.schedule(motion(3), threading.Event())
    right = world.schedule(motion(2), threading.Event())
    assert world.resolve(right) == 2
    assert native.frames == 2 and not left.done()
    world.freeze()
    world.poll_motions()
    with pytest.raises(Exception, match="ended"):
        left.result()
    world.advance()
    assert native.frames == 2
    world.close()


def test_unbounded_budget_roundtrip():
    budget = Budget.coerce({"used": 100_000, "max": None, "unit": "action_frames"})
    assert budget.max is None and budget.remaining is None and not budget.exhausted
    assert Budget.coerce(budget.as_dict()) == budget
    finite = Budget(used=3, max=3)
    assert finite.exhausted and finite.remaining == 0
