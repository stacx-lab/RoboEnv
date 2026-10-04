"""RoboDojo's graded score is reported on its native 0..100 scale, the reward normalised."""

from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter


def adapter(gated, success):
    instance = RoboDojoAdapter.__new__(RoboDojoAdapter)
    reward_manager = SimpleNamespace(_gated_score_lst=gated)
    instance._env = SimpleNamespace(
        task_completed=lambda: success, _env=SimpleNamespace(reward_manager=reward_manager)
    )
    return instance


@pytest.mark.parametrize(
    "gated,success,reward,raw",
    [
        ([10.0], False, 0.1, 10.0),  # one of several graded steps reached
        ([100.0], True, 1.0, 100.0),  # every graded step reached
        ([0.0], False, 0.0, 0.0),
        (None, True, 1.0, 100.0),  # a task without grading is binary
        (None, False, 0.0, 0.0),
    ],
)
def test_reward_is_the_native_score_normalised(gated, success, reward, raw):
    report = adapter(gated, success).score()
    assert (report.reward, report.raw_score, report.completed) == (reward, raw, success)
    assert 0.0 <= report.reward <= 1.0
