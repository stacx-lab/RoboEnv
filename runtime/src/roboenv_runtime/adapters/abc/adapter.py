"""ABC lifecycle and native score through the shared device adapter."""

import math

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport


@register_adapter
class ABCAdapter(NativeDeviceAdapter):
    name = "abc"
    embodiment = "yam"

    def __init__(self, task_name="put_plastic_bottles_in_bin", *, stop_on_success=False, **kwargs):
        self._set_stop_on_success(stop_on_success)
        from .backend import ABCEnv

        self._env = ABCEnv(task_name, **kwargs)

    def goal(self):
        return self._env.instruction

    def score(self):
        result = self._env.evaluation
        success = self._env.task_completed()
        return ScoreReport(
            reward=float(success),
            raw_score=float(result["reward"]),
            completed=success,
            detail=result,
        )

    def checkpoint(self):
        return self._env.checkpoint()

    def restore(self, token):
        self._env.restore(token)


def frame_budget(interval):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Native control interval must be finite and positive")
    frames = 300.0 / interval
    return math.floor(math.nextafter(frames, math.inf))
