"""One SIMPLE world, cached native verdicts, and owner-only replay evidence."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class SimpleAdapter(AdapterBase):
    name = "simple"
    embodiment = "g1"
    owner_thread_required = True

    def __init__(self, **kwargs):
        from .backend import SimpleEnv

        self._env = SimpleEnv(**kwargs)

    @property
    def control_interval_s(self):
        return 0.02

    def advance_control(self):
        self._env.control.tick()

    def reset(self, seed):
        self._env.reset(seed)

    def close(self):
        self._env.close()

    def goal(self):
        return self._env.task.instruction

    def budget(self):
        return Budget(used=self._env.steps, max=self._env.max_steps, unit="action_frames")

    def score(self):
        return ScoreReport(
            reward=float(self._env.completed),
            raw_score=self._env.reward,
            completed=self._env.completed,
            detail={
                "control_protocol": "simple-g1-metric-wbc/4",
                "official_equivalence": False,
                "arm_gravity_compensation": self._env.arm_gravity_compensation,
            },
        )

    def observe(self):
        return self._env.record()
