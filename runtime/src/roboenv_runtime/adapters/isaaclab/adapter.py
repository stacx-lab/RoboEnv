"""Go2–D1 device interface with environment-owned clock and verdict."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class IsaacLabAdapter(AdapterBase):
    name = "isaaclab"
    owner_thread_required = True

    def __init__(self, **kwargs):
        from .go2_d1.environment import Go2Env

        self._env = Go2Env(**kwargs)

    @property
    def control_interval_s(self):
        return self._env.control_interval

    def advance_control(self):
        self._env.tick()

    def configure_execution_mode(self, mode):
        """Apply the shared world's mode to the Go2 practice horizon."""
        self._env.configure_execution_mode(mode)

    def reset(self, seed):
        self._env.reset(seed)

    def close(self):
        self._env.close()

    def goal(self):
        return self._env.instruction

    def score(self):
        return ScoreReport(
            reward=float(self._env.completed),
            raw_score=float(self._env.completed),
            completed=self._env.completed,
        )

    def budget(self):
        return Budget(used=self._env.steps, max=self._env.max_steps, unit="action_frames")

    def observe(self):
        return self._env.record()
