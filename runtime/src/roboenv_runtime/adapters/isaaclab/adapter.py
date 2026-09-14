"""G1 device interface with environment-owned clock and verdict."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class IsaacLabAdapter(AdapterBase):
    name = "isaaclab"
    embodiment = "g1"
    realtime_on_main_thread = True

    def __init__(self, **kwargs):
        from .g1 import G1Env

        self._env = G1Env(**kwargs)

    @property
    def realtime_interval(self):
        return self._env.control_interval if self._env.realtime else None

    def realtime_tick(self):
        self._env.control.tick()

    def reset(self, seed):
        self._env.reset(seed)

    def close(self):
        self._env.close()

    def goal(self):
        return (
            "Pick up and place the steering wheel using the G1 humanoid. "
            "Component joint control is a practice interface; the upstream task verdict "
            "is preserved. Realtime runs have no time horizon; synchronous runs retain "
            "the upstream horizon."
        )

    def score(self):
        return ScoreReport(
            reward=float(self._env.completed),
            raw_score=float(self._env.completed),
            completed=self._env.completed,
        )

    def budget(self):
        return Budget(used=self._env.steps, max=self._env.max_steps, unit="action_frames")

    def observe(self):
        return {"budget": self.budget().as_dict()}
