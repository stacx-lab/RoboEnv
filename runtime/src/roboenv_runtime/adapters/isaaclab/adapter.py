"""Go2 device demonstration, with environment-owned clock and verdict."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class IsaacLabAdapter(AdapterBase):
    name = "isaaclab"
    embodiment = "go2"

    def __init__(self, **kwargs):
        from .backend import IsaacLabEnv

        self._env = IsaacLabEnv(**kwargs)

    def reset(self, seed):
        self._env.reset(seed)

    def close(self):
        self._env.close()

    def goal(self):
        from .rig import JOINTS

        return f"Reach and hold the Go2 posture {dict(zip(JOINTS, self._env.model['posture'], strict=True))} in radians for 0.5 simulated seconds."

    def score(self):
        completed = self._env.held >= 25
        return ScoreReport(reward=float(completed), raw_score=float(completed), completed=completed)

    def budget(self):
        return Budget(used=self._env.steps, max=500, unit="action_frames")

    def observe(self):
        return {"budget": self.budget().as_dict()}
