"""Go2 device demonstration, with environment-owned clock and verdict."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport
from .rig import CONTROL_DT, JOINTS
from .task import HOLD_INTERVALS, MAX_INTERVALS


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
        target = dict(zip(JOINTS, self._env.model["posture"], strict=True))
        return (f"Reach and hold the Go2 posture {target} in radians "
                f"for {HOLD_INTERVALS * CONTROL_DT:g} simulated seconds.")

    def score(self):
        completed = self._env.held >= HOLD_INTERVALS
        return ScoreReport(reward=float(completed), raw_score=float(completed), completed=completed)

    def budget(self):
        return Budget(used=self._env.steps, max=MAX_INTERVALS, unit="action_frames")

    def observe(self):
        return {"budget": self.budget().as_dict()}
