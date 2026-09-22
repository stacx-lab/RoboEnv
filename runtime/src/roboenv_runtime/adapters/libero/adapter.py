"""Official LIBERO lifecycle, native success, and owner-only evidence."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport

@register_adapter
class LiberoAdapter(AdapterBase):
    name = "libero"
    embodiment = "panda"
    # Robosuite's GL context stays current on its creating thread. This also
    # keeps synchronous camera reads on that thread, not on RPC handler threads.
    realtime_on_main_thread = True

    def __init__(self, suite_name="libero_spatial", task_id=0, **kwargs):
        from .backend import LiberoEnv

        self._env = LiberoEnv(suite_name, task_id, **kwargs)

    @property
    def realtime_interval(self):
        return 1 / self._env.env.control_freq if self._env.realtime else None

    def realtime_tick(self):
        self._env.tick()

    def reset(self, seed):
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def goal(self):
        return self._env.goal_text()

    def budget(self):
        return Budget(**self._env.get_step_budget(), unit="action_frames")

    def score(self):
        completed = self._env.task_completed()
        return ScoreReport(
            reward=float(completed), raw_score=float(completed), completed=completed,
            detail={"control_protocol": self._env.control_protocol, "native_osc_equivalence": False},
        )

    def capture(self):
        return self._env.render()

    def observe(self):
        return {
            "initialization": self._env.initialization,
            "budget": self.budget().as_dict(),
            "commands": list(self._env.command_log),
            "state": self._env.sim.get_state().flatten().tolist(),
            "native_success": self._env.task_completed(),
        }
