"""Official LIBERO lifecycle, native success, and owner-only evidence."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class LiberoAdapter(AdapterBase):
    name = "libero"
    embodiment = "panda"
    # Robosuite's GL context stays current on its creating thread. This also
    # keeps synchronous camera reads on that thread, not on RPC handler threads.
    owner_thread_required = True

    def __init__(self, suite_name="libero_spatial", task_id=0, *, stop_on_success=False, **kwargs):
        self._set_stop_on_success(stop_on_success)
        from .backend import LiberoEnv

        self._env = LiberoEnv(suite_name, task_id, **kwargs)

    @property
    def control_interval_s(self):
        return 1 / self._env.env.control_freq

    def advance_control(self):
        self._env.tick()

    def configure_execution_mode(self, mode):
        super().configure_execution_mode(mode)
        self._env._max_steps = None if mode == "realtime" else self._env.configured_max_steps

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
            reward=float(completed),
            raw_score=float(completed),
            completed=completed,
            detail={
                "control_protocol": self._env.control_protocol,
                "native_osc_equivalence": False,
            },
        )

    @property
    def bounded_episodes(self):
        return self._env._max_steps is not None or self.stop_on_success

    def set_step_callback(self, callback):
        self._env.on_step = callback

    def freeze_episode(self):
        self._env.terminal = True

    @property
    def simulation_time_s(self):
        return self._env.get_step_budget()["used"] / self._env.env.control_freq

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
