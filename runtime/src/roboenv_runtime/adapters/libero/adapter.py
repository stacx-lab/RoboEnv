"""Official LIBERO lifecycle, native success, and owner-only evidence."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport

@register_adapter
class LiberoAdapter(AdapterBase):
    name = "libero"
    embodiment = "panda"

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

    @property
    def stop_on_success(self):
        return self._env.stop_on_success

    @property
    def bounded_episodes(self):
        return self._env._max_steps is not None or self.stop_on_success

    def set_step_callback(self, callback):
        import os

        # The worker attaches its interval callback after reset and device setup.
        # Recording is independent of scoring mode: LIBERO uses practice budgets.
        if os.environ.get("ROBOENV_RECORD_EPISODE") == "1" and self._env.owner_recording is None:
            from .recording import OwnerRecording

            self._env.owner_recording = OwnerRecording(self._env, self.run_dir / "recording")
        self._env.on_step = callback

    def freeze_episode(self):
        self._env.terminal = True
        if self._env.owner_recording is not None:
            self._env.owner_recording.close()

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
