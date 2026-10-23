"""RoboCasa lifecycle and benchmark verdicts; devices own the basic control API."""

import random

import numpy as np

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport


@register_adapter
class RoboCasaAdapter(AdapterBase):
    name = "robocasa"
    embodiment = "panda_omron"
    # Native graphics resources share the world's creating thread in both clocks.
    owner_thread_required = True

    def __init__(self, task_name="PickPlaceCounterToCabinet", **kwargs):
        from .backend import RoboCasaEnv

        self._env = RoboCasaEnv(task_name, **kwargs)
        if self._env.predicate_diagnostics:
            self.final_diagnostics = self._predicate_diagnostics

    def _predicate_diagnostics(self):
        if self._env.diagnostics is None:
            return {"complete": False, "error": "Native diagnostics were not initialized"}
        return self._env.diagnostics.snapshot()

    @property
    def control_interval_s(self):
        return 1 / self._env.env.control_freq

    def advance_control(self):
        self._env.tick()

    def configure_execution_mode(self, mode):
        super().configure_execution_mode(mode)
        self._env._max_steps = None if mode == "realtime" else self._env.configured_max_steps

    def reset(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def goal(self):
        return self._env.goal_text()

    def task_context(self):
        from .instructions import task_context

        return task_context(self._env.task_name, self._env.env)

    def score(self):
        completed = self._env.task_completed()
        return ScoreReport(reward=float(completed), raw_score=float(completed), completed=completed)

    def budget(self):
        return Budget(**self._env.get_step_budget(), unit="action_frames")

    @property
    def stop_on_success(self):
        return self._env.stop_on_success

    @property
    def simulation_time_s(self):
        return self._env.get_step_budget()["used"] / self._env.env.control_freq

    def capture(self):
        return self._env.render()

    def observe(self):
        return {
            "objects": {
                name: {"position": pos.tolist(), "quaternion_wxyz": quat.tolist()}
                for name in self._env.list_objects()
                for pos, quat in [self._env.get_object_pose(name)]
            },
            "budget": self.budget().as_dict(),
        }
