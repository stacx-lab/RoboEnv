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
    realtime_on_main_thread = True

    def __init__(self, task_name="PickPlaceCounterToCabinet", **kwargs):
        from .backend import RoboCasaEnv

        self._env = RoboCasaEnv(task_name, **kwargs)

    @property
    def realtime_interval(self):
        if getattr(self._env, "realtime", False):
            return 1 / self._env.env.control_freq
        return None

    def realtime_tick(self):
        self._env.tick()

    def reset(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def goal(self):
        return self._env.goal_text()

    def score(self):
        completed = self._env.task_completed()
        return ScoreReport(reward=float(completed), raw_score=float(completed), completed=completed)

    def budget(self):
        return Budget(**self._env.get_step_budget(), unit="action_frames")

    @property
    def stop_on_success(self):
        return self._env.stop_on_success

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
            "objects": {
                name: {"position": pos.tolist(), "quaternion_wxyz": quat.tolist()}
                for name in self._env.list_objects()
                for pos, quat in [self._env.get_object_pose(name)]
            },
            "budget": self.budget().as_dict(),
        }
