"""RoboEnv lifecycle bridge for registered Isaac Lab tasks."""

from roboenv_runtime.core.adapter import AdapterBase, register_adapter

from .registry import get_task


@register_adapter
class IsaacLabAdapter(AdapterBase):
    name = "isaaclab"
    owner_thread_required = True

    def __init__(self, *, task_name="go2-d1-control", stop_on_success=False, **kwargs):
        self._set_stop_on_success(stop_on_success)
        task = get_task(task_name)
        self.embodiment = task.embodiment
        self._env = task.create_environment(**kwargs)
        self._env.stop_on_success = self.stop_on_success

    @property
    def environment(self):
        """Robot-specific device bindings access their registered environment here."""
        return self._env

    @property
    def control_interval_s(self):
        return self._env.control_interval

    def advance_control(self):
        self._env.tick()

    @property
    def episode_done(self):
        """Task environments may end naturally before the shared step limit."""
        return bool(getattr(self._env, "done", False))

    def configure_execution_mode(self, mode):
        """Apply the shared world's execution policy to the selected environment."""
        self._env.configure_execution_mode(mode)
        self.execution_mode = mode

    def reset(self, seed):
        self._env.reset(seed)

    def close(self):
        self._env.close()

    def score(self):
        return self._env.score()

    def budget(self):
        return self._env.budget()

    def observe(self):
        return self._env.record()
