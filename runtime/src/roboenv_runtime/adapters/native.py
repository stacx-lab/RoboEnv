"""Shared lifecycle for native devices whose world owns control and recording."""

from roboenv_runtime.core.adapter import AdapterBase
from roboenv_runtime.core.types import Budget


class NativeDeviceAdapter(AdapterBase):
    owner_thread_required = True

    def configure_execution_mode(self, mode):
        super().configure_execution_mode(mode)
        self._env.unbounded = mode == "realtime"

    @property
    def native_terminated(self):
        return bool(getattr(self._env, "_native_done", False))

    @property
    def simulation_time_s(self):
        return self._env.control.frame * self._env.control_interval

    def episode_initialization(self):
        # Reuse the state already captured at reset. Native scene/deformable
        # snapshots can exceed the RPC response limit; the owner writes a file.
        return {
            "schema": "roboenv/native-device-episode/1",
            "backend": self.name,
            "environment": self._env.replay_config,
            "initialization": self._env.initialization,
            "initial_state": self._env.initial_state,
            "budget": self.budget().as_dict(),
        }

    def set_step_callback(self, callback):
        self._env.control.on_step = callback

    def freeze_episode(self):
        self._env.control.frozen = True

    @property
    def control_interval_s(self):
        return self._env.control_interval

    def advance_control(self):
        self._env.control.tick()

    def reset(self, seed):
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def budget(self):
        return Budget(used=self._env.control.frame, max=self._env.frame_limit, unit="action_frames")

    def capture(self):
        camera = next(
            name for name, fact in self._env.device_facts.items() if fact["type"] in {"rgb", "rgbd"}
        )
        return self._env.device_image(camera)[0]

    def observe(self):
        from roboenv_runtime.adapters.native_record import observe

        return observe(self)
