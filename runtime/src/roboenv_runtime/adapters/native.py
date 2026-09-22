"""Shared lifecycle for native devices whose world owns control and recording."""

from roboenv_runtime.core.adapter import AdapterBase
from roboenv_runtime.core.types import Budget


class NativeDeviceAdapter(AdapterBase):
    @property
    def realtime_interval(self):
        return self._env.control_interval if self._env.realtime else None

    def realtime_tick(self):
        self._env.control.tick()

    def reset(self, seed):
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def budget(self):
        return Budget(used=self._env.control.frame, max=self._env.frame_limit, unit="action_frames")

    def capture(self):
        camera = next(name for name, fact in self._env.device_facts.items()
                      if fact["type"] in {"rgb", "rgbd"})
        return self._env.device_image(camera)[0]

    def observe(self):
        from roboenv_runtime.adapters.native_record import observe

        return observe(self)
