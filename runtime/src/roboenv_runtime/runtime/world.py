"""World ownership and a shared execution lane, independent of device types.

The existing simulator adapter is retained as the native backend during the
migration. Devices do not own reset/close, and this object does not own scoring.
"""

import threading
from typing import Protocol


class World(Protocol):
    backend: str
    embodiment: str

    def reset(self, seed: int) -> None: ...

    def close(self) -> None: ...


class WorldSession:
    def __init__(self, native):
        self.native = native
        self.backend = native.name
        self.embodiment = native.embodiment
        self.lock = threading.RLock()
        self.epoch = 0
        self._closed = False

    def reset(self, seed):
        with self.lock:
            if self._closed:
                raise RuntimeError("world is closed")
            self.native.reset(seed)
            self.epoch += 1

    def close(self):
        with self.lock:
            if not self._closed:
                self._closed = True
                self.native.close()
