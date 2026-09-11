"""World ownership and a shared execution lane, independent of device types.

The existing simulator adapter is retained as the native backend during the
migration. Devices do not own reset/close, and this object does not own scoring.
"""

import math
import threading
import time
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
        self._halt = threading.Event()
        self._thread = None
        self.main_thread_clock = bool(getattr(native, "realtime_on_main_thread", False)) and (
            getattr(native, "realtime_interval", None) is not None
        )
        self._interval = None
        self._deadline = None
        self._owner_thread = None
        self.failure = None

    @property
    def poll_interval(self):
        return min(0.2, self._interval) if self.main_thread_clock and self._interval else 0.2

    def start(self):
        """Start an adapter's optional wall-clock-paced simulation after binding."""
        interval = getattr(self.native, "realtime_interval", None)
        if interval is None:
            return
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("realtime interval must be positive")
        with self.lock:
            if self._closed or self._interval is not None:
                raise RuntimeError("world cannot be started twice")
            self._interval = interval
            if self.main_thread_clock:
                self._owner_thread = threading.get_ident()
                self._deadline = time.monotonic() + interval
                return
            self._thread = threading.Thread(
                target=self._run, args=(interval,), name="roboenv-world", daemon=True
            )
            self._thread.start()

    def _run(self, interval):
        deadline = time.monotonic() + interval
        while not self._halt.wait(max(0, deadline - time.monotonic())):
            self._tick()
            # Best effort 1x speed. A slow simulator never skips physics or
            # runs an unbounded catch-up burst after a slow device read.
            deadline = max(deadline + interval, time.monotonic())

    def poll(self):
        """Service a thread-affine native clock from the worker's execution lane."""
        if not self.main_thread_clock or self._deadline is None or self._halt.is_set():
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("native clock must run on its creating thread")
        if time.monotonic() < self._deadline:
            return
        self._tick()
        self._deadline = max(self._deadline + self._interval, time.monotonic())

    def _tick(self):
        with self.lock:
            if self._halt.is_set():
                return
            try:
                if self.native.budget().exhausted:
                    self._halt.set()
                    return
                self.native.realtime_tick()
            except Exception as exc:
                # Publish failure before releasing the state lock so submit
                # cannot seal a partially advanced, failed physics frame.
                self.failure = exc
                self._halt.set()

    def freeze(self):
        """Linearize the terminal snapshot with ticks; safe while holding lock."""
        with self.lock:
            self._halt.set()

    def stop_clock(self):
        self.freeze()
        if self._thread is not None:
            self._thread.join()

    def reset(self, seed):
        with self.lock:
            if self._closed:
                raise RuntimeError("world is closed")
            self.native.reset(seed)
            self.epoch += 1

    def close(self):
        self.stop_clock()
        with self.lock:
            if not self._closed:
                self._closed = True
                self.native.close()
