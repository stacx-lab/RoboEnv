"""One native owner and shared stepped or realtime execution."""

import threading
import time
from concurrent.futures import Future

from roboenv.devices.model import DeviceError


class WorldSession:
    def __init__(self, native, mode="realtime"):
        if mode not in {"stepped", "realtime"}:
            raise ValueError("execution mode must be stepped or realtime")
        native.configure_execution_mode(mode)
        self.mode = mode
        self.next_tick = time.monotonic()
        self.native = native
        self.backend, self.embodiment = native.name, native.embodiment
        self.lock = threading.RLock()
        self._closed = False
        self._frozen = False
        self.owner_thread_required = bool(getattr(native, "owner_thread_required", False))
        self._owner_thread = threading.get_ident()
        self._motions = []
        self._waiters = []
        self.time_s = 0.0
        self.failure = None

    def _check_thread(self):
        if self.owner_thread_required and threading.get_ident() != self._owner_thread:
            raise RuntimeError("native operations must run on their creating thread")

    def wait(self, duration_s):
        self._check_thread()
        future = Future()
        with self.lock:
            if self._closed or self._frozen or self.failure:
                raise DeviceError("episode_finished", "world is frozen")
            self._waiters.append((future, self.time_s, self.time_s + duration_s))
            self._settle_waiters()
        return future

    def _settle_waiters(self):
        pending, self._waiters = self._waiters, []
        for future, start, end in pending:
            if future.done():
                continue
            if self._closed or self._frozen or self.failure:
                future.set_exception(
                    DeviceError(
                        getattr(self.failure, "code", "episode_finished"),
                        str(self.failure) if self.failure else "world execution ended",
                    )
                )
            elif self.time_s + 1e-9 >= end:
                future.set_result({"time_s": self.time_s, "elapsed_s": self.time_s - start})
            else:
                self._waiters.append((future, start, end))

    def schedule(self, motion, cancel):
        self._check_thread()
        future = Future()
        with self.lock:
            if self._closed or self._frozen or self.failure:
                raise DeviceError("episode_finished", "world is frozen")
            self._resume((future, motion, cancel))
        return future

    def _resume(self, job, *, error=None):
        future, motion, cancel = job
        try:
            if cancel.is_set():
                error = DeviceError("operation_interrupted", "device motion interrupted by stop")
            if error is not None:
                motion.throw(error)
            else:
                next(motion)
            self._motions.append(job)
        except StopIteration as done:
            future.set_result(done.value)
        except BaseException as exc:
            future.set_exception(exc)

    def poll_motions(self):
        """Cancel pending work without requesting another physics step."""
        self._check_thread()
        with self.lock:
            pending, self._motions = self._motions, []
            for job in pending:
                if job[2].is_set() or self._frozen or self.failure:
                    self._resume(
                        job, error=DeviceError("operation_interrupted", "world execution ended")
                    )
                else:
                    self._motions.append(job)

    @property
    def pending(self):
        return bool(self._motions or self._waiters)

    def advance(self):
        self._check_thread()
        with self.lock:
            self.poll_motions()
            if self._closed or self._frozen or self.failure:
                return
            self._settle_waiters()
            if self.mode == "stepped" and not self._waiters:
                return
            if self.mode == "realtime":
                now = time.monotonic()
                if now < self.next_tick:
                    return
                self.next_tick += self.native.control_interval_s
            pending, self._motions = self._motions, []
            error = None
            try:
                if self.mode == "stepped" and self.native.budget().exhausted:
                    raise DeviceError("episode_finished", "native episode budget exhausted")
                self.native.advance_control()
                self.time_s += self.native.control_interval_s
            except Exception as exc:
                self.failure = error = exc
            for job in pending:
                self._resume(job, error=error)
            self.poll_motions()
            self._settle_waiters()

    def resolve(self, value):
        if not isinstance(value, Future):
            return value
        while not value.done():
            self.advance()
            if self.mode == "realtime" and not value.done():
                time.sleep(min(0.01, max(0, self.next_tick - time.monotonic())))
        return value.result()

    def freeze(self):
        with self.lock:
            self._frozen = True
            self._settle_waiters()

    def reset(self, seed):
        self._check_thread()
        with self.lock:
            if self._closed:
                raise RuntimeError("world is closed")
            self.freeze()
            self.poll_motions()
            self.native.reset(seed)
            self.time_s = 0.0
            self._frozen = False
            self.failure = None
            self.next_tick = time.monotonic()

    def close(self):
        self._check_thread()
        with self.lock:
            self.freeze()
            self.poll_motions()
            if not self._closed:
                self._closed = True
                self.native.close()
