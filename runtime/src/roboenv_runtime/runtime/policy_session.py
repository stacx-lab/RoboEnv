"""Pause one continuous policy rollout after each action for an agent-side predicate.

The original vla_run is invoked exactly once: its controller, cached action chunk
and RNG cadence are preserved. No callable or executable source crosses the wire.
"""

import threading
import time
import uuid


class PolicySession:
    def __init__(self, invoke, *, idle_timeout=120.0):
        self.token = uuid.uuid4().hex
        self.invoke = invoke
        self.idle_timeout = idle_timeout
        self.condition = threading.Condition()
        self.sequence = 0
        self.waiting = False
        self.decision = None
        self.done = False
        self.report = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="roboenv-policy", daemon=True)

    def _until(self):
        with self.condition:
            self.sequence += 1
            self.waiting = True
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: self.decision is not None, self.idle_timeout):
                raise TimeoutError("agent callback timed out while policy was paused")
            stop = self.decision
            self.decision = None
            self.waiting = False
            return stop

    def _run(self):
        try:
            self.report = self.invoke(self._until)
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self.condition:
                self.done = True
                self.waiting = False
                self.condition.notify_all()

    def start(self):
        self.thread.start()
        return self._next(0)

    def _next(self, previous, timeout=600):
        with self.condition:
            if not self.condition.wait_for(lambda: self.done or self.sequence > previous, timeout):
                raise TimeoutError("policy did not reach its next action boundary")
            if self.error:
                raise RuntimeError(self.error)
            return {"protocol": "roboenv/vla-callback/1", "token": self.token,
                    "sequence": self.sequence, "status": "done" if self.done else "paused",
                    "report": self.report if self.done else None}

    def advance(self, sequence, stop):
        with self.condition:
            if self.done:
                return self._next(sequence)
            if sequence != self.sequence or not self.waiting or self.decision is not None:
                raise ValueError("stale or concurrent policy callback decision")
            self.decision = bool(stop)
            self.condition.notify_all()
        return self._next(sequence)

    def close(self):
        # Wait only for an action boundary: never interrupt halfway through a physical step.
        deadline = time.monotonic() + 30
        with self.condition:
            self.condition.wait_for(lambda: self.waiting or self.done,
                                    max(0, deadline - time.monotonic()))
            if not self.done:
                self.decision = True
                self.condition.notify_all()
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise TimeoutError("policy failed to stop at an action boundary")
