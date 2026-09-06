"""Resumable policy session that preserves simulator/renderer thread affinity."""

import uuid


class GeneratorPolicySession:
    def __init__(self, iterator):
        self.token = uuid.uuid4().hex
        self.iterator = iterator
        self.sequence = 0
        self.done = False
        self.report = None

    def _step(self, decision=None):
        try:
            self.iterator.send(decision)
            self.sequence += 1
        except StopIteration as stopped:
            self.done = True
            self.report = stopped.value
        return {"protocol": "roboenv/vla-callback/1", "token": self.token,
                "sequence": self.sequence, "status": "done" if self.done else "paused",
                "report": self.report}

    def start(self):
        return self._step()

    def advance(self, sequence, stop):
        if sequence != self.sequence or self.done:
            raise ValueError("stale policy callback decision")
        return self._step(bool(stop))

    def close(self):
        if not self.done:
            self._step(True)
        self.iterator.close()
