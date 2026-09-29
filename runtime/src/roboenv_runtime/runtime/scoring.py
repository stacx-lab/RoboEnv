"""One-shot owner evaluation and stable terminal snapshots of native verdicts."""

from __future__ import annotations

import copy
import threading
from typing import Any

from roboenv_runtime.core.adapter import EnvAdapter

__all__ = ["SimpleScorer"]

SUCCESS_THRESHOLD = 1.0 - 1e-3


class SimpleScorer:
    """Score authority for adapters without benchmark-side scoring machinery."""

    def __init__(self, adapter: EnvAdapter) -> None:
        self.adapter = adapter
        self._lock = threading.RLock()
        self._latched = False
        self._submitted = False
        self._result: dict[str, Any] | None = None
        self._submit_error: Exception | None = None

    @property
    def latched(self) -> bool:
        """True once a successful evaluation has been observed."""
        with self._lock:
            return self._latched

    @property
    def submitted(self) -> bool:
        """True after :meth:`submit` ran."""
        with self._lock:
            return self._submitted

    def _snapshot(self) -> dict[str, Any]:
        score = self.adapter.score()
        if score.reward >= SUCCESS_THRESHOLD or score.completed:
            self._latched = True
        return {
            **score.as_dict(),
            "latched": self._latched,
            "submitted": self._submitted,
            "budget": self.adapter.budget().as_dict(),
        }

    def peek(self) -> dict[str, Any]:
        """Read the current evaluation without side effects on the benchmark."""
        with self._lock:
            if self._submit_error is not None:
                raise RuntimeError(
                    "the one-shot final submit previously failed"
                ) from self._submit_error
            if self._result is not None:
                return copy.deepcopy(self._result)
            return self._snapshot()

    def submit(self) -> dict[str, Any]:
        """Run the final evaluation once and freeze its result."""
        with self._lock:
            if self._submit_error is not None:
                raise RuntimeError(
                    "the one-shot final submit previously failed"
                ) from self._submit_error
            if self._result is None:
                self._submitted = True
                try:
                    self._result = self._snapshot()
                except Exception as exc:
                    self._submit_error = exc
                    raise
            return copy.deepcopy(self._result)

    def close(self) -> None:
        """Nothing to restore."""
        return None
