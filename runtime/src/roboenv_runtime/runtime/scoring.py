"""Who owns the score inside a worker.

A worker never lets agent code touch the adapter's ``score()`` directly; a
*score authority* answers ``score.peek`` (read-only) and ``submit`` (the one
final evaluation), and remembers whether success was ever observed
(``latched``).  Two implementations:

* :class:`SimpleScorer` — wraps :meth:`EnvAdapter.score` for any adapter.
* the RoboDojo ``ScoreClock`` (``roboenv_runtime.adapters.robodojo.score_clock``) —
  seals the benchmark's registrars and advances its reward manager at the
  official action-frame cadence.  Adapters that need such a thing expose it
  through :meth:`SupportsScoreAuthority.score_authority`.
"""

from __future__ import annotations

import contextlib
import copy
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from roboenv_runtime.core.adapter import EnvAdapter

__all__ = ["ScoreAuthority", "SimpleScorer", "SupportsScoreAuthority", "make_score_authority"]

SUCCESS_THRESHOLD = 1.0 - 1e-3


@runtime_checkable
class ScoreAuthority(Protocol):
    """The scoring surface a worker exposes."""

    @property
    def latched(self) -> bool:
        """True once a successful evaluation has been observed."""
        ...

    @property
    def submitted(self) -> bool:
        """True after :meth:`submit` ran."""
        ...

    def peek(self) -> dict[str, Any]:
        """Read the current evaluation without side effects on the benchmark."""
        ...

    def submit(self) -> dict[str, Any]:
        """Run the final evaluation once and freeze the episode."""
        ...

    def native_action(self) -> contextlib.AbstractContextManager[None]:
        """Context for one benchmark-native action frame."""
        ...

    def close(self) -> None:
        """Release patches/resources so the adapter can be reset again."""
        ...


@runtime_checkable
class SupportsScoreAuthority(Protocol):
    """Adapters that provide their own score authority (official budgets)."""

    def score_authority(self, run_dir: Path, **options: Any) -> ScoreAuthority:
        """A fresh authority for the current episode (call after ``reset``)."""
        ...


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
        """See :class:`ScoreAuthority`."""
        with self._lock:
            return self._latched

    @property
    def submitted(self) -> bool:
        """See :class:`ScoreAuthority`."""
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
        """See :class:`ScoreAuthority`."""
        with self._lock:
            if self._submit_error is not None:
                raise RuntimeError(
                    "the one-shot final submit previously failed"
                ) from self._submit_error
            if self._result is not None:
                return copy.deepcopy(self._result)
            return self._snapshot()

    def submit(self) -> dict[str, Any]:
        """See :class:`ScoreAuthority`."""
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

    @contextlib.contextmanager
    def native_action(self) -> Iterator[None]:
        """No benchmark clock to advance; a plain context."""
        yield

    def close(self) -> None:
        """Nothing to restore."""
        return None


def make_score_authority(
    adapter: EnvAdapter, run_dir: Path, *, budget_mode: str = "practice", **options: Any
) -> ScoreAuthority:
    """The adapter's own authority when it has one and the mode asks for it.

    ``official`` runs use the adapter's benchmark-side authority (RoboDojo's ``ScoreClock``,
    which seals the registrars and owns the reward cadence).  ``practice`` runs score through
    :class:`SimpleScorer` → :meth:`EnvAdapter.score`, exactly as in-process trials do — so
    programs that drive the reward clock themselves (the campaign REPL libraries) behave the
    same under a worker as they did in the REPL.
    """
    factory = getattr(adapter, "score_authority", None)
    if budget_mode == "official" and callable(factory):
        return factory(run_dir, **options)
    return SimpleScorer(adapter)
