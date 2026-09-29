"""Native score and budget value types.

These are deliberately tiny dataclasses with explicit semantics, because the
pre-refactor harnesses disagreed on exactly these points (normalised vs raw
scores, step vs call budgets).  The rules are:

* :attr:`ScoreReport.reward` is **always** normalised to ``[0, 1]``;
  :attr:`ScoreReport.raw_score` is **always** the benchmark's native scale.
* :attr:`Budget.unit` names what is being counted, and consumers must only
  ever compare ``used`` against ``max`` in that same unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Budget", "ScoreReport"]


@dataclass
class ScoreReport:
    """The evaluation of the current world state.

    Attributes:
        reward: Normalised score in ``[0, 1]``; ``1.0`` means the task is solved.
        raw_score: The benchmark's native score (e.g. RoboDojo's ``0..100``).
        tier: Number of graded milestones reached, ``0`` when the benchmark has
            no tiers.
        completed: The benchmark's own binary success flag.
        detail: Free-form benchmark-specific breakdown (check-queue state, per
            object gates, ...).  Never required by the harness.
    """

    reward: float = 0.0
    raw_score: float = 0.0
    tier: int = 0
    completed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return asdict(self)


@dataclass
class Budget:
    """How much of the episode has been consumed.

    Attributes:
        used: Amount consumed so far, in ``unit``.
        max: The cap for this episode, in ``unit``, or None for no cap.
        unit: What is counted — ``"sim_steps"`` (physics steps),
            ``"action_frames"`` (benchmark policy steps) or ``"api_calls"``
            (platforms that keep no simulator counter).
    """

    used: int = 0
    max: int | None = None
    unit: str = "sim_steps"

    @property
    def remaining(self) -> int | None:
        """Units left, or None for an unbounded episode."""
        return None if self.max is None else max(0, self.max - self.used)

    @property
    def exhausted(self) -> bool:
        """True once ``used`` reaches ``max``."""
        return self.max is not None and self.used >= self.max

    def as_dict(self) -> dict[str, Any]:
        """The effective episode budget, with null for an unbounded limit."""
        out: dict[str, Any] = {"used": self.used, "max": self.max, "unit": self.unit}
        return out

    def __str__(self) -> str:
        return f"{self.used}/{self.max} {self.unit}"
