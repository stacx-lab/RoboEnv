"""Value types shared by adapters, the agent loop and the records.

These are deliberately tiny dataclasses with explicit semantics, because the
pre-refactor harnesses disagreed on exactly these points (normalised vs raw
scores, step vs call budgets).  The rules are:

* :attr:`ScoreReport.reward` is **always** normalised to ``[0, 1]``;
  :attr:`ScoreReport.raw_score` is **always** the benchmark's native scale.
* :attr:`Budget.unit` names what is being counted, and consumers must only
  ever compare ``used`` against ``max`` in that same unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Budget", "Capability", "ScoreReport"]


class Capability:
    """Optional features an adapter may declare in ``capabilities``.

    Capabilities are plain strings so that adapters written against an older
    harness keep working; the constants exist to avoid typos.
    """

    CHECKPOINT = "checkpoint"
    """``checkpoint()`` / ``restore(id)`` tools rewind the world in place."""
    FUNCTIONAL_POINTS = "functional_points"
    """Objects carry named functional points (rim, handle, bottom, ...)."""
    OFFICIAL_BUDGET = "official_budget"
    """The benchmark defines an official per-task step cap."""
    BASE_MOTION = "base_motion"
    """The robot has a mobile base (e.g. RoboCasa's Panda-Omron)."""
    VIDEO = "video"
    """``frames()`` returns captured camera frames after an episode."""
    PRIVILEGED = "privileged"
    """Ground-truth state (poses, checker internals) is readable."""

    ALL = frozenset(
        {CHECKPOINT, FUNCTIONAL_POINTS, OFFICIAL_BUDGET, BASE_MOTION, VIDEO, PRIVILEGED}
    )


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

    def summary(self) -> str:
        """Compact one-line rendering used in agent feedback."""
        return (
            f"{{reward: {self.reward}, raw: {self.raw_score}, tier: {self.tier}, "
            f"completed: {self.completed}}}"
        )


@dataclass
class Budget:
    """How much of the episode has been consumed.

    Attributes:
        used: Amount consumed so far, in ``unit``.
        max: The cap for this episode, in ``unit``, or None for no cap.
        unit: What is counted — ``"sim_steps"`` (physics steps),
            ``"action_frames"`` (benchmark policy steps) or ``"api_calls"``
            (platforms that keep no simulator counter).
        official_cap: The benchmark's official cap in ``unit`` when it differs
            from ``max`` (practice budgets are larger than official ones).
    """

    used: int = 0
    max: int | None = 1
    unit: str = "sim_steps"
    official_cap: int | None = None

    @property
    def remaining(self) -> int | None:
        """Units left, or None for an unbounded episode."""
        return None if self.max is None else max(0, self.max - self.used)

    @property
    def exhausted(self) -> bool:
        """True once ``used`` reaches ``max``."""
        return self.max is not None and self.used >= self.max

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view (``official_cap`` omitted when unset)."""
        out: dict[str, Any] = {"used": self.used, "max": self.max, "unit": self.unit}
        if self.official_cap is not None:
            out["official_cap"] = self.official_cap
        return out

    @classmethod
    def coerce(cls, value: Budget | Mapping[str, Any] | None) -> Budget:
        """Accept a :class:`Budget` or the historical ``{"used", "max"}`` dict."""
        if isinstance(value, Budget):
            return value
        if not value:
            return cls()
        return cls(
            used=int(value.get("used", 0) or 0),
            max=None if "max" in value and value["max"] is None else int(value.get("max", 1) or 1),
            unit=str(value.get("unit", "sim_steps") or "sim_steps"),
            official_cap=(
                int(value["official_cap"]) if value.get("official_cap") is not None else None
            ),
        )

    def __str__(self) -> str:
        return f"{self.used}/{self.max} {self.unit}"
