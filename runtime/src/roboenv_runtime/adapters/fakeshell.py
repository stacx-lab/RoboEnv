"""A toy world with the full adapter surface — for tests, demos and CI.

One puck on a line.  Pushing it past ``x >= 0.3`` completes the task; every
push costs 50 budget units.  Supports checkpoint/restore so the attempt loop
and rehearsal rules can be exercised without a simulator.
"""

from __future__ import annotations

from typing import Any, ClassVar

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.tools import tool
from roboenv_runtime.core.types import Budget, Capability, ScoreReport

__all__ = ["FakeShell"]


@register_adapter
class FakeShell(AdapterBase):
    """Deterministic toy adapter."""

    name: ClassVar[str] = "fakeshell"
    embodiment: ClassVar[str] = "general"
    capabilities: ClassVar[frozenset[str]] = frozenset({Capability.CHECKPOINT})
    situation_text: ClassVar[str] = "Toy world: one puck on a line, world frame x to the right."
    facts_text: ClassVar[str] = "Each push costs 50 budget units."

    GOAL_X = 0.3
    PUSH_COST = 50

    def __init__(self, max_budget: int = 10_000, goal_x: float | None = None) -> None:
        self.max_budget = max_budget
        self.goal_x = self.GOAL_X if goal_x is None else goal_x
        self.pos = [0.0, 0.0, 0.0]
        self.used = 0
        self.seed = 0
        self._snapshots: dict[int, tuple[list[float], int]] = {}

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int) -> None:
        """Put the puck back at the origin."""
        self.seed = seed
        self.pos = [0.0, 0.0, 0.0]
        self.used = 0
        self._snapshots.clear()

    def score(self) -> ScoreReport:
        """Solved once the puck passed ``goal_x``."""
        done = self.pos[0] >= self.goal_x
        return ScoreReport(reward=float(done), raw_score=float(done), completed=done)

    def budget(self) -> Budget:
        """Units consumed by pushes."""
        return Budget(used=self.used, max=self.max_budget, unit="sim_steps")

    def goal(self) -> str:
        """The language goal."""
        return f"Push the puck to x >= {self.goal_x}."

    # ------------------------------------------------------------ tools

    @tool
    def list_objects(self) -> list[str]:
        """Names of the objects in the world."""
        return ["puck"]

    @tool
    def get_object_pose(self, name: str) -> tuple[list[float], list[float]]:
        """Ground-truth pose: ``(position xyz, quaternion wxyz)``."""
        if name != "puck":
            raise KeyError(f"unknown object {name!r}")
        return list(self.pos), [1.0, 0.0, 0.0, 0.0]

    @tool
    def push(self, dx: float = 0.1) -> None:
        """Push the puck along x by ``dx`` metres (costs budget)."""
        if self.used >= self.max_budget:
            raise RuntimeError("terminated episode: budget exhausted")
        self.pos[0] += float(dx)
        self.used += self.PUSH_COST

    @tool
    def checkpoint(self) -> int:
        """Snapshot the world; returns a token for ``restore``."""
        token = len(self._snapshots) + 1
        self._snapshots[token] = (list(self.pos), self.used)
        return token

    @tool
    def restore(self, token: int) -> None:
        """Rewind the world to a checkpoint."""
        pos, used = self._snapshots[int(token)]
        self.pos, self.used = list(pos), used

    # ------------------------------------------------------------ media

    def frames(self) -> list[Any]:
        """No camera in the toy world."""
        return []
