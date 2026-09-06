"""The trial record: one schema for every interaction style.

A trial leaves ``record.json`` in its run directory.  The file is read back by
the board, by post-mortem builders and by the skill-evolution tools, so the
schema is versioned and only ever grows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roboenv_runtime.core.types import Budget, ScoreReport

__all__ = ["RECORD_SCHEMA_VERSION", "TrialRecord", "TrialStatus"]

RECORD_SCHEMA_VERSION = 1


class TrialStatus:
    """How a trial ended.  Plain strings so records stay greppable."""

    COMPLETED = "completed"
    """The benchmark's own success flag turned true."""
    SUBMITTED = "submitted"
    """The agent claimed completion and the claim was verified."""
    FINISHED = "finished"
    """The agent stopped voluntarily without a verified success."""
    STEP_LIMIT = "step_limit"
    BUDGET = "budget"
    TERMINATED = "terminated"
    """The environment ended the episode (e.g. an object fell off the world)."""
    FORMAT_ERRORS = "format_errors"
    STALLED = "stalled"
    LLM_ERROR = "llm_error"
    ERROR = "error"


@dataclass
class TrialRecord:
    """Everything worth keeping about one trial.

    Attributes:
        task_id: Task identifier (``robotwin.click_bell``).
        adapter: Adapter name the task ran on.
        style: Interaction style (``program``, ``react``, ``shell``, ``repl``).
        mode: Knowledge-injection mode (``discovery`` ... ``custom``).
        model: LLM identifier (``"human"`` for the turn-file REPL).
        seed: Episode seed.
        status: One of :class:`TrialStatus`.
        turns: Number of model/agent turns executed.
        reward / raw_score / tier / completed: Final :class:`ScoreReport` fields.
        budget: Final :class:`Budget` as a dict.
        rejected_submits: Completion claims that failed verification.
        error: Free-form failure reason when ``status`` is an error.
        seconds: Wall-clock duration.
        run_dir: Where the artefacts live.
        skills_used: Names of knowledge skills injected into the prompt.
        extra: Style-specific additions (kept out of the fixed schema).
    """

    task_id: str
    adapter: str = ""
    style: str = "program"
    mode: str = "discovery"
    model: str = ""
    seed: int = 0
    status: str = TrialStatus.STEP_LIMIT
    turns: int = 0
    reward: float = 0.0
    raw_score: float = 0.0
    tier: int = 0
    completed: bool = False
    budget: dict[str, Any] = field(default_factory=dict)
    rejected_submits: int = 0
    error: str = ""
    seconds: float = 0.0
    run_dir: str = ""
    skills_used: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RECORD_SCHEMA_VERSION

    # ------------------------------------------------------------ helpers

    def apply(self, score: ScoreReport, budget: Budget | None = None) -> None:
        """Copy the latest evaluation into the record."""
        self.reward = float(score.reward)
        self.raw_score = float(score.raw_score)
        self.tier = int(score.tier)
        self.completed = bool(score.completed)
        if budget is not None:
            self.budget = budget.as_dict()

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return asdict(self)

    def write(self, run_dir: Path | str, filename: str = "record.json") -> Path:
        """Persist as ``<run_dir>/record.json`` and return the path."""
        path = Path(run_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path | str) -> TrialRecord:
        """Load a record, tolerating unknown/missing keys from other writers."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        record = cls(**kwargs)
        if extra:
            record.extra = {**extra, **record.extra}
        return record

    def summary(self) -> str:
        """One line for logs and CLI output."""
        return (
            f"RESULT [{self.task_id} {self.style} seed={self.seed} mode={self.mode}] "
            f"status={self.status} reward={self.reward:.3f} completed={self.completed} "
            f"turns={self.turns} ({self.seconds}s)"
        )
