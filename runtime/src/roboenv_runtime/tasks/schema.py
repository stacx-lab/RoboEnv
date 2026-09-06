"""Validated worker input derived from the public environment task.

Only environment fields belong here. Agent styles, turn limits, skills, and
prompt options are owned by the harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roboenv.tasks.model import Task
from roboenv.validation import json_value


@dataclass
class TaskSpec:
    """Worker initialization and benchmark metadata."""

    id: str
    adapter: str
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    seeds: list[int] = field(default_factory=lambda: [0])
    budget: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_dict(cls, data, source=None):
        if not isinstance(data, dict):
            raise ValueError("task definition must be a mapping")
        if "backend" in data or "simulation" in data:
            task = Task.from_dict(data)
            data = {
                "id": task.backend_task,
                "adapter": task.backend,
                "adapter_kwargs": task.backend_kwargs,
                "text": task.instruction,
                "seeds": [task.seed],
                "budget": {"mode": task.budget_mode},
            }
        allowed = {"id", "adapter", "adapter_kwargs", "text", "seeds", "budget", "tags"}
        if set(data) - allowed:
            raise ValueError("task contains non-environment or unknown fields")
        identifier = data.get("id")
        if identifier is None and source is not None:
            identifier = f"{source.parent.name}.{source.stem}"
        adapter = data.get("adapter")
        for name, value in (("id", identifier), ("adapter", adapter)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"task requires a nonempty {name}")
        seeds = data.get("seeds", [0])
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError("seeds must be nonempty nonnegative integers")
        for key in ("adapter_kwargs", "budget", "tags"):
            if not isinstance(data.get(key, {}), dict):
                raise ValueError(f"{key} must be an object")
        if not isinstance(data.get("text", ""), str):
            raise ValueError("text must be a string")
        json_value(data)
        return cls(
            id=identifier,
            adapter=adapter,
            adapter_kwargs=dict(data.get("adapter_kwargs", {})),
            text=data.get("text", ""),
            seeds=list(seeds),
            budget=dict(data.get("budget", {})),
            tags=dict(data.get("tags", {})),
            source=source,
        )

    def to_dict(self):
        result = asdict(self)
        result.pop("source")
        return result

    @property
    def group(self):
        return self.id.split(".", 1)[0]

    @property
    def short_name(self):
        return self.id.split(".", 1)[-1]
