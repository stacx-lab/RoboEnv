"""The task definition schema.

A task file is YAML::

    id: robotwin.click_bell            # optional; defaults to <folder>.<stem>
    adapter: robotwin                  # registry name
    adapter_kwargs: {task_name: click_bell}
    text: |                            # the `task` prompt section (legacy key: task)
      Task: touch the bell ...
    mode: skilled                      # discovery | documented | skilled | custom
    style: react                       # program | chunked | react | repl  (legacy key: interaction)
    max_turns: 25
    seeds: [0]
    budget: {mode: practice}           # practice | official (+ optional max)
    tags: {mechanism: contact}         # free-form labels for the board
    options: {digest: true}            # style-specific options

Only ``adapter`` is required; ``text`` may be empty when the simulator carries
its own language goal (LIBERO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["TaskSpec"]

_LEGACY_STYLE = {"repl": "chunked", "program": "program"}


@dataclass
class TaskSpec:
    """One task definition."""

    id: str
    adapter: str
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    mode: str = "discovery"
    style: str = "program"
    max_turns: int = 4
    seeds: list[int] = field(default_factory=lambda: [0])
    budget: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> TaskSpec:
        """Build from a parsed YAML mapping, accepting the pre-refactor keys."""
        if not isinstance(data, dict):
            raise ValueError(f"task definition must be a mapping, got {type(data).__name__}")
        if "backend" in data:
            data = {"id": data.get("backend_task", data.get("id")),
                    "adapter": data["backend"], "adapter_kwargs": data.get("backend_kwargs", {}),
                    "text": data.get("instruction", ""), "seeds": [data.get("seed", 0)],
                    "budget": {"mode": data.get("budget_mode", "official")}}
        adapter = data.get("adapter")
        if not adapter:
            raise ValueError(f"task definition {source or ''} has no 'adapter'")
        task_id = data.get("id")
        if not task_id and source is not None:
            task_id = f"{source.parent.name}.{source.stem}" if source.parent.name else source.stem
        if not task_id:
            raise ValueError("task definition needs an 'id' (or a file path to derive it from)")
        style = data.get("style") or _LEGACY_STYLE.get(data.get("interaction", ""), "program")
        options = dict(data.get("options") or {})
        if "digest" in data:
            options.setdefault("digest", bool(data["digest"]))
        return cls(
            id=str(task_id),
            adapter=str(adapter),
            adapter_kwargs=dict(data.get("adapter_kwargs") or {}),
            text=str(data.get("text") if data.get("text") is not None else data.get("task", "")),
            mode=str(data.get("mode", "discovery")),
            style=str(style),
            max_turns=int(data.get("max_turns", 4)),
            seeds=[int(s) for s in (data.get("seeds") or [0])],
            budget=dict(data.get("budget") or {}),
            tags=dict(data.get("tags") or {}),
            options=options,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        """The canonical YAML shape (new keys only)."""
        out: dict[str, Any] = {
            "id": self.id,
            "adapter": self.adapter,
            "adapter_kwargs": dict(self.adapter_kwargs),
            "text": self.text,
            "mode": self.mode,
            "style": self.style,
            "max_turns": self.max_turns,
            "seeds": list(self.seeds),
        }
        for key in ("budget", "tags", "options"):
            value = getattr(self, key)
            if value:
                out[key] = dict(value)
        return out

    @property
    def group(self) -> str:
        """The part of the id before the first dot (usually the simulator)."""
        return self.id.split(".", 1)[0]

    @property
    def short_name(self) -> str:
        """The part of the id after the first dot."""
        return self.id.split(".", 1)[1] if "." in self.id else self.id
