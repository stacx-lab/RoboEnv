"""Tools: the agent-callable surface of an adapter, with one source of truth.

A tool is a callable plus its documentation — and the documentation *is* the
callable's docstring.  Adapters mark methods with :func:`tool`;
:class:`ToolBox` collects them, supports per-run include/exclude (ablation
experiments), and renders the prompt's ``api`` section from the same
docstrings the code carries.  There are no parallel doc strings anywhere.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ToolBox", "ToolSpec", "tool"]

_TOOL_ATTR = "__tool_name__"


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Mark a function or method as an agent-callable tool.

    Usable bare (``@tool``) or with an explicit public name
    (``@tool(name="goto")``).  The docstring of the decorated callable becomes
    its agent-facing documentation, so write it for the agent.
    """

    def mark(f: Callable[..., Any]) -> Callable[..., Any]:
        setattr(f, _TOOL_ATTR, name or f.__name__)
        return f

    return mark(fn) if fn is not None else mark


@dataclass(frozen=True)
class ToolSpec:
    """The documentation of one tool, independent of the live callable."""

    name: str
    signature: str
    doc: str

    @property
    def summary(self) -> str:
        """The first line of the docstring."""
        return self.doc.splitlines()[0] if self.doc else ""

    @classmethod
    def of(cls, name: str, fn: Callable[..., Any]) -> ToolSpec:
        """Build a spec from a live callable."""
        try:
            # eval_str resolves `from __future__ import annotations` strings so the
            # agent reads `dx: float`, not `dx: 'float'`
            signature = str(inspect.signature(fn, eval_str=True))
        except (TypeError, ValueError, NameError):
            try:
                signature = str(inspect.signature(fn))
            except (TypeError, ValueError):
                signature = "(...)"
        return cls(name=name, signature=signature, doc=inspect.getdoc(fn) or "")


class ToolBox(Mapping[str, Callable[..., Any]]):
    """An ordered, named collection of bound tools.

    Behaves like a read-only mapping ``name -> callable``.  Insertion order is
    preserved so API documentation renders deterministically.
    """

    def __init__(self, tools: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Callable[..., Any]] = dict(tools or {})

    # ------------------------------------------------------------ construction

    @classmethod
    def collect(cls, *hosts: Any) -> ToolBox:
        """Gather ``@tool``-marked methods from one or more host objects."""
        box = cls()
        for host in hosts:
            for _, member in inspect.getmembers(host):
                tool_name = getattr(member, _TOOL_ATTR, None)
                if tool_name:
                    box._tools[tool_name] = member
        return box

    @classmethod
    def from_mapping(cls, tools: Mapping[str, Callable[..., Any]]) -> ToolBox:
        """Wrap an existing ``name -> callable`` mapping."""
        return cls(tools)

    def add(self, fn: Callable[..., Any], name: str | None = None) -> None:
        """Register one callable under ``name`` (default: its tool/function name)."""
        self._tools[name or getattr(fn, _TOOL_ATTR, getattr(fn, "__name__", "fn"))] = fn

    def setdefault(self, name: str, fn: Callable[..., Any]) -> None:
        """Register ``fn`` only when ``name`` is not already present."""
        self._tools.setdefault(name, fn)

    def subset(
        self,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> ToolBox:
        """A new box restricted to ``include`` (if given) minus ``exclude``."""
        names = list(self._tools)
        if include is not None:
            wanted = set(include)
            names = [n for n in names if n in wanted]
        if exclude:
            dropped = set(exclude)
            names = [n for n in names if n not in dropped]
        return ToolBox({n: self._tools[n] for n in names})

    # ------------------------------------------------------------ mapping API

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self._tools[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        """Tool names in registration order."""
        return list(self._tools)

    def namespace(self) -> dict[str, Callable[..., Any]]:
        """A fresh dict of the tools — the globals injected into agent code."""
        return dict(self._tools)

    # ------------------------------------------------------------ documentation

    def specs(self) -> list[ToolSpec]:
        """One :class:`ToolSpec` per tool, in order."""
        return [ToolSpec.of(name, fn) for name, fn in self._tools.items()]

    def render_docs(self, style: str = "full") -> str:
        """Render the ``api`` prompt section from the tools' own docstrings.

        Args:
            style: ``"full"`` (signature + complete docstring, one block per
                tool) or ``"brief"`` (one line per tool: signature + summary).
        """
        specs = self.specs()
        if style == "brief":
            return "\n".join(f"- {s.name}{s.signature}: {s.summary}" for s in specs)
        blocks = ["Available API functions:"]
        for spec in specs:
            doc = spec.doc or "(no documentation)"
            body = "\n".join("    " + line for line in doc.splitlines())
            blocks.append(f"{spec.name}{spec.signature}\n{body}")
        return "\n\n".join(blocks)

    def __repr__(self) -> str:
        return f"ToolBox({self.names()})"
