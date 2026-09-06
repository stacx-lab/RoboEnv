"""Environment-owned adapters, typed contracts and journaled simulator runtime.

Agent loops, model prompting, skills and experiment orchestration are deliberately
not part of this package. Heavy simulators load only when their adapter is made.
"""

from __future__ import annotations

from roboenv_runtime.core import (
    AdapterBase,
    Budget,
    Capability,
    EnvAdapter,
    ScoreReport,
    ToolBox,
    TrialRecord,
    list_adapters,
    make_adapter,
    register_adapter,
    tool,
)

__version__ = "0.2.0"

__all__ = [
    "AdapterBase",
    "Budget",
    "Capability",
    "EnvAdapter",
    "ScoreReport",
    "ToolBox",
    "TrialRecord",
    "__version__",
    "list_adapters",
    "make_adapter",
    "register_adapter",
    "tool",
]

