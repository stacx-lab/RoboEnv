"""BEHAVIOR-1K (OmniGibson / Isaac Sim, R1Pro in a house) — adapter and backend.

Importing this package registers :class:`BehaviorAdapter`; omnigibson is
imported only on the first reset (conda env ``behavior``).
"""

from __future__ import annotations

from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter

__all__ = ["BehaviorAdapter"]
