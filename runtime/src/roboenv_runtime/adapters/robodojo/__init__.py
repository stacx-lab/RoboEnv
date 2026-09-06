"""RoboDojo (Isaac Sim) — adapter, backend, cap-x API and launcher env.

Importing this package registers :class:`RoboDojoAdapter`; nothing heavy is
imported until an adapter (or the backend) is instantiated.
"""

from __future__ import annotations

from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter

__all__ = ["RoboDojoAdapter"]
