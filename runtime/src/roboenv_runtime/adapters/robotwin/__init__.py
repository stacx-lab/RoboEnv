"""RoboTwin 2.0 (SAPIEN) — adapter, backend, cap-x APIs and launcher envs.

Importing this package registers :class:`RoboTwinAdapter`; SAPIEN is imported
only when an adapter (or the backend) is instantiated.
"""

from __future__ import annotations

from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter

__all__ = ["RoboTwinAdapter"]
