"""RoboCasa365 (robosuite / MuJoCo, Panda on a mobile base) — adapter and backend.

Importing this package registers :class:`RoboCasaAdapter`; robosuite/robocasa are
imported only when an adapter is instantiated (conda env ``robocasa``).
"""

from __future__ import annotations

from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter

__all__ = ["RoboCasaAdapter"]
