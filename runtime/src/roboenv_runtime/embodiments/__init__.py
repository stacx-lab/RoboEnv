"""Embodiment profiles: the robot-specific facts adapters and skills rely on.

Every constant that used to live in a function default or a prompt string —
table height, arm base positions, quaternion conventions, jaw geometry, tool
offsets — is data here, one YAML per embodiment.  Adapters render their
``situation``/``facts`` prompt sections from the profile's templates, and
primitive packs read calibration numbers from ``gripper`` / ``tool`` sections
instead of hard-coding them.

Profiles ship with the package (``roboenv/embodiments/*.yaml``); a directory
named by ``$ROBOENV_EMBODIMENTS`` is searched first so a deployment can override
or add profiles without editing the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Embodiment", "list_embodiments", "load_embodiment"]

_HERE = Path(__file__).resolve().parent


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    override = os.environ.get("ROBOENV_EMBODIMENTS")
    if override:
        dirs.append(Path(override).expanduser())
    dirs.append(_HERE)
    return dirs


@dataclass(frozen=True)
class Embodiment:
    """One robot embodiment.

    Attributes:
        name: Profile id (``arx_x5``, ``yam``, ``piper``, ``panda``).
        simulator: The simulator family the profile was measured in.
        description: One line for humans.
        arms: Arm names (``["left", "right"]`` or ``["arm"]``).
        frame: World-frame conventions (``quaternion_order``, ``table_z`` ...).
        workspace: Where objects spawn / what each arm reaches, in metres.
        gripper: Jaw geometry and calibration (``max_opening_m``, ``gw_*`` fits ...).
        tool: Tool-frame conventions (``axis``, ``down_quaternions_wxyz``, ``tcp_bias_m``).
        planner: How motion is generated (notes agents need to know).
        situation: Prompt template for the ``situation`` section.
        facts: Prompt template for the ``facts`` section.
        extra: Anything else the YAML carried.
    """

    name: str
    simulator: str = ""
    description: str = ""
    arms: tuple[str, ...] = ("left", "right")
    frame: dict[str, Any] = field(default_factory=dict)
    workspace: dict[str, Any] = field(default_factory=dict)
    gripper: dict[str, Any] = field(default_factory=dict)
    tool: dict[str, Any] = field(default_factory=dict)
    planner: dict[str, Any] = field(default_factory=dict)
    situation: str = ""
    facts: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def table_z(self) -> float | None:
        """Tabletop height in the world frame, when known."""
        value = self.frame.get("table_z")
        return float(value) if value is not None else None

    def render_situation(self, **params: Any) -> str:
        """The ``situation`` prompt section (``{name}`` placeholders filled)."""
        return self.situation.format(**params).strip() + "\n" if self.situation else ""

    def render_facts(self, **params: Any) -> str:
        """The ``facts`` prompt section."""
        return self.facts.format(**params).strip() + "\n" if self.facts else ""


def _find(name: str) -> Path:
    for directory in _search_dirs():
        candidate = directory / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    raise KeyError(f"no embodiment profile {name!r}; known: {list_embodiments()}")


@cache
def load_embodiment(name: str) -> Embodiment:
    """Load (and cache) the profile called ``name``."""
    data = yaml.safe_load(_find(name).read_text(encoding="utf-8")) or {}
    known = set(Embodiment.__dataclass_fields__) - {"extra", "name"}
    kwargs = {k: v for k, v in data.items() if k in known}
    if "arms" in kwargs:
        kwargs["arms"] = tuple(kwargs["arms"])
    extra = {k: v for k, v in data.items() if k not in known and k != "name"}
    return Embodiment(name=name, extra=extra, **kwargs)


def list_embodiments() -> list[str]:
    """Profile names available in the search path."""
    names: set[str] = set()
    for directory in _search_dirs():
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.yaml"))
    return sorted(names)
