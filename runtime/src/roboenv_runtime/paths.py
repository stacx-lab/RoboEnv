"""Where things live: repository root, data directories, external simulators.

Every location can be overridden by an environment variable so the same
package works from a source checkout, from an installed wheel, and inside a
simulator's own interpreter.  Nothing here touches the filesystem beyond
``Path.exists`` checks.

Environment variables (all optional):

``ROBOENV_ROOT``
    Repository root.  Default: the first ancestor of this package that holds a
    ``pyproject.toml``; failing that, the current working directory.
``ROBOENV_RUNS`` / ``ROBOENV_SKILLS`` / ``ROBOENV_SOLUTIONS`` / ``ROBOENV_TASKS``
    Override the ``runs/``, ``skills/knowledge/``, ``skills/solutions/`` and ``tasks/``
    data directories.
``CAPX_ROOT``
    A cap-x checkout.  Default: ``<root>/third_party/cap-x``.
``ROBODOJO_ROOT`` / ``ROBOTWIN_ROOT``
    The RoboDojo checkout and the RoboTwin checkout.  Defaults:
    ``<root>/third_party/RoboDojo`` (falling back to the pre-refactor ``<root>/RoboDojo``)
    and ``<root>/third_party/RoboTwin``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "capx_root",
    "repo_root",
    "robodojo_root",
    "robotwin_root",
    "runs_root",
    "skills_root",
    "tasks_root",
]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def repo_root() -> Path:
    """The repository root (holds ``pyproject.toml``, ``tasks/``, ``skills/``)."""
    override = _env_path("ROBOENV_ROOT")
    if override is not None:
        return override
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "upstreams.lock.json").is_file():
            return candidate
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def runs_root() -> Path:
    """Where trial records are written (``runs/<task_id>/<timestamp>/``)."""
    return _env_path("ROBOENV_RUNS") or repo_root() / "runs"


def skills_root() -> Path:
    """Root of the knowledge-skill store (``shared/`` and ``tasks/`` below it)."""
    return _env_path("ROBOENV_SKILLS") or repo_root() / "skills" / "knowledge"


def solutions_root() -> Path:
    """Root of the distilled solutions (``<task_id>/solution.py`` + its certificate)."""
    return _env_path("ROBOENV_SOLUTIONS") or repo_root() / "skills" / "solutions"


def tasks_root() -> Path:
    """Root of the task definitions (``<simulator>/<task>.yaml``)."""
    override = _env_path("ROBOENV_TASKS")
    if override is not None:
        return override
    native = repo_root() / "benchmarks" / "manipulation" / "tasks"
    return native if native.is_dir() else repo_root() / "tasks"


def capx_root() -> Path:
    """A cap-x checkout (the submodule by default)."""
    override = _env_path("CAPX_ROOT")
    if override is not None:
        return override
    candidate = repo_root() / "third_party" / "cap-x"
    if candidate.exists():
        return candidate
    # historical location, kept as a fallback for pre-refactor checkouts
    return repo_root() / "cap-x"


def robodojo_root() -> Path:
    """The RoboDojo benchmark checkout (submodule ``third_party/RoboDojo``)."""
    override = _env_path("ROBODOJO_ROOT")
    if override:
        return override
    candidate = repo_root() / "third_party" / "RoboDojo"
    legacy = repo_root() / "RoboDojo"
    return candidate if candidate.exists() or not legacy.exists() else legacy


def robotwin_root() -> Path:
    """The RoboTwin 2.0 checkout (assets are resolved relative to it)."""
    override = _env_path("ROBOTWIN_ROOT")
    if override is not None:
        return override
    return repo_root() / "third_party" / "RoboTwin"
