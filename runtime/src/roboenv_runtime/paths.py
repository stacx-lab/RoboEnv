"""Where things live: repository root, data directories, external simulators.

Every location can be overridden by an environment variable so the same
package works from a source checkout, from an installed wheel, and inside a
simulator's own interpreter.  Nothing here touches the filesystem beyond
``Path.exists`` checks.

Environment variables (all optional):

``ROBOENV_ROOT``
    Repository root.  Default: the first ancestor of this package that holds a
    ``pyproject.toml``; failing that, the current working directory.
``ROBOENV_TASKS``
    The task definitions. Default: ``<root>/catalog/tasks``.
``ROBODOJO_ROOT`` / ``ROBOTWIN_ROOT``
    The RoboDojo checkout and the RoboTwin checkout.  Defaults:
    ``<root>/third_party/RoboDojo``
    and ``<root>/third_party/RoboTwin``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "robodojo_root",
    "robotwin_root",
    "tasks_root",
]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def repo_root() -> Path:
    """The repository root (holds ``upstreams.lock.json``, ``pyproject.toml`` and ``catalog/``)."""
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


def tasks_root() -> Path:
    """Root of the task definitions (``<simulator>/<task>/task.yaml``)."""
    return _env_path("ROBOENV_TASKS") or repo_root() / "catalog" / "tasks"


def robodojo_root() -> Path:
    """The RoboDojo benchmark checkout (submodule ``third_party/RoboDojo``)."""
    override = _env_path("ROBODOJO_ROOT")
    if override:
        return override
    return repo_root() / "third_party" / "RoboDojo"


def robotwin_root() -> Path:
    """The RoboTwin 2.0 checkout (assets are resolved relative to it)."""
    override = _env_path("ROBOTWIN_ROOT")
    if override is not None:
        return override
    return repo_root() / "third_party" / "RoboTwin"
