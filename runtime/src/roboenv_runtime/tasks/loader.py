"""Finding and loading task files.

Layout under :func:`roboenv_runtime.paths.tasks_root` (``tasks/`` by default)::

    tasks/<group>/<name>.yaml          canonical; id defaults to "<group>.<name>"
    tasks/<group>.<name>.yaml          flat form, also accepted

``<group>`` is normally the adapter name; it may carry a variant suffix
(``robodojo-yam``) when the same adapter is run with a different embodiment.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

from roboenv_runtime.paths import tasks_root
from roboenv_runtime.tasks.schema import TaskSpec

__all__ = ["find_task", "iter_task_files", "list_tasks", "load_task"]


def load_task(path: Path | str) -> TaskSpec:
    """Parse one task file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TaskSpec.from_dict(data, source=path)


def iter_task_files(root: Path | None = None) -> Iterator[Path]:
    """Every ``*.yaml`` task file below ``root``, sorted."""
    root = Path(root or tasks_root())
    if not root.is_dir():
        return
    yield from sorted(p for p in root.rglob("*.yaml") if not p.name.startswith("_"))


def find_task(ref: str, root: Path | None = None) -> TaskSpec:
    """Resolve ``ref`` — a file path or a task id — to a :class:`TaskSpec`.

    Raises:
        FileNotFoundError: when neither a file nor a known id matches.
    """
    candidate = Path(ref)
    if candidate.suffix in (".yaml", ".yml") and candidate.is_file():
        return load_task(candidate)
    root = Path(root or tasks_root())
    group, _, name = ref.partition(".")
    for path in (root / group / f"{name}.yaml", root / f"{ref}.yaml", root / group / f"{name}.yml"):
        if name and path.is_file():
            return load_task(path)
    for path in iter_task_files(root):
        spec = load_task(path)
        if spec.id == ref:
            return spec
    raise FileNotFoundError(f"no task file for {ref!r} under {root}")


def list_tasks(root: Path | None = None, adapter: str | None = None) -> list[TaskSpec]:
    """All task definitions, optionally filtered by adapter name."""
    specs = [load_task(p) for p in iter_task_files(root)]
    if adapter:
        specs = [s for s in specs if s.adapter == adapter]
    return specs
