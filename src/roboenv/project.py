"""Locate an explicit Roboenv source project, never an unrelated parent repository."""

import os
from pathlib import Path


def project_root(value=None):
    if value or os.environ.get("ROBOENV_PROJECT"):
        candidates = [Path(value or os.environ["ROBOENV_PROJECT"])]
    else:
        candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "pyproject.toml").is_file() and (root / "runtime/pyproject.toml").is_file():
            if (root / "src/roboenv").is_dir():
                return root
    raise ValueError("Select a Roboenv source project with --project or ROBOENV_PROJECT")
