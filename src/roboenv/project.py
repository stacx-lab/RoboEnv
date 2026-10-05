"""Locate an explicit Roboenv source project, never an unrelated parent repository."""

import os
from pathlib import Path

# A standalone simulator image carries a source project's runtime data (its upstream lock, the
# selected sources and the receipt of their verification at build time) without its packaging.
PACKAGED_RECEIPT = ".roboenv-sources.json"


def is_source_project(root):
    return (
        (root / "pyproject.toml").is_file()
        and (root / "runtime/pyproject.toml").is_file()
        and (root / "src/roboenv").is_dir()
    )


def is_packaged_root(root):
    return (root / "upstreams.lock.json").is_file() and (root / PACKAGED_RECEIPT).is_file()


def project_root(value=None):
    if value or os.environ.get("ROBOENV_PROJECT"):
        candidates = [Path(value or os.environ["ROBOENV_PROJECT"])]
    else:
        candidates = [Path.cwd(), *Path.cwd().parents]
        # The runtime root a launch hands its simulator processes comes before this package's
        # own checkout: in a standalone image it is the packaged project, wherever they run from.
        if os.environ.get("ROBOENV_ROOT"):
            candidates.append(Path(os.environ["ROBOENV_ROOT"]))
        candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        root = candidate.resolve()
        if is_source_project(root) or is_packaged_root(root):
            return root
    raise ValueError("Select a Roboenv source project with --project or ROBOENV_PROJECT")
