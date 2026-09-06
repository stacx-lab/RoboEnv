"""Create an immutable, self-describing run directory.

A *run* is one episode of one task under one budget mode, evaluated through a
worker.  ``create_run`` lays out ``runs/<task_id>/<run_id>/`` with the
subdirectories the supervisor and worker expect, copies the task definition
in as evidence, and writes ``manifest.json`` — what was run, with which
package files (hashed), on which host — so ``result.json`` can later be
derived offline by :mod:`roboenv_runtime.runtime.finalize`.

Rewritten from ``codex_dojo_alltasks/harness/run_manifest.py`` (2026-08-26):
the task matrix and campaign boundary became the task definition and the
package hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from roboenv_runtime import __version__
from roboenv_runtime.paths import repo_root, runs_root
from roboenv_runtime.runtime.atomic import atomic_write_json
from roboenv_runtime.tasks.loader import find_task
from roboenv_runtime.tasks.schema import TaskSpec

__all__ = ["MANIFEST_SCHEMA_VERSION", "RUN_SUBDIRS", "create_run", "package_hashes"]

MANIFEST_SCHEMA_VERSION = 1
RUN_SUBDIRS = (
    "workspace",
    "turns",
    "logs",
    "captures",
    "state",
    "journal",
    "evidence",
    "vla",
    "tmp",
    "cache",
    "home",
)
ACTION_FRAME_TO_SIM_STEPS = 10
"""RoboDojo's ``collect_interval``: simulator steps per official action frame."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_hashes(root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every python file of the ``roboenv`` package (provenance)."""
    package = Path(root) / "roboenv_runtime" if root else Path(__file__).resolve().parents[1]
    if not package.is_dir():
        package = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(package.parent)): _sha(path)
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def create_run(
    task: TaskSpec | str,
    *,
    seed: int = 0,
    budget_mode: str = "practice",
    gpu: int | None = None,
    runs_dir: Path | None = None,
    source_lock: Path | None = None,
) -> dict[str, Any]:
    """Create ``runs/<task_id>/<run_id>/`` and return its manifest."""
    spec = task if isinstance(task, TaskSpec) else find_task(task)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if budget_mode not in {"official", "practice"}:
        raise ValueError("budget_mode must be official or practice")
    frames = spec.budget.get("official_action_frames")
    if budget_mode == "official" and not frames:
        raise ValueError(
            f"task {spec.id} declares no budget.official_action_frames; an official run needs one"
        )
    root = (runs_dir or runs_root()).resolve()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / spec.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in RUN_SUBDIRS:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {}
    if spec.source is not None and spec.source.is_file():
        copy = run_dir / "evidence" / spec.source.name
        shutil.copy2(spec.source, copy)
        evidence["task_file"] = {
            "source": str(spec.source),
            "copy": str(copy),
            "sha256": _sha(copy),
        }
    if source_lock is not None:
        from roboenv_runtime.runtime.source_lock import verify

        check = verify(source_lock, repo_root())
        atomic_write_json(run_dir / "evidence" / "source_lock_check.json", check)
        if not check.get("ok"):
            raise RuntimeError("required source lock mismatch; run cannot start")
        evidence["source_lock"] = str(source_lock)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "roboenv_version": __version__,
        "task": spec.id,
        "adapter": spec.adapter,
        "adapter_kwargs": dict(spec.adapter_kwargs),
        "seed": int(seed),
        "budget_mode": budget_mode,
        "official_action_frames": int(frames) if frames else None,
        "official_sim_step_cap": int(frames) * ACTION_FRAME_TO_SIM_STEPS if frames else None,
        "mechanism": spec.tags.get("mechanism"),
        "physical_gpu": gpu,
        "run_dir": str(run_dir),
        "evidence": evidence,
        "package_files": package_hashes(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """``python -m roboenv_runtime.runtime.manifest --task T [--seed N] [--budget-mode M] [--gpu G]``."""
    parser = argparse.ArgumentParser(description="create a run directory + manifest")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-mode", choices=("official", "practice"), default="practice")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--source-lock", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = create_run(
        args.task,
        seed=args.seed,
        budget_mode=args.budget_mode,
        gpu=args.gpu,
        runs_dir=args.runs_dir,
        source_lock=args.source_lock,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "package_files"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
