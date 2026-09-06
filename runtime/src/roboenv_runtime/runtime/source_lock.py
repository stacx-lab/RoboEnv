"""Verify read-only dependency provenance before a simulator boot."""

# Moved from codex_dojo_alltasks/harness/source_lock.py on 2026-08-26 (refactor P3).

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _inside_source(path: Path, source_root: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved == source_root or source_root not in resolved.parents:
        raise ValueError(f"dependency path escapes source root: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(lock_path: str | Path, source_root: str | Path) -> dict[str, Any]:
    lock_path = Path(lock_path).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dependencies = lock.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("source lock requires a non-empty dependencies mapping")
    entries = []
    ok = True
    for relative, expected in sorted(dependencies.items()):
        error = None
        candidate = source_root / "__invalid_dependency__"
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            error = "dependency key must be a non-empty relative path without '..'"
        elif (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in expected)
        ):
            error = "expected digest must be 64 hexadecimal characters"
        else:
            try:
                candidate = _inside_source(source_root / relative, source_root)
            except ValueError as exc:
                error = str(exc)
        # Warriors deploys the two dojo files at its root.  They are historical
        # inputs only; campaign skills are separate, but still provenance-lock
        # them when the deployed copies are present.
        if error is None and not candidate.exists() and relative.startswith("refactor/dojo/"):
            try:
                candidate = _inside_source(source_root / Path(relative).name, source_root)
            except ValueError as exc:
                error = str(exc)
        exists = error is None and candidate.is_file()
        actual = sha256_file(candidate) if exists else None
        match = bool(exists and actual == expected)
        entries.append(
            {
                "relative": relative,
                "resolved": str(candidate),
                "expected": expected,
                "actual": actual,
                "match": match,
                "error": error,
            }
        )
        ok = ok and match
    return {"ok": ok, "source_root": str(source_root), "entries": entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    result = verify(args.lock, args.source_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
