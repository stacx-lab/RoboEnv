"""Prepare a private, history-free source snapshot; never initialize, publish or push Git.

The approved destination is a private source-preview repository. An absent core
license remains absent and does not become an open-source licensing declaration.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from release_preflight import inspect
from roboenv.privacy import owned_files, validate_public_text

ALLOWED_ROOTS = {
    ".github",
    ".gitignore",
    ".gitmodules",
    ".gitattributes",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    "pyproject.toml",
    "upstreams.lock.json",
    "catalog",
    "docs",
    "locks",
    "runtime",
    "src",
    "tests",
    "third_party",
    "tools",
}
EXCLUSIONS = [
    "Git history and metadata",
    "local caches, environments, build outputs, assets and checkpoints",
    "generated task sets and local deployments",
    "historical bulk benchmark tasks and experiment evidence",
    "upstream checkout contents (represented by pinned Git submodule entries)",
]


def manifest(root):
    files = []
    for source in owned_files(root):
        relative = source.relative_to(root)
        if relative.parts[0] not in ALLOWED_ROOTS:
            raise ValueError("unexpected_owned_source_entry")
        if source.is_symlink() or not source.is_file():
            raise ValueError("source_snapshot_requires_regular_files")
        data = source.read_bytes()
        validate_public_text(relative.as_posix(), "snapshot path")
        try:
            validate_public_text(data.decode(), relative.as_posix())
        except UnicodeError:
            pass
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "executable": bool(source.stat().st_mode & 0o111),
            }
        )
    lock = json.loads((root / "upstreams.lock.json").read_text())
    return {
        "schema": "roboenv/source-preview/1",
        "destination": {"repository": "stacx-lab/RoboEnv", "visibility": "private"},
        "history_included": False,
        "core_license_declared": False,
        "package_release_approved": False,
        "scope": "private source snapshot, not an open-source or package release",
        "exclusions": EXCLUSIONS,
        "files": sorted(files, key=lambda item: item["path"]),
        "gitlinks": [
            {
                "path": item["path"],
                "revision": item["revision"],
                "url": item["url"],
            }
            for item in lock["sources"].values()
            if not item.get("parent")
        ],
    }


def prepare(root, destination):
    report = inspect(root, verify_upstreams=True, mode="source-preview")
    if not report["private_source_preview_ready"]:
        raise ValueError("source_preview_preflight_failed")
    inventory = manifest(root)
    if destination.exists():
        raise ValueError("snapshot_destination_already_exists")
    destination.mkdir(parents=True)
    for item in inventory["files"]:
        source = root / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("source_changed_during_snapshot")
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.project.resolve()
    if args.out is None:
        (root / ".cache").mkdir(exist_ok=True)
        holder = Path(tempfile.mkdtemp(prefix="private-source-preview-", dir=root / ".cache"))
        destination = holder / "snapshot"
    else:
        destination = args.out.resolve()
    try:
        inventory = prepare(root, destination)
    except Exception as error:
        print(json.dumps({"passed": False, "error": type(error).__name__}))
        return 1
    summary = {key: value for key, value in inventory.items() if key != "files"}
    summary["passed"] = True
    summary["file_count"] = len(inventory["files"])
    summary["total_bytes"] = sum(item["bytes"] for item in inventory["files"])
    if destination.is_relative_to(root):
        summary["snapshot"] = destination.relative_to(root).as_posix()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
