"""Portable content receipts for independently acquired simulator asset depots."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from roboenv.privacy import validate_public_text


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    """Hash files and in-depot links; never follow a link outside the depot."""
    root = Path(root).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("assets require an explicit non-root directory")
    if (root / ".git").exists() or (root / "pyproject.toml").exists():
        raise ValueError("use a data-only depot, not a source checkout")
    entries = []
    for directory, children, files in os.walk(root, followlinks=False):
        children.sort()
        for name in sorted(children + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            validate_public_text(relative, "asset entry")
            if path.is_symlink():
                target = os.readlink(path)
                if Path(target).is_absolute() or not path.resolve().is_relative_to(root):
                    raise ValueError("asset link escapes its depot or uses a host path")
                if not path.exists():
                    raise ValueError("asset depot contains a dangling link")
                entries.append({"path": relative, "kind": "symlink", "target": target})
            elif path.is_file():
                before = path.stat()
                digest = _digest(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("asset changed during hashing; quiesce the depot and retry")
                entries.append(
                    {"path": relative, "kind": "file", "bytes": after.st_size, "sha256": digest}
                )
            elif not path.is_dir():
                raise ValueError("asset depot contains a special file")
    return sorted(entries, key=lambda entry: entry["path"])


def create(root, *, asset_id, source_url, source_revision, license_url):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", asset_id):
        raise ValueError("asset id must be a lowercase portable identifier")
    if not source_revision.strip():
        raise ValueError("record the source revision or published asset version")
    for value in (source_url, license_url):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("source/license URLs must be HTTPS without credentials or tokens")
    metadata = {
        "schema": "roboenv/assets/1",
        "asset_id": asset_id,
        "source": {"url": source_url, "revision": source_revision},
        "license_url": license_url,
        "redistribution_reviewed": False,
    }
    validate_public_text(json.dumps(metadata), "asset provenance")
    entries = inventory(root)
    return {
        **metadata,
        "entries": entries,
        "content_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def verify(root, manifest):
    manifest = json.loads(Path(manifest).read_text())
    if manifest.get("schema") != "roboenv/assets/1":
        raise ValueError("unsupported asset manifest schema")
    actual = inventory(root)
    expected = manifest.get("entries")
    digest = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "roboenv/asset-verification/1",
        "asset_id": manifest.get("asset_id"),
        "passed": actual == expected and digest == manifest.get("content_sha256"),
        "entries": len(actual),
        "content_sha256": digest,
        "license_permission_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("root", type=Path)
    make.add_argument("--out", required=True, type=Path)
    for name in ("asset-id", "source-url", "source-revision", "license-url"):
        make.add_argument("--" + name, required=True)
    check = commands.add_parser("verify")
    check.add_argument("root", type=Path)
    check.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            if args.out.exists() or args.out.resolve().is_relative_to(args.root.resolve()):
                raise ValueError("manifest output must be new and outside the asset depot")
            report = create(
                args.root,
                asset_id=args.asset_id,
                source_url=args.source_url,
                source_revision=args.source_revision,
                license_url=args.license_url,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as stream:
                stream.write(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "asset_id": report["asset_id"],
                        "entries": len(report["entries"]),
                        "content_sha256": report["content_sha256"],
                    }
                )
            )
            return 0
        report = verify(args.root, args.manifest)
    except (ValueError, OSError) as error:
        if isinstance(error, OSError):
            parser.exit(1, "asset manifest I/O failed; check input/output permissions\n")
        parser.exit(1, f"{error}\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
