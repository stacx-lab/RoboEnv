"""Image-owned verifier for complete immutable interpreter-depot snapshots."""

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

PREFIX = "/opt/roboenv/runtime/dependencies"
TARGETS = {PREFIX + "/sim-env", PREFIX + "/python-base"}
ALIASES = {"/opt/sim-env": PREFIX + "/sim-env", "/opt/python": PREFIX + "/python-base"}


def _target(value):
    if value not in TARGETS:
        raise ValueError("invalid executable-depot target")
    return value


def _entries(roots):
    """Enumerate without traversing links; every link is resolved against this graph."""
    graph = {}
    for target, root in roots.items():
        _target(target)
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("dependency depot root must be a real directory")
        graph[target] = {"directory": stat.S_IMODE(root.stat().st_mode)}
        for path in sorted(root.rglob("*")):
            name = target + "/" + path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                graph[name] = {"symlink": str(path.readlink())}
            elif stat.S_ISREG(mode):
                graph[name] = {"path": path, "mode": stat.S_IMODE(mode)}
            elif stat.S_ISDIR(mode):
                graph[name] = {"directory": stat.S_IMODE(mode)}
            else:
                raise ValueError("dependency depot contains a special file")
    return graph


def _resolve_link(name, graph):
    """Resolve path components as the kernel would, including links before '..'."""
    pending = list(PurePosixPath(name).parts[1:])
    resolved, expansions = [], 0
    ancestors = {str(parent) for root in TARGETS for parent in PurePosixPath(root).parents}
    while pending:
        component = pending.pop(0)
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved:
                raise ValueError("dependency link escapes its verified target")
            resolved.pop()
            continue
        candidate = "/" + "/".join([*resolved, component])
        if candidate in ALIASES:
            pending = list(PurePosixPath(ALIASES[candidate]).parts[1:]) + pending
            resolved = []
            continue
        record = graph.get(candidate)
        if record is not None and "symlink" in record:
            expansions += 1
            if expansions > 64:
                raise ValueError("dependency link cycle or excessive indirection")
            destination = PurePosixPath(record["symlink"])
            if destination.is_absolute():
                resolved = []
                pending = list(destination.parts[1:]) + pending
            else:
                pending = list(destination.parts) + pending
            continue
        if record is None and candidate not in ancestors:
            raise ValueError("dependency link target is outside the verified manifest set")
        if record is not None and "path" in record and pending:
            raise ValueError("dependency link traverses a non-directory target")
        resolved.append(component)
    terminal = "/" + "/".join(resolved)
    if terminal not in graph or "symlink" in graph[terminal]:
        raise ValueError("dependency link has no verified terminal target")
    return terminal


def snapshot(root, target, peers=None):
    """Hash one depot with links confined to the supplied complete depot group.

    peers maps the two allowed container targets to their physical staging
    directories. Only supplying a root does not certify it: verify_all compares
    every member of the group with its image-owned manifest before execution.
    """
    target = _target(target)
    root = Path(root)
    roots = dict(peers or {target: root})
    if target not in roots or Path(roots[target]).resolve() != root.resolve():
        raise ValueError("snapshot root does not match its declared target")
    graph = _entries(roots)
    records = {}
    for name, record in graph.items():
        if not name.startswith(target + "/"):
            continue
        relative = name[len(target) + 1 :]
        if "symlink" in record:
            _resolve_link(name, graph)
            records[relative] = record
        elif "directory" in record:
            records[relative] = record
        else:
            path = record["path"]
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_mode) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_mode,
            ):
                raise ValueError("dependency file changed during snapshot verification")
            records[relative] = {"sha256": digest.hexdigest(), "mode": stat.S_IMODE(after.st_mode)}
    return {"schema": "roboenv/executable-depot/1", "target": target, "files": records}


def verify_all(manifests, roots=None):
    """Verify all image-owned manifests before any dependency interpreter executes."""
    expected = {}
    for manifest in manifests:
        if not isinstance(manifest, dict) or manifest.get("schema") != "roboenv/executable-depot/1":
            raise ValueError("invalid executable-depot manifest schema")
        target = _target(manifest.get("target"))
        if target in expected:
            raise ValueError("duplicate executable-depot manifest target")
        if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
            raise ValueError("empty executable-depot manifest")
        expected[target] = manifest
    if not expected:
        raise ValueError("missing executable dependency manifests")
    roots = dict(roots) if roots is not None else {target: Path(target) for target in expected}
    if set(roots) != set(expected):
        raise ValueError("every dependency root requires exactly one manifest")
    for target, manifest in expected.items():
        if snapshot(roots[target], target, peers=roots) != manifest:
            raise ValueError("executable dependency snapshot mismatch")


def verify(manifest, *, roots=None):
    verify_all([manifest], roots=roots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.manifest_dir.glob("*.json"))
    verify_all([json.loads(path.read_text()) for path in paths])
    print(json.dumps({"dependency_snapshots_verified": len(paths)}))


if __name__ == "__main__":
    main()
