"""Package a built Kinex checkout and RoboEnv's Harbor bridge, without credentials."""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Runtime files of the Kinex application and of its core workspace package.
APPLICATION = ("package.json", "package-lock.json", "bin", "dist")
CORE = ("package.json", "dist", "assets")
REQUIRED = (
    "dist/application.js",
    "node_modules/@kinex/core/dist/index.js",
    "node_modules/@earendil-works/pi-coding-agent/package.json",
)


def files(base, name, *, skip=()):
    """Pair regular files below base, in a stable order, with their bundle paths."""
    if not base.is_dir():
        return [(base, name)]
    selected = []
    for directory, subdirectories, names in os.walk(base):
        directory = Path(directory)
        subdirectories[:] = sorted(
            entry for entry in subdirectories if entry not in skip and entry != "__pycache__"
        )
        for entry in subdirectories:
            if (directory / entry).is_symlink():
                raise ValueError(f"Kinex bundle input links a directory: {directory / entry}")
        selected += [
            (directory / entry, name + "/" + (directory / entry).relative_to(base).as_posix())
            for entry in sorted(names)
            if not entry.endswith(".pyc")
        ]
    return selected


def dependencies(kinex):
    """Map installed production packages from package-lock.json to bundle paths.

    Development-only packages are omitted. Workspace links are materialized where
    Node resolves them, because bundles contain only regular files.
    """
    lock = json.loads((kinex / "package-lock.json").read_text())
    if lock.get("lockfileVersion") != 3:
        raise ValueError("Kinex bundles require an npm lockfile version 3")
    packages = lock["packages"]
    selected = []
    for key, entry in packages.items():
        if not key.startswith("node_modules/") or entry.get("dev"):
            continue
        if entry.get("link"):
            workspace = kinex / entry["resolved"]
            if any(other.startswith(entry["resolved"] + "/node_modules/") for other in packages):
                raise ValueError(f"Unsupported nested dependencies in workspace {workspace}")
            sources = [(workspace / name, key + "/" + name) for name in CORE]
            expected = packages[entry["resolved"]]["version"]
        else:
            workspace = kinex / key
            if not workspace.exists() and entry.get("optional"):
                continue  # An optional package for another platform.
            sources = [(workspace, key)]
            expected = entry["version"]
        installed = workspace / "package.json"
        if not installed.is_file() or json.loads(installed.read_text())["version"] != expected:
            raise ValueError(f"{key} differs from package-lock.json; run npm ci in the checkout")
        for base, name in sources:
            if not base.exists():
                raise ValueError(f"Kinex bundle input is missing: {base.relative_to(kinex)}")
            selected += files(base, name, skip={"node_modules"})
    return selected


def check_node(node, kinex):
    """Require the Node version Kinex declares; the bundle must not guess."""
    engines = json.loads((kinex / "package.json").read_text())["engines"]["node"]
    minimum = re.fullmatch(r">=(\d+)\.(\d+)\.(\d+)", engines)
    output = subprocess.run([node, "--version"], capture_output=True, text=True, check=True)
    version = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", output.stdout.strip())
    if not minimum or not version:
        raise ValueError(f"Cannot compare Node {output.stdout.strip()} with {engines}")
    if tuple(map(int, version.groups())) < tuple(map(int, minimum.groups())):
        raise ValueError(f"Kinex requires Node {engines}; {node} is {output.stdout.strip()}")


def package(kinex, node, uv, destination):
    kinex, node, uv = (Path(value).resolve() for value in (kinex, node, uv))
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Select a new bundle path; existing bundles are preserved")
    check_node(node, kinex)
    subprocess.run([uv, "--version"], capture_output=True, check=True)
    selected = []
    for relative in APPLICATION:
        base = kinex / relative
        if not base.exists():
            raise ValueError(f"Kinex bundle input is missing: {relative}")
        selected += files(base, "kinex/" + relative)
    selected += [(file, "kinex/" + name) for file, name in dependencies(kinex)]
    names = {name for _, name in selected}
    for relative in REQUIRED:
        if "kinex/" + relative not in names:
            raise ValueError(
                f"Kinex is not built or installed: {relative}; run npm ci and npm run build"
            )
    for file, _ in selected:
        if not file.resolve().is_relative_to(kinex):
            raise ValueError("Kinex bundle input links outside the selected checkout")
    selected += files(ROOT / "integrations/kinex", "kinex/integrations/roboenv")
    selected += [(node, "bin/node"), (uv, "bin/uv")]
    checksums = {name: hashlib.sha256(file.read_bytes()).hexdigest() for file, name in selected}
    if len(checksums) != len(selected):
        raise ValueError("Kinex bundle inputs map to duplicate paths")
    manifest = {
        "schema": "roboenv/kinex-agent-bundle/1",
        "files": checksums,
        "identity": {
            "kinex_sha256": hashlib.sha256(
                json.dumps(checksums, sort_keys=True).encode()
            ).hexdigest()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Explicit allowlist excludes .git, auth, user config, recordings and caches.
    with tarfile.open(destination, "x:gz", dereference=True) as archive:
        for file, name in selected:
            archive.add(file, arcname=name, recursive=False)
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("bundle.json")
        entry.size, entry.mode = len(data), 0o644
        archive.addfile(entry, io.BytesIO(data))
    return {
        "bundle": str(destination),
        "files": len(checksums),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        **manifest["identity"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kinex", required=True, type=Path, help="Checkout after npm ci and npm run build"
    )
    parser.add_argument(
        "--node", required=True, type=Path, help="Node executable for the agent image"
    )
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.kinex, args.node, args.uv, args.out), indent=2))
