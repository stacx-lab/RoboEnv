"""Package an explicit Kinex checkout and RoboEnv's Harbor bridge, without credentials."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "dist",
    "packages/core/dist",
    "packages/core/assets",
    "packages/core/package.json",
    "packages/pi/dist",
    "packages/pi/package.json",
    "assets",
    "bin",
    "package.json",
    "package-lock.json",
    ".nvmrc",
    "node_modules",
)


def package(kinex, node, uv, destination):
    kinex, node, uv = (Path(value).resolve() for value in (kinex, node, uv))
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Select a new bundle path; existing bundles are preserved")
    version = subprocess.check_output([str(node), "--version"], text=True).strip()
    if tuple(map(int, version.removeprefix("v").split("."))) < (22, 19, 0):
        raise ValueError("Kinex requires Node >=22.19")
    for relative in ("dist/cli.js", "dist/application.js", "node_modules/@kinex/core/dist/index.js",
                     "node_modules/@kinex/pi/dist/owner.js"):
        if not (kinex / relative).is_file():
            raise ValueError("Build Kinex first with npm ci && npm run build")
    selected = []

    def files(base, ancestors=()):
        resolved = base.resolve()
        if not resolved.is_relative_to(kinex):
            raise ValueError("Kinex bundle input links outside the selected checkout")
        if base.name == "__pycache__" or base.suffix == ".pyc":
            return
        if base.is_file():
            yield base
        elif base.is_dir():
            if resolved in ancestors:
                raise ValueError("Kinex bundle input contains a directory link cycle")
            for child in sorted(base.iterdir()):
                yield from files(child, (*ancestors, resolved))

    for relative in SOURCES:
        base = kinex / relative
        if not base.exists():
            raise ValueError(f"Kinex bundle input is missing: {relative}")
        # Materialize npm workspace links inside the archive so relocation keeps
        # @kinex/core resolvable without relying on a host checkout.
        for file in files(base):
            selected.append((file, "kinex/" + file.relative_to(kinex).as_posix()))
    for file in sorted((ROOT / "integrations/kinex").rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
            selected.append(
                (
                    file,
                    "kinex/integrations/roboenv/"
                    + file.relative_to(ROOT / "integrations/kinex").as_posix(),
                )
            )
    selected += [(node, "bin/node"), (uv, "bin/uv")]
    checksums = {name: hashlib.sha256(file.read_bytes()).hexdigest() for file, name in selected}
    manifest = {
        "schema": "roboenv/kinex-agent-bundle/2",
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
        import io

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
    parser.add_argument("--kinex", required=True, type=Path)
    parser.add_argument("--node", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.kinex, args.node, args.uv, args.out), indent=2))
