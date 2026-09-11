"""Package an explicit Kinex checkout and RoboEnv's Harbor bridge, without credentials."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src",
    "python",
    "builtin",
    "bin",
    "package.json",
    "bun.lock",
    "tsconfig.json",
    "node_modules",
)


def package(kinex, bun, uv, destination):
    kinex, bun, uv = (Path(value).resolve() for value in (kinex, bun, uv))
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Select a new bundle path; existing bundles are preserved")
    selected = []
    for relative in SOURCES:
        base = kinex / relative
        if not base.exists():
            raise ValueError(f"Kinex bundle input is missing: {relative}")
        for file in sorted(base.rglob("*")) if base.is_dir() else [base]:
            if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            if not file.resolve().is_relative_to(kinex):
                raise ValueError("Kinex bundle input links outside the selected checkout")
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
    selected += [(bun, "bin/bun"), (uv, "bin/uv")]
    checksums = {name: hashlib.sha256(file.read_bytes()).hexdigest() for file, name in selected}
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
    parser.add_argument("--bun", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.kinex, args.bun, args.uv, args.out), indent=2))
