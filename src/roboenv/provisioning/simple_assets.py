"""Pinned SIMPLE data acquisition; launch never downloads mutable upstream assets."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from roboenv.provisioning.assets import inventory


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract(archive, destination):
    """Validate every member before writing, rejecting links and path traversal."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            path = Path(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
                or not (destination / path).resolve().is_relative_to(destination)
            ):
                raise ValueError("Unsafe SIMPLE archive member")
        source.extractall(destination)


def _link(link, target):
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def expose_native_layout(root, data):
    """Make the verified depot visible where SIMPLE's Isaac engine resolves data.

    ``SIMPLE_DATA_DIR`` only steers SIMPLE's MuJoCo paths. The Isaac engine resolves
    robots, scenes and MDL materials relative to the package checkout
    (``third_party/SIMPLE/data``, ignored by its .gitignore). Each top-level depot
    directory is exposed there with a relative symlink; verified files are never
    moved, copied or overwritten, and existing entries are kept.
    """
    data = Path(data)
    package = Path(root) / "third_party/SIMPLE"
    if package.is_dir():
        package_data = package / "data"
        for child in sorted(data.iterdir()):
            if child.is_dir():
                _link(package_data / child.name, Path(os.path.relpath(child, package_data)))


def prepare_assets(root, *, progress=None, archives=None):
    root = Path(root).resolve()
    pin_path = root / "src/roboenv/assets/simple-g1.json"
    pin = json.loads(pin_path.read_text())
    depot = root / ".cache/simple"
    depot.mkdir(parents=True, exist_ok=True)
    selected = pin["archives"]
    if archives is not None:
        if set(archives) - {entry["file"] for entry in selected}:
            raise ValueError("Unknown SIMPLE archive")
        selected = [entry for entry in selected if entry["file"] in archives]

    def acquire(entry):
        archive = depot / "downloads" / entry["file"]
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            if progress:
                progress(f"Downloading pinned SIMPLE archive: {entry['file']}")
            url = (
                f"https://huggingface.co/datasets/{pin['repository']}/resolve/"
                f"{pin['revision']}/{entry['file']}"
            )
            with tempfile.TemporaryDirectory(dir=archive.parent) as temporary:
                pending = Path(temporary) / "download"
                with urlopen(url, timeout=120) as response, pending.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                if pending.stat().st_size != entry["size"] or digest(pending) != entry["sha256"]:
                    raise ValueError(f"SIMPLE download checksum mismatch: {entry['file']}")
                pending.replace(archive)
        if archive.stat().st_size != entry["size"] or digest(archive) != entry["sha256"]:
            raise ValueError(f"Existing SIMPLE archive differs from pin: {entry['file']}")
        return entry, archive

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        downloads = list(pool.map(acquire, selected))
    data = depot / "data"
    for entry, archive in downloads:
        receipt = depot / (entry["file"] + ".json")
        if receipt.exists() and digest(receipt) == entry["inventory_sha256"]:
            for item in json.loads(receipt.read_text()):
                path = data / item["path"]
                if not path.is_file() or path.is_symlink() or digest(path) != item["sha256"]:
                    raise ValueError(f"SIMPLE extracted asset changed: {item['path']}")
            continue
        if receipt.exists() and progress:
            # The pinned layout changed (for example vMaterials_2 moving to the data
            # root). The archive is checksum-verified, so extract it again: files
            # already present and matching are kept, the receipt is rewritten.
            progress(f"Refreshing SIMPLE assets for the current pin: {entry['file']}")
        with tempfile.TemporaryDirectory(dir=depot) as temporary:
            staged = Path(temporary) / "data"
            extract(archive, staged / entry["destination"])
            entries = inventory(staged)
            serialized = json.dumps(entries, indent=2) + "\n"
            if hashlib.sha256(serialized.encode()).hexdigest() != entry["inventory_sha256"]:
                raise ValueError("SIMPLE extracted inventory differs from pin")
            for item in entries:
                path = data / item["path"]
                if path.exists() and (path.is_symlink() or digest(path) != item["sha256"]):
                    raise ValueError(f"Refusing to overwrite SIMPLE asset: {item['path']}")
            for item in entries:
                path = data / item["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    shutil.copyfile(staged / item["path"], path)
            receipt.write_text(serialized)
    complete = all((depot / (entry["file"] + ".json")).is_file() for entry in pin["archives"])
    if complete:
        expose_native_layout(root, data)
    return {
        "environment_assets_verified": complete,
        "asset_depot": str(data),
        "asset_manifest_sha256": digest(pin_path),
    }


def verify_assets(root):
    root = Path(root)
    pin = json.loads((root / "src/roboenv/assets/simple-g1.json").read_text())
    depot = root / ".cache/simple"
    for entry in pin["archives"]:
        receipt = depot / (entry["file"] + ".json")
        if not receipt.is_file():
            raise ValueError("SIMPLE assets missing; run roboenv prepare simple")
        if digest(receipt) != entry["inventory_sha256"]:
            raise ValueError(
                "SIMPLE inventory differs from the current pin; run roboenv prepare simple"
            )
        for item in json.loads(receipt.read_text()):
            path = depot / "data" / item["path"]
            if not path.is_file() or path.is_symlink() or digest(path) != item["sha256"]:
                raise ValueError(f"SIMPLE asset verification failed: {item['path']}")
    expose_native_layout(root, depot / "data")
    return depot / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--archives", nargs="+")
    args = parser.parse_args()
    print(json.dumps(prepare_assets(args.project, archives=args.archives, progress=print)))


if __name__ == "__main__":
    main()
