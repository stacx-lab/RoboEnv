"""Prepare T2 assets from an independent depot; no original project is imported."""

import hashlib
import json
import os
import shutil
from importlib.resources import files
from pathlib import Path


def asset_root(project):
    cache = Path(os.environ.get("ROBOENV_ASSET_CACHE", Path(project) / ".cache/roboenv/assets"))
    return cache / "booster-t2" / "0bbcdc7016d9"


def asset_pin():
    return json.loads(files("roboenv").joinpath("assets/booster-t2-assets.json").read_text())


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory, *, scenes=True):
    directory = Path(directory).resolve()
    pin = asset_pin()
    entries = dict(pin["files"])
    if scenes:
        entries.update(pin.get("scene_files", {}))
        if not pin.get("scene_files"):
            raise ValueError("T2 scene assets have not been pinned")
    for name, expected in entries.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing T2 asset: {name}; run tools/prepare_booster_t2.py")
        if checksum(path) != expected:
            raise ValueError(f"T2 asset checksum mismatch: {name}")
    return {
        "files": len(entries),
        "asset_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def prepare_assets(project, *, source=None, scenes=True):
    """Copy a verified portable depot, preserving any different existing files."""
    target = asset_root(project)
    source = Path(source).resolve() if source else target.resolve()
    identity = verify_assets(source, scenes=scenes)
    if source != target.resolve():
        entries = dict(asset_pin()["files"])
        if scenes:
            entries.update(asset_pin()["scene_files"])
        for name, expected in entries.items():
            destination = target / name
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"T2 asset destination escapes the depot: {name}")
            if destination.exists():
                if checksum(destination) != expected:
                    raise ValueError(f"Existing T2 asset differs: {name}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
    return {"selected_assets_verified": True, "asset_directory": str(target.resolve()), **identity}
