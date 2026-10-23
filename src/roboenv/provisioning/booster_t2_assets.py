"""Prepare T2 assets from an independent depot; no original project is imported."""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
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


def selected_files(*, scenes=True, articulated=True):
    pin = asset_pin()
    entries = dict(pin["files"])
    if scenes:
        entries.update(pin.get("scene_files", {}))
        if not pin.get("scene_files"):
            raise ValueError("T2 scene assets have not been pinned")
    if articulated:
        entries.update(pin["articulated_files"])
    return entries


def verify_assets(directory, *, scenes=True, articulated=False):
    directory = Path(directory).resolve()
    entries = selected_files(scenes=scenes, articulated=articulated)
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


def _download(url, destination, expected):
    headers = {}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github.raw+json"
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if checksum(destination) != expected:
        raise ValueError(f"Downloaded T2 asset checksum mismatch: {destination.name}")


def download_assets(directory, *, scenes=True, articulated=True):
    """Fetch missing pinned files; never replace an existing, different asset."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pin = asset_pin()
    entries = selected_files(scenes=scenes, articulated=articulated)
    missing = {}
    for name, expected in entries.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError(f"T2 asset destination escapes the depot: {name}")
        if path.exists():
            if checksum(path) != expected:
                raise ValueError(f"Existing T2 asset differs: {name}")
        else:
            missing[name] = expected
    source = pin["scene_source"]
    github = "https://api.github.com/repos/ZhenyuWu323/locomanip_agentic/contents"
    dataset = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}"
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=directory) as staging:
        staging = Path(staging)
        archives = {}
        try:
            for name, expected in missing.items():
                temporary = staging / "asset"
                if name.startswith("scenes/hssd/") or name.startswith("scenes/articulated/"):
                    archive_name = (
                        f"scenes_hssd_{Path(name).parts[2]}.zip"
                        if name.startswith("scenes/hssd/")
                        else "assets_articulated.zip"
                    )
                    if archive_name not in archives:
                        archive = staging / archive_name
                        _download(
                            f"{dataset}/{archive_name}", archive, source["downloads"][archive_name]
                        )
                        archives[archive_name] = zipfile.ZipFile(archive)
                    member = name.replace("scenes/articulated/", "assets/articulated/", 1)
                    # Extract only a manifest-named regular file, never archive paths or symlinks.
                    with (
                        archives[archive_name].open(member) as stream,
                        temporary.open("wb") as output,
                    ):
                        shutil.copyfileobj(stream, output)
                    if checksum(temporary) != expected:
                        raise ValueError(f"Extracted T2 asset checksum mismatch: {name}")
                else:
                    url = (
                        f"{dataset}/assets/graspnet/ruled_models/{name.removeprefix('scenes/graspnet/')}"
                        if name.startswith("scenes/graspnet/")
                        else f"{github}/{name}?ref={pin['source_revision']}"
                    )
                    _download(url, temporary, expected)
                destination = directory / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects a concurrently prepared depot.
                try:
                    with destination.open("xb") as output, temporary.open("rb") as stream:
                        shutil.copyfileobj(stream, output)
                except FileExistsError:
                    if checksum(destination) != expected:
                        raise ValueError(f"Existing T2 asset differs: {name}") from None
        finally:
            for archive in archives.values():
                archive.close()
    return verify_assets(directory, scenes=scenes, articulated=articulated)


def prepare_assets(project, *, source=None, scenes=True, articulated=True, download=False):
    """Copy a verified portable depot, preserving any different existing files."""
    target = asset_root(project)
    if download and source is not None:
        raise ValueError("Select either --download or --source")
    if download:
        download_assets(target, scenes=scenes, articulated=articulated)
    source = Path(source).resolve() if source else target.resolve()
    identity = verify_assets(source, scenes=scenes, articulated=articulated)
    if source != target.resolve():
        entries = selected_files(scenes=scenes, articulated=articulated)
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
