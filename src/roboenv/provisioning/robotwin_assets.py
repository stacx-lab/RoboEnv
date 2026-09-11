"""Pinned public RoboTwin assets for the clean-background device protocol."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.progress import stage
from roboenv.provisioning.robocasa_assets import _target

REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
ARCHIVES = {
    "embodiments": (219859313, "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"),
    "objects": (3737778549, "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"),
}


def prepare_assets(project, *, progress=None, cache=None):
    project = Path(project)
    cache = Path(cache) if cache else (
        Path(os.environ.get("ROBOENV_ASSET_CACHE", project / ".cache/roboenv/assets")) / "robotwin"
    )
    cache.mkdir(parents=True, exist_ok=True)
    assets = project / "third_party/RoboTwin/assets"
    receipts = []
    for name, (size, digest) in ARCHIVES.items():
        archive = cache / (name + ".zip")
        with stage(progress, f"Verifying RoboTwin {name} archive"):
            verified = (
                archive.is_file() and archive.stat().st_size == size and _digest(archive) == digest
            )
        if not verified:
            if archive.exists():
                raise ValueError(
                    "Existing asset archive differs from the pin; preserve and inspect it"
                )
            if shutil.disk_usage(cache).free < size + 1024**3:
                raise ValueError(
                    "Insufficient space for asset download; choose a larger asset cache"
                )
            with stage(progress, f"Downloading RoboTwin {name} ({size / 1e9:.2f} GB)"):
                url = f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REVISION}/{name}.zip"
                with tempfile.TemporaryDirectory(prefix=name + "-download-", dir=cache) as download:
                    pending = Path(download) / (name + ".zip")
                    with urlopen(url, timeout=60) as response, pending.open("xb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                    if pending.stat().st_size != size or _digest(pending) != digest:
                        raise ValueError("Downloaded asset archive failed SHA-256 verification")
                    pending.replace(archive)
        destination = assets / name
        with stage(progress, f"Verifying and installing RoboTwin {name}"):
            with ZipFile(archive) as bundle:
                entries = []
                for item in bundle.infolist():
                    _target(cache, item.filename)
                    if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                        raise ValueError("Asset archive contains a symlink")
                    if (
                        Path(item.filename).parts[0] == "__MACOSX"
                        or Path(item.filename).name == ".DS_Store"
                    ):
                        continue
                    if Path(item.filename).parts[0] != name:
                        raise ValueError("Unexpected asset archive root")
                    entries.append(item)
                files = [item for item in entries if not item.is_dir()]
                if not files:
                    raise ValueError("Empty asset archive")
                if destination.exists() or destination.is_symlink():
                    if not destination.is_dir():
                        raise ValueError("Asset destination is not a readable directory")
                    # Existing user assets are verified, never overwritten.
                    for item in files:
                        relative = Path(item.filename).relative_to(name)
                        path = destination / relative
                        if (
                            not path.is_file()
                            or _digest(path) != hashlib.sha256(bundle.read(item)).hexdigest()
                        ):
                            raise ValueError(f"Existing RoboTwin {name} assets differ from the pin")
                else:
                    if (
                        shutil.disk_usage(cache).free
                        < sum(item.file_size for item in files) + 1024**3
                    ):
                        raise ValueError(
                            "Insufficient extraction space; choose a larger asset cache"
                        )
                    installed = cache / (name + "-" + digest[:12])
                    if installed.exists():
                        raise ValueError(
                            "Unreferenced asset destination exists; inspect before reuse"
                        )
                    with tempfile.TemporaryDirectory(prefix=name + "-", dir=cache) as temporary:
                        bundle.extractall(temporary, members=entries)
                        os.rename(Path(temporary) / name, installed)
                    destination.symlink_to(installed.resolve(), target_is_directory=True)
                record = {
                    "schema": "roboenv/robotwin-assets/1",
                    "revision": REVISION,
                    "archive_sha256": digest,
                    "files": {
                        str(Path(item.filename).relative_to(name)): _digest(
                            destination / Path(item.filename).relative_to(name)
                        )
                        for item in files
                    },
                }
            receipt = cache / (name + ".json")
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            receipts.append(str(receipt))
    return {
        "environment_assets_verified": True,
        "asset_receipts": receipts,
        "asset_scope": "demo_clean; randomized background textures excluded",
        "gpu_replay_verified": False,
    }
