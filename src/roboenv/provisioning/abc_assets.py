"""Install the checksum-pinned ABC simulator archives, preserving existing assets."""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.progress import stage
from roboenv.provisioning.robocasa_assets import _target


def prepare_assets(project, *, progress=None, cache=None):
    project = Path(project)
    models = project / "third_party/abc/abc_sim/models"
    manifest = json.loads((models / "assets_manifest.json").read_text())
    cache = (
        Path(cache)
        if cache
        else Path(os.environ.get("ROBOENV_ASSET_CACHE", project / ".cache/roboenv/assets")) / "abc"
    )
    cache.mkdir(parents=True, exist_ok=True)
    assets = models / "assets"
    installed = cache / "installed"
    installed.mkdir(exist_ok=True)
    if assets.is_symlink() and assets.resolve() == installed.resolve():
        assets.unlink()
    assets.mkdir(exist_ok=True)
    for package in manifest["packages"]:
        for directory in package["contains"]:
            if Path(directory).name != directory or directory in {".", ".."}:
                raise ValueError("Unsafe ABC asset directory")
            destination = assets / directory
            if not destination.exists() and not destination.is_symlink():
                depot = _target(installed, directory)
                depot.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(depot.resolve(), target_is_directory=True)
    receipts = []
    for package in manifest["packages"]:
        name = package["name"]
        archive = _target(cache, package["archive"])
        with stage(progress, f"Verifying ABC assets: {name}"):
            if not archive.exists():
                with tempfile.TemporaryDirectory(dir=cache, prefix="download-") as temporary:
                    pending = Path(temporary) / "archive"
                    request = Request(package["uri"], headers={"User-Agent": "RoboEnv/0.10"})
                    with urlopen(request, timeout=60) as response, pending.open("xb") as out:
                        shutil.copyfileobj(response, out)
                    if (
                        pending.stat().st_size != package["size_bytes"]
                        or _digest(pending) != package["sha256"]
                    ):
                        raise ValueError("ABC asset download failed checksum verification")
                    pending.replace(archive)
            if _digest(archive) != package["sha256"]:
                raise ValueError(
                    "Existing ABC archive differs from its pin; preserve and inspect it"
                )
            files = {}
            with tarfile.open(archive) as bundle:
                entries = bundle.getmembers()
                for item in entries:
                    _asset_target(assets, item.name)
                    if (not item.isdir() and not item.isfile()) or (
                        Path(item.name).parts[0] not in package["contains"]
                    ):
                        raise ValueError("Unsupported ABC archive entry")
                for item in entries:
                    if not item.isfile():
                        continue
                    destination = _asset_target(assets, item.name)
                    with bundle.extractfile(item) as source:
                        content = source.read()
                    digest = hashlib.sha256(content).hexdigest()
                    if destination.exists():
                        if not destination.is_file() or _digest(destination) != digest:
                            raise ValueError(f"Existing ABC asset differs from pin: {item.name}")
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            dir=destination.parent, delete=False
                        ) as out:
                            pending = Path(out.name)
                            out.write(content)
                        pending.replace(destination)
                    files[item.name] = digest
            if len(files) != package["file_count"]:
                raise ValueError("ABC archive file count differs from manifest")
            receipt = cache / (name + ".json")
            receipt.write_text(
                json.dumps({"archive_sha256": package["sha256"], "files": files}) + "\n"
            )
            receipts.append(str(receipt))
    return {
        "environment_assets_verified": True,
        "asset_receipts": receipts,
        "gpu_replay_verified": False,
    }


def _asset_target(assets, name):
    parts = Path(name).parts
    if not parts or parts[0] in {".", ".."} or Path(name).is_absolute():
        raise ValueError("Unsafe asset path")
    return _target(assets / parts[0], str(Path(*parts[1:])))
