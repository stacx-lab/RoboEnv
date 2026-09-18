"""Acquire the kitchen asset archives listed by the pinned RoboCasa checkout."""

import json
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.progress import stage

# Same archive selection and destinations as download_kitchen_assets.py.
ARCHIVES = {
    "textures": "textures",
    "generative_textures": "generative_textures",
    "fixtures_lightwheel": "fixtures",
    "objaverse": "objects/objaverse",
    "aigen_objs": "objects/aigen_objs",
    "objects_lightwheel": "objects/lightwheel",
}


def _target(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe asset archive path")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset path escapes its destination")
    return target


def _valid(root, receipt, url):
    if not receipt.is_file():
        return False
    record = json.loads(receipt.read_text())
    return (
        record.get("url") == url
        and bool(record.get("files"))
        and all(
            (path := _target(root, item["path"])).is_file() and _digest(path) == item["sha256"]
            for item in record["files"]
        )
    )


def _reuse(parent, receipt, source_parent, source_receipt, url):
    """Import a verified bundle; publish its receipt only after copying succeeds."""
    if not _valid(source_parent, source_receipt, url):
        return False
    record = json.loads(source_receipt.read_text())
    for item in record["files"]:
        source, target = _target(source_parent, item["path"]), _target(parent, item["path"])
        if target.is_file() and _digest(target) == item["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".asset-", delete=False) as f:
            pending = Path(f.name)
        try:
            shutil.copyfile(source, pending)
            if _digest(pending) != item["sha256"]:
                raise ValueError("source asset changed during reuse")
            pending.chmod(0o644)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="reuse-", dir=receipt.parent) as temporary:
        pending = Path(temporary) / "receipt.json"
        pending.write_text(json.dumps(record) + "\n")
        pending.chmod(0o600)
        pending.replace(receipt)
    return True


def prepare_assets(project, *, progress=None, reuse_from=None):
    """Verify local bundles, reuse a prepared project, then download only missing bundles.

    ``reuse_from`` is another RoboEnv project. Imports require matching source URLs
    and receipt hashes; its interpreter and private state are never copied.
    """
    project = Path(project)
    reuse_from = Path(reuse_from) if reuse_from else None
    assets = project / "third_party/robocasa/robocasa/models/assets"
    links = json.loads((assets / "box_links/box_links_assets.json").read_text())
    # Validate every source before causing download or filesystem effects.
    urls = {}
    for key in ARCHIVES:
        match = re.fullmatch(r"https://utexas\.box\.com/s/([A-Za-z0-9]+)", links[key])
        if match is None:
            raise ValueError("unsupported RoboCasa asset URL in pinned source")
        urls[key] = f"https://utexas.box.com/shared/static/{match[1]}.zip"
    cache = project / ".cache/roboenv/assets/robocasa"
    cache.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Preparing RoboCasa kitchen assets (upstream estimates about 10 GB).")
    for key, relative in ARCHIVES.items():
        parent = (assets / relative).parent
        receipt = cache / f"{key}.json"
        with stage(progress, f"Checking RoboCasa assets: {key}"):
            ready = _valid(parent, receipt, urls[key])
        if ready:
            continue
        if reuse_from and reuse_from.resolve() != project.resolve():
            source_assets = reuse_from / "third_party/robocasa/robocasa/models/assets"
            source_receipt = reuse_from / ".cache/roboenv/assets/robocasa" / f"{key}.json"
            with stage(progress, f"Reusing prepared RoboCasa assets: {key}"):
                if _reuse(parent, receipt, (source_assets / relative).parent,
                          source_receipt, urls[key]):
                    continue
        with stage(progress, f"Downloading and extracting RoboCasa assets: {key}"):
            try:
                with tempfile.TemporaryDirectory(prefix=key + "-", dir=cache) as temporary:
                    archive = Path(temporary) / "assets.zip"
                    with urlopen(urls[key], timeout=60) as response, archive.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    with ZipFile(archive) as bundle:
                        entries = bundle.infolist()
                        selected = []
                        for item in entries:
                            _target(parent, item.filename)
                            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                                raise ValueError("asset archive contains a symbolic link")
                            # Upstream includes a README beside fixtures/. It is
                            # documentation, not an asset to install into the shared root.
                            if item.filename == "README.md" and not item.is_dir():
                                continue
                            if Path(item.filename).parts[0] != Path(relative).name:
                                raise ValueError("unexpected asset archive root")
                            selected.append(item)
                        files = [item for item in selected if not item.is_dir()]
                        if not files:
                            raise ValueError("empty asset archive")
                        # Validate CRCs before writing any installed files.
                        if bundle.testzip() is not None:
                            raise ValueError("corrupt asset archive")
                        bundle.extractall(parent, members=selected)
                    record = {
                        "url": urls[key],
                        "files": [
                            {"path": item.filename, "sha256": _digest(parent / item.filename)}
                            for item in files
                        ],
                    }
                    pending = Path(temporary) / "receipt.json"
                    pending.write_text(json.dumps(record) + "\n")
                    pending.chmod(0o600)
                    pending.replace(receipt)
            except Exception as error:
                logfile = cache / f"{key}.private.log"
                with logfile.open("w") as log:
                    logfile.chmod(0o600)
                    traceback.print_exc(file=log)
                raise RuntimeError(
                    f"RoboCasa asset preparation failed for {key}: {type(error).__name__}; "
                    f"inspect {logfile.relative_to(project).as_posix()}; "
                    "rerun 'roboenv prepare robocasa' to retry. "
                    "The installed interpreter and completed asset bundles were preserved."
                ) from error
    return {"downloaded_assets_verified": True, "gpu_replay_verified": False}
