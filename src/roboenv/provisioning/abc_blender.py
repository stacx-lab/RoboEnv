"""Optional, checksum-pinned Blender and studio lighting for ABC."""

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.progress import stage

VERSION = "4.5.3"
ARCHIVE = f"blender-{VERSION}-linux-x64.tar.xz"
BLENDER_SHA256 = "975c58fcb244273838534bba771e64ad87739216b0f9b39a888531a49a72d845"
HDRI = "photo_studio_01_2k.hdr"
HDRI_SHA256 = "b6178cf04ea2ac9390b8794d3088a04c0254905f335f5f30d2b582c57c40f387"


def cache_root(project):
    return (
        Path(os.environ.get("ROBOENV_ASSET_CACHE", Path(project) / ".cache/roboenv/assets"))
        / "abc/blender"
    )


def paths(project):
    root = cache_root(project)
    executable = os.environ.get("BLENDER") or str(root / f"blender-{VERSION}-linux-x64/blender")
    binary = Path(shutil.which(executable) or executable).expanduser().resolve()
    hdri = Path(os.environ.get("ABC_HDRI") or root / HDRI).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK) or not hdri.is_file():
        raise ValueError("ABC Blender is not prepared; run roboenv prepare abc --renderer blender")
    return binary, hdri


def _download(url, target, checksum):
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=target.parent, prefix="download-") as temporary:
            pending = Path(temporary) / target.name
            request = Request(url, headers={"User-Agent": "RoboEnv/0.10"})
            with urlopen(request, timeout=60) as response, pending.open("xb") as stream:
                shutil.copyfileobj(response, stream)
            if _digest(pending) != checksum:
                raise ValueError(f"Blender asset checksum mismatch: {target.name}")
            pending.replace(target)
    if _digest(target) != checksum:
        raise ValueError(f"Existing Blender asset differs from its pin: {target.name}")


def prepare(project, *, progress=None):
    root = cache_root(project)
    root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("BLENDER"):
        with stage(progress, "Preparing pinned Blender 4.5.3"):
            archive = root / ARCHIVE
            _download(
                f"https://download.blender.org/release/Blender4.5/{ARCHIVE}",
                archive,
                BLENDER_SHA256,
            )
            destination = root / f"blender-{VERSION}-linux-x64"
            if not destination.exists():
                with tempfile.TemporaryDirectory(dir=root, prefix="extract-") as temporary:
                    with tarfile.open(archive) as bundle:
                        bundle.extractall(temporary, filter="data")
                    (Path(temporary) / destination.name).rename(destination)
    if not os.environ.get("ABC_HDRI"):
        with stage(progress, "Preparing studio lighting"):
            _download(
                f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/{HDRI}",
                root / HDRI,
                HDRI_SHA256,
            )
    binary, hdri = paths(project)
    version = subprocess.run(
        [str(binary), "--version"], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()[0]
    return {
        "renderer": "blender",
        "version": version,
        "binary_sha256": _digest(binary),
        "hdri_sha256": _digest(hdri),
    }
