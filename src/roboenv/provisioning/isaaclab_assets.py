"""Acquire the two checksum-pinned Go2 USD files without starting Isaac Sim."""

import hashlib
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen


def manifest():
    return json.loads(files("roboenv").joinpath("assets/isaaclab-go2.json").read_text())


def asset_root(project):
    return Path(os.environ.get("ROBOENV_ISAACLAB_ASSETS", Path(project) / ".cache/isaaclab/assets"))


def verify_assets(project):
    root = asset_root(project)
    for name, digest in manifest()["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Go2 assets are missing or changed; run roboenv prepare isaaclab")
    return root


def prepare_assets(project, *, progress=None):
    spec, root = manifest(), asset_root(project)
    for name, digest in spec["files"].items():
        path = root / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        if path.exists():
            raise ValueError(
                "Existing Go2 asset differs from its pin; "
                "select a fresh ROBOENV_ISAACLAB_ASSETS directory"
            )
        if progress:
            progress(f"Downloading Go2 {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(spec["url"] + "/" + name, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Downloaded Go2 asset differs from its checksum")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        temporary.replace(path)
    verify_assets(project)
    return {"environment_assets_verified": True, "gpu_replay_verified": False}
