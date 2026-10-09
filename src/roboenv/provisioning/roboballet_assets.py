"""Explicit preparation of the checksum-pinned RoboBallet source and geometry."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.robocasa_assets import _target

RECORD = "16546403"
URL = f"https://zenodo.org/api/records/{RECORD}/files/roboballet.tar.zst/content"
SHA256 = "ffb8447e6d0f309026d9baa8538d0c77931655323cbb532e0bf8c11a617ad537"
MODULES = {"mujoco_simulator.py", "simulator.py", "ik_solver.py", "geometry.py"}


def asset_root(project):
    cache = Path(os.environ.get("ROBOENV_ASSET_CACHE", Path(project) / ".cache/roboenv/assets"))
    return cache / "roboballet" / RECORD


def verify_assets(selected):
    """Verify an existing prepared depot without downloading or changing it."""
    selected = Path(selected)
    manifest = json.loads((selected.parent / "selected-files.json").read_text())
    required = {f"roboballet/environments/{name}" for name in MODULES}
    required.add("roboballet/data/mujoco_world/8_pandas_world.xml")
    if not required <= manifest.keys():
        raise ValueError("Incomplete RoboBallet asset manifest; run tools/prepare_roboballet.py")
    for name, expected in manifest.items():
        if _digest(_target(selected, name)) != expected:
            raise ValueError(f"RoboBallet asset differs from its prepared manifest: {name}")
    return {"files": len(manifest), "sha256": hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()}


def prepare_assets(project):
    root = asset_root(project)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "roboballet.tar.zst"
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=root, prefix="download-") as directory:
            pending = Path(directory) / "archive"
            with urlopen(URL, timeout=120) as response, pending.open("xb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            if _digest(pending) != SHA256:
                raise ValueError("RoboBallet download differs from the author's pinned release")
            pending.replace(archive)
    if _digest(archive) != SHA256:
        raise ValueError("Existing RoboBallet archive differs from its pin; preserve and inspect")
    selected = root / "selected"
    selected.mkdir(exist_ok=True)
    manifest = {}
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as content:
            for member in content:
                parts = Path(member.name).parts
                keep = (member.name in {"roboballet/LICENSE", "roboballet/README.md"}
                        or parts[:3] in {("roboballet", "data", "mujoco_world"),
                                         ("roboballet", "data", "paper_test_tasksets")}
                        or member.name == "roboballet/data/data_locator.py"
                        or parts[:2] == ("roboballet", "environments") and parts[-1] in MODULES)
                if not keep or not member.isfile():
                    continue
                destination = _target(selected, member.name)
                with content.extractfile(member) as source:
                    data = source.read()
                checksum = hashlib.sha256(data).hexdigest()
                if destination.exists():
                    if _digest(destination) != checksum:
                        raise ValueError(f"Existing RoboBallet asset differs: {member.name}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as output:
                        output.write(data)
                manifest[member.name] = checksum
        if process.wait():
            raise RuntimeError("Could not decompress verified RoboBallet assets")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    receipt = root / "selected-files.json"
    if receipt.exists() and json.loads(receipt.read_text()) != manifest:
        raise ValueError("Existing RoboBallet asset manifest differs; preserve and inspect it")
    if not receipt.exists():
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"roboballet_assets_verified": True, "archive_sha256": SHA256,
            "asset_directory": str(selected.resolve()), **verify_assets(selected)}
