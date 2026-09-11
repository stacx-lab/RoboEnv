"""Acquire pinned BEHAVIOR bundles only after explicit third-party acceptance."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from zipfile import ZipFile

from roboenv.provisioning.assets import _digest
from roboenv.provisioning.progress import stage
from roboenv.provisioning.robocasa_assets import _target

REVISION = "9f0d57d465726976ed98138d3f8b8ca3e2186775"
ARCHIVES = {
    "omnigibson-robot-assets": {
        "file": "omnigibson-robot-assets-3.8.2.zip",
        "bytes": 641004353,
        "sha256": "3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126",
        "version": "3.8.2",
    },
    "behavior-1k-assets": {
        "file": "behavior-1k-assets-3.9.0.zip",
        "bytes": 31457673073,
        "sha256": "09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e",
        "version": "3.9.0",
    },
    "2026-challenge-task-instances": {
        "file": "2026-challenge-task-instances.zip",
        "bytes": 108543852,
        "sha256": "d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81",
    },
}


def acceptance_missing():
    return [
        name
        for variable, name in (
            ("OMNI_KIT_ACCEPT_EULA", "Isaac Sim EULA"),
            ("ROBOENV_BEHAVIOR_ACCEPT_LICENSE", "BEHAVIOR Data Bundle agreement"),
        )
        if os.environ.get(variable, "").upper() != "YES"
    ]


def install_archive(archive, destination):
    """Extract encrypted native data as shipped; never decrypt outside OmniGibson."""
    destination = Path(destination)
    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [str(Path(entry.filename)) for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError("Dataset archive has duplicate entries")
        files = []
        for entry in entries:
            _target(destination, entry.filename)
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Dataset archive contains a symlink")
            if not entry.is_dir():
                files.append(entry)
        if not files:
            raise ValueError("Dataset archive is empty")
        if destination.is_symlink() and not destination.is_dir():
            raise ValueError("Preserve the existing dangling dataset link")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                shutil.disk_usage(destination.parent).free
                < sum(x.file_size for x in files) + 2 * 1024**3
            ):
                raise ValueError("Insufficient dataset space; choose a larger ROBOENV_ASSET_CACHE")
            with tempfile.TemporaryDirectory(
                prefix=".behavior-extract-", dir=destination.parent
            ) as tmp:
                pending = Path(tmp) / "data"
                pending.mkdir()
                bundle.extractall(pending)
                pending.rename(destination)
        hashes = {}
        for entry in files:
            path = _target(destination, entry.filename)
            expected = hashlib.sha256()
            with bundle.open(entry) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    expected.update(block)
            if not path.is_file() or path.is_symlink() or _digest(path) != expected.hexdigest():
                raise ValueError(
                    "Existing BEHAVIOR data differs from the pinned bundle; preserve and inspect it"
                )
            hashes[entry.filename] = expected.hexdigest()
        return hashes


def acquire(project, cache):
    missing = acceptance_missing()
    if missing:
        raise ValueError(
            "Explicit acceptance is required before acquisition: " + ", ".join(missing)
        )
    # Available in the selected simulator recipe; the lightweight SDK needs no HF dependency.
    from huggingface_hub import hf_hub_download

    project, cache = Path(project).resolve(), Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    depot = Path(os.environ.get("OMNIGIBSON_DATA_PATH", cache / "datasets")).resolve()
    depot.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, expected in ARCHIVES.items():
        print("Acquiring/verifying pinned BEHAVIOR bundle: " + name, flush=True)
        archive = Path(
            hf_hub_download(
                repo_id="behavior-1k/zipped-datasets",
                repo_type="dataset",
                revision=REVISION,
                filename=expected["file"],
                local_dir=cache / "archives",
            )
        )
        if archive.stat().st_size != expected["bytes"] or _digest(archive) != expected["sha256"]:
            raise ValueError("BEHAVIOR bundle differs from its pinned source checksum")
        hashes = install_archive(archive, depot / name)
        if (
            "version" in expected
            and (depot / name / "VERSION").read_text().strip() != expected["version"]
        ):
            raise ValueError("Extracted BEHAVIOR asset version differs from the recipe")
        receipt = cache / (name + ".json")
        receipt.write_text(
            json.dumps(
                {
                    "schema": "roboenv/behavior-assets/1",
                    "revision": REVISION,
                    "archive": expected,
                    "files": hashes,
                },
                indent=2,
            )
            + "\n"
        )
        receipts.append(str(receipt))
    link = project / "third_party/BEHAVIOR-1K/OmniGibson/datasets"
    if link.exists() or link.is_symlink():
        if link.resolve() != depot:
            raise ValueError(
                "Preserve existing OmniGibson datasets; configure that depot explicitly"
            )
    else:
        link.symlink_to(depot, target_is_directory=True)
    # Use the upstream key installation within OmniGibson after explicit acceptance.
    # No key bytes or decryption output enter receipts or the agent workspace.
    os.environ["OMNIGIBSON_DATA_PATH"] = str(depot)
    from omnigibson.utils.asset_utils import download_key, get_key_path

    key = Path(get_key_path())
    existed = key.exists()
    download_key()
    if not key.is_file():
        raise ValueError("Upstream BEHAVIOR key installation did not produce a key file")
    if not existed:
        key.chmod(0o600)
    return {
        "environment_assets_verified": True,
        "asset_depot": str(depot),
        "asset_receipts": receipts,
        "key_installed": True,
        "license_acceptance": "explicit environment settings; no terms accepted by this tool",
        "gpu_replay_verified": False,
    }


def prepare_assets(project, *, progress=None):
    missing = acceptance_missing()
    if missing:
        return {"environment_assets_verified": False, "license_acceptance_required": missing}
    project = Path(project).resolve()
    cache = (
        Path(os.environ.get("ROBOENV_ASSET_CACHE", project / ".cache/roboenv/assets")) / "behavior"
    )
    registration = json.loads((project / ".cache/roboenv/installations/behavior.json").read_text())
    output = (
        project / ".cache/roboenv/installations" / ("behavior-assets-" + uuid.uuid4().hex + ".json")
    )
    with stage(progress, "Preparing licensed BEHAVIOR assets through the selected interpreter"):
        subprocess.run(
            [
                registration["python"],
                "-m",
                "roboenv.provisioning.behavior_assets",
                "--project",
                str(project),
                "--cache",
                str(cache),
                "--out",
                str(output),
            ],
            cwd=project,
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=14400,
        )
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve existing receipts; select a new output")
    report = acquire(args.project, args.cache)
    with args.out.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
