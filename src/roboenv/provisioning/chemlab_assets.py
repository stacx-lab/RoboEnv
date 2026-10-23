"""Pinned ChemLab release acquisition; discovery does not import instrument code."""

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .assets import _digest

REPOSITORY = "MBZUAI/ChemLab-Assets"
REVISION = "a6c006f921c9bee62f5585dd7f769163a7f51573"


def asset_root(directory=None):
    if directory is not None:
        return Path(directory).resolve()
    cache = Path(os.environ.get("ROBOENV_ASSET_CACHE", Path.home() / ".cache/roboenv/assets"))
    return cache / "chemlab" / REVISION


def resource(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe ChemLab resource: {relative}")
    return path


def catalog(directory=None):
    root = asset_root(directory)
    with resource(root, "catalog/catalog.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = {row["id"]: row for row in rows}
    if len(result) != 203 or len(result) != len(rows):
        raise ValueError("Pinned ChemLab catalog must contain 203 unique instruments")
    return result


def prepare(directory=None):
    root = asset_root(directory)
    if root.exists():
        return verify_assets(root, write_receipt=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem. Never replace an existing depot.
    with tempfile.TemporaryDirectory(prefix=".chemlab-", dir=root.parent) as temporary:
        staging = Path(temporary) / "release"
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                f"https://huggingface.co/datasets/{REPOSITORY}",
                str(staging),
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "checkout", "--detach", REVISION], env=env, check=True
        )
        subprocess.run(["git", "-C", str(staging), "lfs", "pull"], check=True)
        verify_assets(staging, write_receipt=True)
        staging.rename(root)
    return verify_assets(root, write_receipt=True)


def verify_assets(directory=None, *, write_receipt=False):
    """Check released resources, retaining a content manifest outside the source tree."""
    root = asset_root(directory)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if not (root / ".git").is_dir() or git("rev-parse", "HEAD") != REVISION:
        raise ValueError("ChemLab requires the pinned Git/LFS depot; run prepare_chemlab.py")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("ChemLab tracked resources have changed; preserve and inspect the depot")
    # Git verifies source/metadata; LFS verifies all binary resources, including
    # accessory geometry, textures, Blender files and example assets.
    git("lfs", "fsck")
    rows = catalog(root)
    records = {}
    for name, row in rows.items():
        contract = json.loads(resource(root, row["contract"]).read_text())
        if contract["asset_type_id"] != name:
            raise ValueError(f"Contract identity mismatch: {name}")
        records[name] = {
            "contract_sha256": _digest(resource(root, row["contract"])),
            "urdf_sha256": _digest(resource(root, row["urdf"])),
            "glb_sha256": _digest(resource(root, row["glb"])),
            "capabilities": len(contract["capabilities"]),
            "unsupported": sum(
                c.get("availability") == "unsupported" for c in contract["capabilities"].values()
            ),
        }
    receipt = {"repository": REPOSITORY, "revision": REVISION, "instruments": records}
    target = root / "roboenv-assets.json"
    if target.exists() and json.loads(target.read_text()) != receipt:
        raise ValueError("ChemLab resources differ from their preparation receipt")
    if write_receipt and not target.exists():
        target.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
