"""Verify a locally prepared G1 bundle before starting Kit; never download assets."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bundle(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("schema") != "roboenv/g1-assets/1":
        raise ValueError("Unsupported G1 asset bundle")
    root = path.parent
    files = value["files"]
    for name, checksum in files.items():
        local = (root / name).resolve(strict=True)
        if not local.is_relative_to(root) or not local.is_file() or digest(local) != checksum:
            raise ValueError(f"G1 asset changed or escaped bundle: {name}")
    roles = value["roles"]
    if set(roles) != {"robot", "table", "object", "policy"} or any(
        name not in files for name in roles.values()
    ):
        raise ValueError("G1 requires verified robot, table, object and policy assets")
    return value, {role: str(root / name) for role, name in roles.items()}
