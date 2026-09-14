#!/usr/bin/env python3
"""Acquire the catalog's checksum-pinned G1 scene bundle without starting Kit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))
from roboenv_runtime.adapters.isaaclab.g1_assets import digest, load_bundle  # noqa: E402

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"


def prepare(manifest, root):
    root = root.resolve()
    spec = json.loads(manifest.read_text())
    for name, checksum in spec["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Asset path escapes destination")
        if path.is_file() and digest(path) == checksum:
            continue
        if path.exists():
            raise ValueError(f"Existing asset differs from the pin: {name}")
        print(f"Downloading {name}", flush=True)
        with urlopen(BASE + quote(name, safe="/"), timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Downloaded asset differs from the pin: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "assets.json").write_bytes(manifest.read_bytes())
    load_bundle(root / "assets.json")
    print(f"Verified G1 bundle: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "src/roboenv/assets/isaaclab-g1.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / ".cache/isaaclab/g1")
    options = parser.parse_args()
    prepare(options.manifest, options.out)
