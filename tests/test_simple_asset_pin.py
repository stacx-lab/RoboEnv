"""SIMPLE's asset pin comes from the package; a packaged runtime root has no source tree."""

import hashlib
from pathlib import Path

import pytest
from roboenv.provisioning.simple_assets import asset_pin_bytes, verify_assets

ROOT = Path(__file__).resolve().parents[1]


def test_pin_is_the_packaged_file():
    assert asset_pin_bytes() == (ROOT / "src/roboenv/assets/simple-g1.json").read_bytes()


def test_verification_reads_the_pin_without_a_source_tree(tmp_path):
    # A runtime root with a data depot but no src/: the pin loads, the receipts are what is missing.
    (tmp_path / ".cache/simple/data").mkdir(parents=True)
    with pytest.raises(ValueError, match="SIMPLE assets missing"):
        verify_assets(tmp_path)


def test_pin_digest_matches_the_source_file():
    assert hashlib.sha256(asset_pin_bytes()).hexdigest() == hashlib.sha256(
        (ROOT / "src/roboenv/assets/simple-g1.json").read_bytes()
    ).hexdigest()
