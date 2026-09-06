"""The candidate is a narrow, explicit change from preserved experiment pins."""

import hashlib
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def requirements(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            item = Requirement(line)
            result[canonicalize_name(item.name)] = item
    return result


def test_candidate_preserves_core_pins_and_changes_only_recorded_dependencies():
    source = requirements(ROOT / "runtime/locks/robotwin.txt")
    candidate = requirements(ROOT / "runtime/locks/robotwin.candidate.txt")
    assert set(candidate) == set(source) - {"opencv-python-headless"} | {"opencv-python"}
    for name in set(source) - {"opencv-python-headless", "jaxls"}:
        assert str(candidate[name]) == str(source[name]), name
    assert str(candidate["opencv-python"].specifier) == "==4.11.0.86"
    assert candidate["jaxls"].url.endswith("@6fe7cf9d56223736b2c0272d080ec7346203c648")
    overrides = requirements(ROOT / "runtime/locks/robotwin.candidate-source-pins.txt")
    assert set(overrides) == {"jaxls"}
    assert str(overrides["jaxls"]) == str(candidate["jaxls"])


def test_candidate_receipt_does_not_overstate_the_installation_or_replay_scope():
    report = json.loads((ROOT / "runtime/locks/robotwin.candidate.json").read_text())
    assert (
        report["source_lock_sha256"]
        == hashlib.sha256((ROOT / report["source_lock"]).read_bytes()).hexdigest()
    )
    for recipe_key, digest_key in (
        ("lock", "candidate_lock_sha256"),
        ("source_pins", "source_pins_sha256"),
    ):
        assert (
            report[digest_key]
            == hashlib.sha256((ROOT / report["recipe"][recipe_key]).read_bytes()).hexdigest()
        )
    assert report["recipe"]["editables"] == []
    assert "third_party/cap-x" in report["recipe"]["source_only"]
    assert report["verification"]["dependency_bypass_used"] is False
    assert report["verification"]["opencv_distributions"] == ["opencv-python"]
    assert report["replay"]["containerized"] is False
    assert report["replay"]["reference_origin"] == "project-reference"
    assert report["replay"]["budget"]["used"] == 64
