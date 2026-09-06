"""A recorded BEHAVIOR freeze must never masquerade as a resolved installation."""

import hashlib
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "runtime/locks"


def test_behavior_candidate_hashes_source_boundaries_and_claims_are_consistent():
    report = json.loads((LOCKS / "behavior.candidate.json").read_text())
    for filename, field in (
        ("behavior.candidate.txt", "candidate_lock_sha256"),
        ("behavior.candidate-source-pins.txt", "source_pins_sha256"),
    ):
        assert hashlib.sha256((LOCKS / filename).read_bytes()).hexdigest() == report[field]
    requirements = [
        Requirement(line)
        for line in (LOCKS / "behavior.candidate.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    names = {canonicalize_name(requirement.name) for requirement in requirements}
    assert len(names) == len(requirements) == 240
    assert not {"bddl", "omnigibson"} & names
    assert {"lerobot", "nvidia-curobo", "isaacsim-kernel"} <= names
    assert report["recipe"]["installable"] is False
    assert report["verification"]["clean_install_verified"] is False
    assert report["verification"]["gpu_replay_verified"] is False
    assert report["verification"]["dependency_bypass_used"] is False
    assert report["verification"]["numeric_requirement_overrides_used"] is False
    assert report["metadata_audit"]["passed"] is False
    assert report["sources"]["behavior"]["prepared_revision_matches_locked_source"] is True
    assert len(report["metadata_audit"]["conflicts"]) == 8
    assert all(core["exit_code"] != 0 for core in report["verification"]["resolver_cores"])


def test_behavior_candidate_preserves_observed_pins_and_exposes_intrinsic_conflicts():
    requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in (
            Requirement(line)
            for line in (LOCKS / "behavior.candidate.txt").read_text().splitlines()
            if line and not line.startswith("#")
        )
    }
    for name, version in {
        "websockets": "17.1",
        "pillow": "11.0.0",
        "packaging": "25.0",
        "numpy": "1.26.0",
        "torch": "2.7.0+cu128",
    }.items():
        assert str(requirements[name].specifier) == "==" + version
    assert requirements["lerobot"].url.endswith("@436812bd8ee39b768c645c248c91f1330834e687")
    assert requirements["nvidia-curobo"].url.endswith("@78612f45cef52c3fa0298de243a54cd7ca614414")
    report = json.loads((LOCKS / "behavior.candidate.json").read_text())
    assert {core["name"] for core in report["irreducible_conflicts"]} == {
        "websockets",
        "Pillow",
        "packaging",
    }
    assert report["verification"]["full_online_compile_attempted"] is False
