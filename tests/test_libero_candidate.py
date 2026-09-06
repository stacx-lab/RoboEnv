"""Preserve the LIBERO numerical stack while resolving its actual runtime closure."""

import hashlib
import importlib.util
import json
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_libero_candidate", ROOT / "tools/replay_libero_candidate.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def pins(path):
    return {
        canonicalize_name(requirement.name): requirement
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
        for requirement in [Requirement(line)]
    }


def test_historical_freeze_is_unchanged_and_numerics_are_preserved():
    historical = ROOT / "runtime/locks/libero.txt"
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == (
        "828bf65a80a99906643ce9375be6a392d379454cb96031ca5df1ed627e039b91"
    )
    old, candidate = pins(historical), pins(ROOT / "runtime/locks/libero.candidate.txt")
    for name in ("numpy", "mujoco", "jax", "jaxlib", "torch", "pyopengl", "scipy"):
        assert str(candidate[name].specifier) == str(old[name].specifier)
    assert "pyrender" not in candidate
    assert "future" in candidate
    assert {name for name in candidate if name.startswith("opencv-")} == {"opencv-python"}
    assert str(candidate["opencv-python"].specifier) == "==4.11.0.86"


def test_vcs_override_only_pins_same_source_identity():
    overrides = pins(ROOT / "runtime/locks/libero.candidate-source-pins.txt")
    assert set(overrides) == {"jaxls"}
    assert overrides["jaxls"].url == (
        "git+https://github.com/brentyi/jaxls.git@6fe7cf9d56223736b2c0272d080ec7346203c648"
    )
    assert not overrides["jaxls"].specifier


def test_verified_opencv_metadata_accepts_numpy_1_26_on_python_3_12():
    # Requires-Dist values observed in the official PyPI version metadata.
    dependencies = [
        'numpy>=1.21.2; python_version >= "3.10"',
        'numpy>=1.23.5; python_version >= "3.11"',
        'numpy>=1.26.0; python_version >= "3.12"',
    ]
    environment = {**default_environment(), "python_version": "3.12"}
    assert all(
        Version("1.26.4") in requirement.specifier
        for text in dependencies
        for requirement in [Requirement(text)]
        if requirement.marker.evaluate(environment)
    )
    assert Version("1.26.4") not in Requirement("numpy>=2,<2.3").specifier


def test_candidate_probe_uses_new_config_and_never_forwards_model_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXAMPLE_API_KEY", "not-forwarded")
    monkeypatch.setenv("HF_TOKEN", "not-forwarded")
    monkeypatch.setenv("PYTHONPATH", "not-used")
    monkeypatch.setenv("ROBOENV_DETERMINISTIC_IK", "0")
    result = PROBE.environment(ROOT, tmp_path, "device-probe", 3)
    assert "EXAMPLE_API_KEY" not in result and "HF_TOKEN" not in result
    assert "PYTHONPATH" not in result
    assert result["ROBOENV_DETERMINISTIC_IK"] == "1"
    assert result["MUJOCO_EGL_DEVICE_ID"] == result["CUDA_VISIBLE_DEVICES"] == "3"
    assert Path(result["LIBERO_CONFIG_PATH"]).is_relative_to(tmp_path)
    assert Path(result["ROBOT_DESCRIPTIONS_CACHE"]).is_relative_to(tmp_path)


def test_acceptance_evidence_does_not_claim_docker_or_all_seeds():
    report = json.loads((ROOT / "docs/validation/libero-candidate.json").read_text())
    assert report["historical_freeze_unchanged"]
    assert report["pip_check"]["passed"]
    assert not report["dependency_resolution_bypassed"]
    assert not report["docker_harbor_verified"]
    assert not report["all_tasks_or_seeds_verified"]
    replay = report["replay"]["phases"][-1]
    assert replay["reference_replay_passed"]
    assert replay["verdict"]["reward"] == 1.0
    assert replay["verdict"]["budget"] == replay["post_submit_budget"]
