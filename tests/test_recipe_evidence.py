"""Historical recipes retain their pins, provenance, and limited validation scope."""

import hashlib
import json

import pytest
from helpers import ROOT, load_tool
from roboenv.provisioning.setup import _requirements


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_candidate_preserves_numerics_and_uses_one_opencv(backend):
    locks = ROOT / "runtime/locks"
    old = _requirements(locks / f"{backend}.txt")
    candidate = _requirements(locks / f"{backend}.candidate.txt")
    for name in ("numpy", "mujoco", "jax", "jaxlib", "torch", "pyopengl", "scipy"):
        if name in old:
            assert candidate[name].specifier == old[name].specifier
    assert {name for name in candidate if name.startswith("opencv-")} == {"opencv-python"}
    assert str(candidate["opencv-python"].specifier) == "==4.11.0.86"
    overrides = _requirements(locks / f"{backend}.candidate-source-pins.txt")
    assert set(overrides) == {"jaxls"}
    assert overrides["jaxls"].url == candidate["jaxls"].url
    assert not overrides["jaxls"].specifier


@pytest.mark.parametrize("backend", ["robotwin", "behavior"])
def test_candidate_receipt_matches_files_and_does_not_bypass_resolution(backend):
    locks = ROOT / "runtime/locks"
    report = json.loads((locks / f"{backend}.candidate.json").read_text())
    for suffix, field in (
        (".txt", "candidate_lock_sha256"),
        ("-source-pins.txt", "source_pins_sha256"),
    ):
        path = locks / f"{backend}.candidate{suffix}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report[field]
    assert not report["verification"]["dependency_bypass_used"]
    if backend == "behavior":
        assert not report["recipe"]["installable"]
        assert not report["verification"]["clean_install_verified"]
        assert not report["verification"]["gpu_replay_verified"]
        assert not report["metadata_audit"]["passed"]
        assert all(core["exit_code"] != 0 for core in report["verification"]["resolver_cores"])
    else:
        assert not report["replay"]["containerized"]
        assert report["replay"]["reference_origin"] == "project-reference"
        assert (
            hashlib.sha256((ROOT / report["source_lock"]).read_bytes()).hexdigest()
            == report["source_lock_sha256"]
        )


def test_libero_evidence_preserves_history_and_does_not_claim_broad_replay():
    report = json.loads((ROOT / "docs/validation/libero-candidate.json").read_text())
    assert hashlib.sha256((ROOT / "runtime/locks/libero.txt").read_bytes()).hexdigest() == (
        "828bf65a80a99906643ce9375be6a392d379454cb96031ca5df1ed627e039b91"
    )
    assert report["historical_freeze_unchanged"] and report["pip_check"]["passed"]
    assert not report["dependency_resolution_bypassed"]
    assert not report["docker_harbor_verified"] and not report["all_tasks_or_seeds_verified"]
    replay = report["replay"]["phases"][-1]
    assert replay["reference_replay_passed"]
    assert replay["verdict"]["reward"] == 1.0
    assert replay["verdict"]["budget"] == replay["post_submit_budget"]


def test_candidate_probe_is_isolated_and_never_forwards_credentials(tmp_path, monkeypatch):
    for name in ("EXAMPLE_API_KEY", "HF_TOKEN", "PYTHONPATH"):
        monkeypatch.setenv(name, "not-forwarded")
    monkeypatch.setenv("ROBOENV_DETERMINISTIC_IK", "0")
    result = load_tool("replay_libero_candidate").environment(ROOT, tmp_path, "device-probe", 3)
    assert not {"EXAMPLE_API_KEY", "HF_TOKEN", "PYTHONPATH"} & result.keys()
    assert result["ROBOENV_DETERMINISTIC_IK"] == "1"
    assert result["MUJOCO_EGL_DEVICE_ID"] == result["CUDA_VISIBLE_DEVICES"] == "3"
    for name in ("LIBERO_CONFIG_PATH", "ROBOT_DESCRIPTIONS_CACHE"):
        from pathlib import Path

        assert Path(result[name]).is_relative_to(tmp_path)
