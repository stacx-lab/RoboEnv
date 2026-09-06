"""Basic workflow policy checks; hosted execution is a separate verification layer."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflows():
    return {
        path.name: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for path in (ROOT / ".github/workflows").glob("*.yml")
    }


def test_ci_uses_immutable_top_level_action_pins_and_read_only_tokens():
    for workflow in workflows().values():
        assert workflow["permissions"] == {"contents": "read"}
        assert "pull_request_target" not in workflow["on"]
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", step["uses"])
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] == "false"


def test_candidate_upload_cannot_bypass_failed_privacy_audit():
    candidate = workflows()["release-candidate.yml"]
    assert set(candidate["on"]) == {"workflow_dispatch"}
    steps = candidate["jobs"]["candidate"]["steps"]
    privacy = next(step for step in steps if step.get("id") == "artifact-privacy")
    assert privacy["if"] == "always()"
    upload = next(
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "steps.artifact-privacy.outcome == 'success'" in upload["if"]
    assert upload["with"]["path"] == "build/release/"
    assert all("pypa/gh-action-pypi-publish" not in step.get("uses", "") for step in steps)


def test_ci_compatibility_matches_documented_supported_python_range():
    job = workflows()["check.yml"]["jobs"]["compatibility"]
    assert job["strategy"]["matrix"]["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert job["runs-on"] == "ubuntu-24.04"


def test_feature_branches_use_pr_checks_without_duplicate_push_runs():
    events = workflows()["check.yml"]["on"]
    assert set(events) == {"push", "pull_request", "workflow_dispatch"}
    assert events["push"]["branches"] == ["main"]


def test_dependabot_version_updates_are_paused():
    assert not (ROOT / ".github/dependabot.yml").exists()
    assert not (ROOT / ".github/dependabot.yaml").exists()
