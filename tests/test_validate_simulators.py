"""Simulator validation orchestration on the fixture world: no GPU, image or model calls."""

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml
from helpers import load_tool

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"
WORKTREE_PATHS = [ROOT / "src", ROOT / "runtime/src"]


def fixture_source(tmp_path, *, program=None):
    """The toy task under a fresh public ID, with the fixture's reference program."""
    source = tmp_path / "source/validation-toy"
    source.mkdir(parents=True)
    task = yaml.safe_load((FIXTURE / "task.yaml").read_text())
    task["id"] = "fakeshell/validation-toy"
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    (source / "solution.py").write_text(program or (FIXTURE / "solution.py").read_text())
    return source.parent


def toy_options(tmp_path, **extra):
    return {
        "source": fixture_source(tmp_path, **extra),
        "catalog": FIXTURE / "catalog",
        "backends": ["fakeshell"],
    }


def test_probe_and_reference_run_through_launch_replay_and_verify(tmp_path):
    tool = load_tool("validate_simulators")
    out = tmp_path / "validation"
    options = toy_options(tmp_path)
    receipt = tool.prepare(out, modes=["probe", "reference"], seeds=[0], **options)
    assert receipt["phase"] == "prepared" and receipt["model_calls"] == 0
    assert [case["mode"] for case in receipt["cases"]] == ["reference", "probe"]
    assert not receipt["all_selected_checks_passed"]
    for case in receipt["cases"]:
        assert not case["execution_attempted"]
        assert case["expected_devices"] == ["controller"]
        assert (out / case["path"] / "program.py").is_file()
    result = tool.run_validation(
        out, catalog=options["catalog"], interpreter=sys.executable, python_path=WORKTREE_PATHS
    )
    assert result["phase"] == "completed"
    assert result["all_selected_checks_passed"] is True, result
    assert result["fresh_process_replay_certificate"] is False
    reference, probe = result["cases"]
    for case in (reference, probe):
        assert case["execution_attempted"] and case["world_stopped"] is True
        assert case["boot_verified"] and case["program_completed"]
        assert case["replay_exit_code"] == 0 and case["verify_exit_code"] == 0
        assert case["verifier_valid"] is True and case["failure_category"] is None
        assert set(case["timing_s"]) == {"launch_s", "replay_s", "verify_s"}
        verdict = out / case["path"] / "verdict"
        raw = (verdict / "actions.jsonl").read_bytes()
        assert case["journal"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert case["journal"]["validated"] is True and case["journal"]["requests"] >= 3
        assert case["budget"]["unit"] and type(case["budget"]["used"]) is int
        run = next((out / case["path"] / "runs").glob("*/run.json"))
        assert json.loads(run.read_text())["state"] == "stopped"
    assert reference["reference_replay_passed"] and reference["final_verifier_success"]
    assert reference["reward"] == 1 and reference["budget"]["used"] == 50
    assert probe["device_read_probe_passed"] and not probe["reference_replay_passed"]
    assert probe["reward"] == 0 and probe["budget"]["used"] == 0
    assert not probe["final_verifier_success"]
    private = out / "validation.private.log"
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert "Simulator logs" in private.read_text()
    public = (out / "receipt.json").read_text()
    assert str(tmp_path) not in public and str(ROOT) not in public
    assert json.loads(public) == result
    with pytest.raises(ValueError, match="must be new"):
        tool.prepare(out, **options)


def test_failed_reference_program_is_not_certified_and_the_world_is_stopped(tmp_path):
    tool = load_tool("validate_simulators")
    out = tmp_path / "validation"
    options = toy_options(tmp_path, program="raise RuntimeError('fixture controller failure')\n")
    tool.prepare(out, modes=["reference"], **options)
    result = tool.run_validation(
        out, catalog=options["catalog"], interpreter=sys.executable, python_path=WORKTREE_PATHS
    )
    (case,) = result["cases"]
    assert not result["all_selected_checks_passed"]
    assert case["replay_exit_code"] == 1 and not case["program_completed"]
    assert case["failure_category"] == "reference-program-failed"
    # The verifier still certified the untouched world, and it was stopped afterwards.
    assert case["verify_exit_code"] == 0 and case["verifier_valid"] is True
    assert case["reward"] == 0 and not case["final_verifier_success"]
    assert case["world_stopped"] is True
    assert "fixture controller failure" in (out / "validation.private.log").read_text()
    assert "fixture controller failure" not in (out / "receipt.json").read_text()


def test_modified_program_or_launch_failure_is_recorded_without_evidence(tmp_path, monkeypatch):
    tool = load_tool("validate_simulators")
    out = tmp_path / "validation"
    options = toy_options(tmp_path)
    receipt = tool.prepare(out, modes=["probe"], **options)
    program = out / receipt["cases"][0]["path"] / "program.py"
    original = program.read_text()
    program.write_text("# changed after preparation\n")
    with pytest.raises(ValueError, match="changed"):
        tool.run_validation(out, catalog=options["catalog"], interpreter=sys.executable)
    program.write_text(original)
    monkeypatch.setattr(
        "roboenv.environment.launch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boot"))
    )
    result = tool.run_validation(out, catalog=options["catalog"], interpreter=sys.executable)
    (case,) = result["cases"]
    assert case["execution_attempted"] and case["failure_category"] == "launch-failed"
    assert not case["boot_verified"] and case["replay_exit_code"] is None
    assert not result["all_selected_checks_passed"]
    with pytest.raises(ValueError, match="fresh prepared"):
        tool.run_validation(out, catalog=options["catalog"], interpreter=sys.executable)


def journal_lines(reads, *, failed=(), extra=()):
    def event(phase, op, args, response):
        return {"phase": phase, "request": {"op": op, "args": args}, "response": response}

    events = [
        event("commit", "devices.call", {"device": name, "action": "read"},
              {"ok": True, "result": {"read_only": True}})
        for name in reads
    ]
    events += [
        event("error", "devices.call", {"device": name, "action": "read"},
              {"ok": False, "error": {"code": "device_failed"}})
        for name in failed
    ]
    events += list(extra)
    events.append(event("commit", "submit", {}, {"ok": True, "result": {"completed": False}}))
    return "".join(json.dumps(item) + "\n" for item in events)


def verdict(directory, raw, *, reward=0.0, valid=True, task="fakeshell/case", seed=0):
    directory.mkdir(parents=True)
    (directory / "actions.jsonl").write_text(raw)
    (directory / "reward.json").write_text(json.dumps({"reward": reward}))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return {
        "task": task, "seed": seed, "valid": valid, "trial_success": reward == 1.0,
        "health": {"worker": {"state": "ready"}},
        "submit": {"budget": {"used": 0, "max": 1000, "unit": "steps"}},
        "journal": {"sha256": digest, "requests": raw.count("\n")},
        "journal_artifact": {"sha256": digest, "bytes": len(raw), "validated": valid},
    }


@pytest.mark.parametrize(
    "reads,failed,extra,category",
    [
        (["arm", "world"], [], [], None),
        (["arm"], ["world"], [], "device-read-failed"),
        (["arm"], [], [], "incomplete-device-read-evidence"),
        (
            ["arm", "world"], [],
            [{"phase": "commit", "request": {"op": "devices.call",
                                            "args": {"device": "arm", "action": "write"}},
              "response": {"ok": True, "result": {"read_only": False}}}],
            "probe-mutated-environment",
        ),
    ],
)
def test_probe_evidence_requires_every_declared_read_and_no_mutation(
    tmp_path, reads, failed, extra, category
):
    tool = load_tool("validate_simulators")
    record = {
        **tool._pending({"id": "fakeshell/case", "mode": "probe", "seed": 0,
                         "expected_devices": ["arm", "world"], "expected_reads": ["arm", "world"]}),
        "program_completed": True,
    }
    report = verdict(tmp_path / "verdict", journal_lines(reads, failed=failed, extra=extra))
    evidence = tool._evidence(record, report, tmp_path / "verdict")
    assert evidence["device_read_probe_passed"] is (category is None)
    assert evidence["failure_category"] == category
    assert evidence["failed_device_reads"] == list(failed)
    assert evidence["verifier_valid"] and evidence["boot_verified"]


def test_reference_evidence_needs_full_success_and_a_matching_journal(tmp_path):
    tool = load_tool("validate_simulators")
    record = {
        **tool._pending({"id": "fakeshell/case", "mode": "reference", "seed": 0,
                         "expected_devices": ["arm"], "expected_reads": ["arm"]}),
        "program_completed": True,
    }
    report = verdict(tmp_path / "success", journal_lines(["arm"]), reward=1.0)
    evidence = tool._evidence(record, report, tmp_path / "success")
    assert evidence["reference_replay_passed"] and evidence["final_verifier_success"]
    assert evidence["reward"] == 1.0 and evidence["failure_category"] is None
    report = verdict(tmp_path / "unsolved", journal_lines(["arm"]), reward=0.0)
    evidence = tool._evidence(record, report, tmp_path / "unsolved")
    assert not evidence["reference_replay_passed"]
    assert evidence["failure_category"] == "reference-not-successful"
    report = verdict(tmp_path / "tampered", journal_lines(["arm"]), reward=1.0)
    (tmp_path / "tampered/actions.jsonl").write_text("changed\n")
    evidence = tool._evidence(record, report, tmp_path / "tampered")
    assert not evidence["reference_replay_passed"] and "journal" not in evidence
    assert evidence["failure_category"] == "invalid-verifier-evidence"
    report = verdict(tmp_path / "mismatch", journal_lines(["arm"]), reward=1.0, seed=3)
    evidence = tool._evidence(record, report, tmp_path / "mismatch")
    assert not evidence["boot_verified"] and not evidence["final_verifier_success"]


def test_retired_catalog_references_are_skipped_by_default_and_refused_when_selected():
    tool = load_tool("validate_simulators")
    with pytest.raises(ValueError, match="no validation cases matched"):
        tool.select_cases(backends=["robocasa"], modes=["reference"])
    with pytest.raises(ValueError, match="retired"):
        tool.select_cases(
            backends=["robocasa"], cases=["robocasa/arrange-vegetables"], modes=["reference"]
        )


def test_reference_selection_requires_a_solution():
    tool = load_tool("validate_simulators")
    with pytest.raises(ValueError, match="requires an explicit regular solution.py"):
        tool.select_cases(backends=["robotwin"], cases=["robotwin/handover-mic"])


def test_probe_plans_use_scene_robots_and_explicit_behavior_config(tmp_path):
    tool = load_tool("validate_simulators")
    source = tmp_path / "source/behavior-probe"
    source.mkdir(parents=True)
    task = {
        "id": "explicit-behavior-case",
        "execution_mode": "stepped",
        "backend": "behavior",
        "backend_task": "behavior.explicit",
        "robot": "behavior/behavior-r1pro",
        "instruction": "Fixture instruction from explicit source",
        "backend_kwargs": {"activity_name": "fixture", "scene_model": "explicit_scene"},
    }
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    cases = tool.select_cases(
        source=source.parent, backends=[*tool.DEFAULT_BACKENDS, "behavior"], modes=["probe"]
    )
    assert len(cases) == 7
    assert {task.robot for _, task, _, _ in cases if task.backend == "libero"} == {
        "libero/libero-panda-tabletop",
        "libero/libero-panda-kitchen",
    }
    aloha = next(task for _, task, _, _ in cases if task.robot == "robotwin/robotwin-aloha-agilex")
    assert aloha.backend_kwargs["embodiment"] == "aloha_agilex"
    behavior = next(task for _, task, _, _ in cases if task.backend == "behavior")
    assert behavior.backend_kwargs["scene_model"] == "explicit_scene"
    for _, _, _, program in cases:
        assert "env.device(name).read()" in program
        assert not any(value in program for value in ("write", "submit", "reset", "shutdown"))
    prepared = tool.prepare(
        tmp_path / "probes", source=source.parent,
        backends=[*tool.DEFAULT_BACKENDS, "behavior"], modes=["probe"], seeds=[0, 1],
    )
    assert len(prepared["cases"]) == 14 and {c["seed"] for c in prepared["cases"]} == {0, 1}
    assert str(tmp_path) not in (tmp_path / "probes/receipt.json").read_text()
    task["backend_kwargs"].pop("scene_model")
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match="explicit scene_model"):
        tool.select_cases(source=source.parent, backends=["behavior"], modes=["probe"])


def test_case_selection_alone_narrows_probe_backends_without_guessing():
    tool = load_tool("validate_simulators")
    cases = tool.select_cases(
        cases=["robotwin/handover-mic"], robots=["robotwin/robotwin-dual-piper"], modes=["probe"]
    )
    assert len(cases) == 1 and cases[0][1].backend == "robotwin"
    with pytest.raises(ValueError, match="explicit task source"):
        tool.select_cases(backends=["fakeshell"], modes=["probe"])


def test_gpu_workflow_requires_trusted_manual_context_and_privacy_gate():
    workflow = yaml.load(
        (ROOT / ".github/workflows/gpu-validation.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    job = workflow["jobs"]["validate"]
    assert "default_branch" in job["if"]
    assert job["environment"] == "gpu-validation"
    assert "self-hosted" in job["runs-on"]
    validation = next(step for step in job["steps"] if step.get("id") == "validation")
    assert "tools/validate_simulators.py" in validation["run"] and "--run" in validation["run"]
    assert "--deployments" not in validation["run"]
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "steps.privacy.outcome == 'success'" in upload["if"]
    assert upload["with"]["path"] == "build/gpu-validation/receipt.json"
