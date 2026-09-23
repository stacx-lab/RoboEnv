"""GPU validation orchestration is tested without Docker, GPUs or model calls."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from helpers import load_tool

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"test-only wheel; never installed")
    mapping = {}
    for backend in ("libero", "robocasa", "robotwin", "robodojo", "behavior"):
        deployment = tmp_path / f"{backend}.json"
        deployment.write_text(
            json.dumps(
                {
                    "schema": "roboenv/deployment/1",
                    "backend": backend,
                    "image": "sha256:" + "a" * 64,
                    "legacy_tree_required": False,
                    "runtime_wheel_sha256": "b" * 64,
                    "upstream_lock_sha256": "c" * 64,
                    "source_revisions": {"source": "d" * 40},
                }
            )
        )
        mapping[backend] = deployment.name
    deployments = tmp_path / "map.json"
    deployments.write_text(json.dumps(mapping))
    return wheel, deployments


def typed_reference_fixture(tmp_path):
    source = tmp_path / "reference-source"
    source.mkdir()
    task = yaml.safe_load((ROOT / "catalog/tasks/robotwin/handover-mic/task.yaml").read_text())
    task["id"] = "robotwin/test-reference"
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    (source / "solution.py").write_text("# Fixture for result classification; never executed.\n")
    return source


def test_default_preparation_excludes_retired_primitive_references(tmp_path, monkeypatch):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    monkeypatch.setattr(
        tool.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not execute")
    )
    out = tmp_path / "validation"
    receipt = tool.prepare(deployments, wheel, out)
    assert len(receipt["cases"]) == 2
    assert {case["backend"] for case in receipt["cases"]} == {"robocasa"}
    assert receipt["phase"] == "prepared" and receipt["model_calls"] == 0
    assert receipt["reference_official"] is False
    assert receipt["harbor_loader_validated"] is True
    assert not receipt["all_selected_checks_passed"]
    assert str(tmp_path) not in (out / "receipt.json").read_text()
    with pytest.raises(ValueError, match="must be new"):
        tool.prepare(deployments, wheel, out)


def test_probe_plans_use_scene_robots_and_explicit_behavior_and_yam_config(tmp_path):
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
        "action_abstraction": "devices",
    }
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    cases = tool.select_cases(
        source=source.parent, backends=[*tool.DEFAULT_BACKENDS, "behavior"], modes=["probe"]
    )
    assert len(cases) == 8
    assert {task.robot for _, task, _, _ in cases if task.backend == "libero"} == {
        "libero/libero-panda-tabletop", "libero/libero-panda-kitchen",
    }
    yam = next(task for _, task, _, _ in cases if task.robot == "robodojo/robodojo-yam")
    assert yam.backend_kwargs["env_cfg_type"] == "yam"
    assert yam.backend_kwargs["layout_config"] == "arx_x5"
    aloha = next(task for _, task, _, _ in cases if task.robot == "robotwin/robotwin-aloha-agilex")
    assert aloha.backend_kwargs["embodiment"] == "aloha_agilex"
    behavior = next(task for _, task, _, _ in cases if task.backend == "behavior")
    assert behavior.backend_kwargs["scene_model"] == "explicit_scene"
    assert behavior.action_abstraction == "devices"
    for _, _, _, program in cases:
        assert "client.device(descriptor['name']).read()" in program
        assert not any(value in program for value in ("submit", "reset", "shutdown", "_private"))
    wheel, deployments = inputs(tmp_path)
    prepared = tool.prepare(
        deployments,
        wheel,
        tmp_path / "probes",
        source=source.parent,
        backends=[*tool.DEFAULT_BACKENDS, "behavior"],
        modes=["probe"],
    )
    assert len(prepared["cases"]) == 8 and prepared["harbor_loader_validated"]
    task["backend_kwargs"].pop("scene_model")
    (source / "task.yaml").write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match="explicit scene_model"):
        tool.select_cases(source=source.parent, backends=["behavior"], modes=["probe"])


def write_trial(out, record, *, reward, program_completed=True):
    trial = out / "taskset/jobs/gpu-validation/trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir()
    events = [
        {
            "phase": "commit",
            "request": {"op": "devices.call", "args": {"device": name, "action": "read"}},
            "response": {"ok": True, "result": {"read_only": True}},
        }
        for name in record["expected_reads"]
    ]
    raw = "".join(json.dumps(event) + "\n" for event in events)
    (trial / "verifier/actions.jsonl").write_text(raw)
    (trial / "verifier/submit.json").write_text(
        json.dumps(
            {
                "task": record["backend_task"],
                "seed": record["seed"],
                "valid": True,
                "health": {"worker": {"state": "ready"}},
                "benchmark_eligible": True,
                "trial_success": reward == 1,
                "journal": {"sha256": hashlib.sha256(raw.encode()).hexdigest()},
            }
        )
    )
    (trial / "verifier/reward.json").write_text(json.dumps({"reward": reward}))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": record["name"],
                "exception_info": None,
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )
    if program_completed:
        (trial / "agent/validation-program.json").write_text(
            json.dumps(
                {
                    "schema": "roboenv/validation-program/1",
                    "case_id": record["id"],
                    "seed": record["seed"],
                    "program_sha256": record["program_sha256"],
                }
            )
        )
    return trial


@pytest.mark.parametrize(
    "mode,reward,expected", [("reference", 0, False), ("reference", 1, True), ("probe", 0, True)]
)
def test_runner_classifies_real_success_separately_from_probe_connectivity(
    tmp_path, monkeypatch, mode, reward, expected
):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    out = tmp_path / "validation"
    receipt = tool.prepare(
        deployments,
        wheel,
        out,
        backends=["robotwin"],
        cases=["robotwin/test-reference"],
        source=typed_reference_fixture(tmp_path),
        robots=["robotwin/robotwin-dual-piper"],
        modes=[mode],
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "docker":
            return SimpleNamespace(stdout="sha256:" + "a" * 64 + "\n", returncode=0)
        assert "agent='oracle'" in command[3]
        assert "run_taskset" in command[3]
        assert kwargs["stdout"] is not None and kwargs["stderr"] is not None
        write_trial(out, receipt["cases"][0], reward=reward)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    result = tool.run_validation(out, deployments)
    case = result["cases"][0]
    assert result["all_selected_checks_passed"] is expected
    assert case["reference_replay_passed"] is (mode == "reference" and reward == 1)
    assert case["device_read_probe_passed"] is (mode == "probe")
    assert case["final_verifier_success"] is (reward == 1)
    assert result["fresh_process_replay_certificate"] is False
    assert result["model_calls"] == 0 and len(calls) == 2
    assert str(tmp_path) not in (out / "receipt.json").read_text()


def test_missing_completion_or_modified_journal_cannot_certify_a_reference(tmp_path):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    out = tmp_path / "validation"
    receipt = tool.prepare(
        deployments,
        wheel,
        out,
        backends=["robotwin"],
        cases=["robotwin/test-reference"],
        source=typed_reference_fixture(tmp_path),
        robots=["robotwin/robotwin-dual-piper"],
    )
    record = receipt["cases"][0]
    trial = write_trial(out, record, reward=1, program_completed=False)
    evidence = tool._trial_evidence(record, trial / "result.json")
    assert not evidence["reference_replay_passed"]
    (trial / "verifier/actions.jsonl").write_text("changed")
    evidence = tool._trial_evidence(record, trial / "result.json")
    assert not evidence["verifier_evidence_valid"] and not evidence["final_verifier_success"]


def test_gpu_workflow_requires_trusted_manual_context_and_privacy_gate():
    workflow = yaml.load(
        (ROOT / ".github/workflows/gpu-validation.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    job = workflow["jobs"]["validate"]
    assert "default_branch" in job["if"]
    assert job["environment"] == "gpu-validation"
    assert "self-hosted" in job["runs-on"]
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "steps.privacy.outcome == 'success'" in upload["if"]
    assert upload["with"]["path"] == "build/gpu-validation/receipt.json"


def test_unavailable_image_is_a_recorded_preflight_failure_not_a_trial(tmp_path, monkeypatch):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    out = tmp_path / "validation"
    tool.prepare(
        deployments,
        wheel,
        out,
        backends=["robotwin"],
        cases=["robotwin/handover-mic"],
        robots=["robotwin/robotwin-dual-piper"],
        modes=["probe"],
    )

    def fail(*args, **kwargs):
        raise FileNotFoundError("fixture Docker unavailable")

    monkeypatch.setattr(tool.subprocess, "run", fail)
    result = tool.run_validation(out, deployments)
    assert result["phase"] == "preflight-failed"
    assert not result["all_selected_checks_passed"]
    assert result["failure_category"] == "image-unavailable"
    assert not result["cases"][0]["execution_attempted"]


def test_modified_prepared_program_is_rejected_before_docker_or_harbor(tmp_path, monkeypatch):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    out = tmp_path / "validation"
    receipt = tool.prepare(
        deployments,
        wheel,
        out,
        backends=["robotwin"],
        cases=["robotwin/handover-mic"],
        robots=["robotwin/robotwin-dual-piper"],
        modes=["probe"],
    )
    program = out / "taskset" / receipt["cases"][0]["path"] / "solution/solution.py"
    program.write_text("# changed after preparation\n")
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="bundle changed"):
        tool.run_validation(out, deployments)


def test_case_selection_alone_narrows_probe_backends_without_guessing():
    tool = load_tool("validate_simulators")
    cases = tool.select_cases(
        cases=["robotwin/handover-mic"], robots=["robotwin/robotwin-dual-piper"], modes=["probe"]
    )
    assert len(cases) == 1 and cases[0][1].backend == "robotwin"


def test_failed_advertised_device_read_has_a_failure_category_without_harbor_exception(tmp_path):
    tool = load_tool("validate_simulators")
    record = {
        "id": "world-probe",
        "name": "roboenv/world-probe",
        "program_sha256": "a" * 64,
        "backend_task": "robotwin.handover_mic",
        "seed": 0,
        "mode": "probe",
        "expected_devices": ["left_arm", "world"],
        "expected_reads": ["left_arm", "world"],
    }
    trial = write_trial(tmp_path, record, reward=0, program_completed=False)
    path = trial / "verifier/actions.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(phase="error", response={"ok": False, "error": {"code": "device_failed"}})
    raw = "".join(json.dumps(event) + "\n" for event in events)
    path.write_text(raw)
    report_path = trial / "verifier/submit.json"
    report = json.loads(report_path.read_text())
    report["journal"]["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    report_path.write_text(json.dumps(report))
    evidence = tool._trial_evidence(record, trial / "result.json")
    assert evidence["verifier_evidence_valid"]
    assert not evidence["device_read_probe_passed"]
    assert evidence["failure_category"] == "device-read-failed"
    assert evidence["failed_device_reads"] == ["world"]


@pytest.mark.parametrize("backend,world_readable", [("libero", False), ("robocasa", False),
                                                   ("robotwin", False)])
def test_probe_reads_match_evidence_for_native_and_controller_worlds(
    tmp_path, monkeypatch, backend, world_readable,
):
    tool = load_tool("validate_simulators")
    wheel, deployments = inputs(tmp_path)
    receipt = tool.prepare(deployments, wheel, tmp_path / "prepared", backends=[backend],
                           modes=["probe"], seeds=[0])
    record = next(r for r in receipt["cases"] if "world" in r["expected_devices"])
    assert "world" in record["expected_devices"]
    assert ("world" in record["expected_reads"]) == world_readable
    reads = []
    inventory = {"schema_version": "roboenv/devices/1", "devices": [
        {"name": name, "type": "world" if name == "world" else "other", "actions": {
            "read": {"supported": True, "read_only": True},
        } if name in record["expected_reads"] else {}}
        for name in record["expected_devices"]
    ]}

    def read(name):
        reads.append(name)
        return {"read_only": True}

    monkeypatch.setattr("roboenv.client.Client", lambda: SimpleNamespace(
        devices=lambda: inventory,
        device=lambda name: SimpleNamespace(read=lambda: read(name)),
    ))
    exec(tool._probe_program(record["expected_devices"], record["expected_reads"]), {})
    assert reads == record["expected_reads"]
    trial = write_trial(tmp_path / "evidence", record, reward=0)
    assert tool._trial_evidence(record, trial / "result.json")["device_read_probe_passed"]


def test_original_version_one_probe_receipts_keep_their_read_requirements(tmp_path):
    tool = load_tool("validate_simulators")
    record = {
        "id": "original-probe", "name": "roboenv/original-probe", "seed": 0,
        "backend_task": "robotwin.handover_mic", "program_sha256": "a" * 64,
        "mode": "probe", "expected_devices": ["left_arm", "world"],
    }
    for reads in (["left_arm", "world"], ["left_arm"]):
        trial = write_trial(tmp_path / str(len(reads)), {**record, "expected_reads": reads},
                            reward=0)
        evidence = tool._trial_evidence(record, trial / "result.json")
        assert evidence["device_read_probe_passed"] == (len(reads) == 2)
        if len(reads) == 1:
            assert evidence["failure_category"] == "incomplete-device-read-evidence"


@pytest.mark.parametrize("identifier", [
    "robotwin/handover-mic", "robotwin/stack-blocks-three",
    "robodojo/general-pickup", "robodojo/stack-bowls",
])
def test_removed_native_reference_cannot_be_selected(identifier):
    tool = load_tool("validate_simulators")
    with pytest.raises(ValueError, match="requires an explicit regular solution.py"):
        tool.select_cases(backends=[identifier.split("/")[0]], cases=[identifier])
