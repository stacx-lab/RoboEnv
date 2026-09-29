"""roboenv verify certifies a local run, fail-closed, from its launch record and journal."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import roboenv
import yaml
from roboenv.cli.main import main
from roboenv.client import Client, RemoteError
from roboenv.environment import launch
from roboenv.verify import validate_journal

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"


def toy(tmp_path):
    """The fixture world under tmp_path, ending at the first successful control interval."""
    task = yaml.safe_load((FIXTURE / "task.yaml").read_text())
    task["backend_kwargs"] = {"max_steps": 100, "stop_on_success": True}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    return launch(path, runtime_root=tmp_path, catalog=FIXTURE / "catalog",
                  interpreter=sys.executable, run_dir=tmp_path / "runs", timeout=20)


def owner_of(run):
    return Client(run.record["socket"], session=run.record["session"],
                  generation=run.record["generation"])


def stale_success(out):
    """A previous certificate that any failed attempt must replace."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text('{"reward": 1.0}\n')
    (out / "submit.json").write_text('{"valid": true, "trial_success": true}\n')
    (out / "actions.jsonl").write_text("stale evidence\n")


def verdict_files(out):
    return {
        "report": json.loads((out / "submit.json").read_text()),
        "reward": json.loads((out / "reward.json").read_text()),
        "names": sorted(path.name for path in out.iterdir()),
    }


def test_cli_verify_certifies_a_successful_run(tmp_path):
    out = tmp_path / "verdict"
    env = {k: v for k, v in os.environ.items() if k not in {"ROBOENV_SOCKET", "ROBOENV_ENDPOINT"}}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "runtime/src")])
    with toy(tmp_path) as run:
        agent = roboenv.connect(run.endpoint)
        assert agent.device("controller").write([0.3])["simulation"]["terminal"] is True
        completed = subprocess.run(
            [sys.executable, "-m", "roboenv.cli.main", "verify", "--run", str(run.path),
             "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads(completed.stdout)
        assert report["valid"] is True and report["trial_success"] is True
        assert (report["task"], report["seed"], report["budget_mode"]) == (
            "native-toy-push", 0, "official",
        )
        assert (report["session"], report["generation"]) == (run.record["session"], 0)
        assert report["health"]["task"] == "native-toy-push"
        assert report["submit"]["completed"] and report["submit"]["submitted"]
        assert report["submit"]["budget"] == report["observation"]["budget"]
        assert report["submit"]["budget"]["used"] == 50
        files = verdict_files(out)
        assert files["names"] == ["actions.jsonl", "reward.json", "submit.json"]
        assert files["reward"] == {"reward": 1.0} and files["report"] == report
        raw = (out / "actions.jsonl").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert report["journal"]["sha256"] == digest and report["journal"]["requests"] >= 4
        assert report["journal_artifact"] == {
            "sha256": digest, "bytes": len(raw), "validated": True,
        }
        assert raw == (run.path.parent / "evidence/journal/actions.jsonl").read_bytes()
        # The certified world is frozen: a later agent write is refused before dispatch,
        # which the journal rules accept, while a write that took effect would not be.
        with pytest.raises(RemoteError) as refused:
            agent.device("controller").write([0.3])
        assert refused.value.code == "episode_finished"
        assert refused.value.details["effects"] == "none"
        # A status poll after the verdict, as `roboenv replay` makes on completion, is a
        # pure read that the supervisor proxies to the worker and journals: accepted.
        assert agent.status()["terminal"] is True
        later = (run.path.parent / "evidence/journal/actions.jsonl").read_bytes()
        committed = [json.loads(line) for line in later.splitlines()]
        ops = [e["request"]["op"] for e in committed if e["phase"] == "commit"]
        assert "simulation.status" in ops[ops.index("submit") + 1:]
        identity = {"session": run.record["session"], "submit_id": report["submit_request_id"],
                    "verdict": report["submit"]}
        assert validate_journal(later, **identity)["requests"] > report["journal"]["requests"]
        request = {
            "protocol": "roboenv/1", "session": run.record["session"], "generation": 0,
            "request_id": "forged-write", "deadline_s": 10, "op": "devices.call",
            "args": {"device": "controller", "action": "write", "args": {"target": [0.3]}},
        }
        response = {key: request[key] for key in ("protocol", "session", "generation",
                                                   "request_id")}
        forged = later + b"".join(
            json.dumps({"journal_version": 1, "phase": phase, "request": request, **extra}).encode()
            + b"\n"
            for phase, extra in (
                ("intent", {}),
                ("commit", {"response": {**response, "ok": True, "result": {"position": [0.6]}}}),
            )
        )
        with pytest.raises(ValueError, match="after final verdict"):
            validate_journal(forged, **identity)
        # The verifier's submit was the run's one evaluation: a second certificate is refused.
        stale_success(out)
        assert main(["verify", "--run", str(run.path), "--out", str(out)]) == 1
        files = verdict_files(out)
        assert files["reward"] == {"reward": 0.0} and files["report"]["valid"] is False
        assert "sole evaluation" in files["report"]["error"]


def test_cli_verify_fails_closed_on_identity_mismatch(tmp_path, capsys):
    out = tmp_path / "verdict"
    with toy(tmp_path) as run:
        roboenv.connect(run.endpoint).device("controller").write([0.3])
        manifest = run.path.parent / "environment.json"
        launched = json.loads(manifest.read_text())
        launched["task"]["seed"] = 1
        manifest.write_text(json.dumps(launched))
        stale_success(out)
        assert main(["verify", "--run", str(run.path), "--out", str(out)]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["valid"] is False and report["trial_success"] is False
        assert "identity differs" in report["error"]
        assert report["health"]["seed"] == 0 and report["seed"] == 1
        files = verdict_files(out)
        assert files["reward"] == {"reward": 0.0} and files["report"] == report
        assert files["names"] == ["actions.jsonl", "reward.json", "submit.json"]
        assert report["journal_artifact"]["validated"] is False
        assert (out / "actions.jsonl").read_bytes() == (
            run.path.parent / "evidence/journal/actions.jsonl"
        ).read_bytes()
        # The mismatch was detected before submit: a verifier with the recorded identity
        # still certifies the run afterwards.
        assert "submit" not in report
        launched["task"]["seed"] = 0
        manifest.write_text(json.dumps(launched))
        assert main(["verify", "--run", str(run.path), "--out", str(out)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["valid"] is True and report["trial_success"] is True
        assert verdict_files(out)["reward"] == {"reward": 1.0}


def test_cli_verify_rejects_a_run_submitted_before_the_verifier(tmp_path, capsys):
    out = tmp_path / "verdict"
    with toy(tmp_path) as run:
        agent = roboenv.connect(run.endpoint)
        agent.device("controller").write([0.3])
        owner = owner_of(run)
        owner.allowed_ops = owner.allowed_ops | {"submit"}
        verdict = owner.request("submit", request_id="agent-submit")
        assert verdict["completed"] is True
        with pytest.raises(RemoteError, match="episode_finished"):
            agent.device("controller").write([0.3])
        stale_success(out)
        assert main(["verify", "--run", str(run.path), "--out", str(out)]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["valid"] is False and report["trial_success"] is False
        assert "sole evaluation" in report["error"]
        assert report["submit"] == verdict  # the cached verdict is evidence, not a certificate
        files = verdict_files(out)
        assert files["reward"] == {"reward": 0.0} and files["report"] == report
        assert report["journal_artifact"]["validated"] is False


def test_cli_verify_refuses_a_stopped_or_unknown_run(tmp_path, capsys):
    out = tmp_path / "verdict"
    with toy(tmp_path) as run:
        roboenv.connect(run.endpoint).device("controller").write([0.3])
    assert not Path(run.record["socket"]).exists()
    stale_success(out)
    assert main(["verify", "--run", str(run.path), "--out", str(out)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False and report["trial_success"] is False
    assert "health failed" in report["error"] or "Error" in report["error"]
    assert "submit" not in report
    files = verdict_files(out)
    assert files["reward"] == {"reward": 0.0} and files["report"] == report
    # The journal outlives the service and is retained as evidence, unvalidated.
    assert report["journal_artifact"]["validated"] is False
    assert (out / "actions.jsonl").read_bytes() == (
        run.path.parent / "evidence/journal/actions.jsonl"
    ).read_bytes()
    stale_success(out)
    assert main(["verify", "--run", str(tmp_path / "absent/run.json"), "--out", str(out)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False and "run.json" in report["error"]
    assert "health" not in report and "submit" not in report
    files = verdict_files(out)
    assert files["reward"] == {"reward": 0.0} and files["report"] == report
    assert files["names"] == ["reward.json", "submit.json"]


def test_cli_verify_keeps_host_paths_out_of_the_verdict(tmp_path, capsys):
    # submit.json travels with the run; the run directory does not. A failure is
    # named by its type and errno text there, and in full on stderr.
    out = tmp_path / "verdict"
    with toy(tmp_path) as run:
        roboenv.connect(run.endpoint).device("controller").write([0.3])
        orphan = tmp_path / "orphan"
        orphan.mkdir()
        shutil.copy2(run.path, orphan / "run.json")  # no launch manifest beside it
        stale_success(out)
        assert main(["verify", "--run", str(orphan / "run.json"), "--out", str(out)]) == 1
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report == {
            "valid": False, "trial_success": False,
            "error": "FileNotFoundError: No such file or directory",
        }
        assert str(tmp_path) not in (out / "submit.json").read_text()
        assert str(orphan / "environment.json") in captured.err
        assert verdict_files(out)["names"] == ["reward.json", "submit.json"]
        # A record whose evidence directory is gone passes the identity check and is
        # submitted, but not certified: the verdict still names no path.
        moved = tmp_path / "moved"
        moved.mkdir()
        shutil.copy2(run.path.parent / "environment.json", moved / "environment.json")
        record = json.loads(run.path.read_text())
        record["directory"] = str(tmp_path / "missing-evidence")
        (moved / "run.json").write_text(json.dumps(record))
        stale_success(out)
        assert main(["verify", "--run", str(moved / "run.json"), "--out", str(out)]) == 1
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["valid"] is False and report["submit"]["submitted"] is True
        assert report["error"] == "FileNotFoundError: No such file or directory"
        assert report["journal_capture_error"] == report["error"]
        assert "journal_artifact" not in report
        assert str(tmp_path) not in (out / "submit.json").read_text()
        assert str(tmp_path / "missing-evidence/evidence/journal/actions.jsonl") in captured.err
        files = verdict_files(out)
        assert files["reward"] == {"reward": 0.0} and files["report"] == report
        assert files["names"] == ["reward.json", "submit.json"]
