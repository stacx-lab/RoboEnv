"""Bounded episodes use the real worker, transport and durable owner snapshot."""

import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import roboenv
import yaml
from roboenv.client import Client, RemoteError
from roboenv.environment import launch, status, stop

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"


def bounded(tmp_path, *, limit=100, success=True, allow_stop=True, execution_mode=None):
    task = yaml.safe_load((FIXTURE / "task.yaml").read_text())
    task["backend_kwargs"] = {"max_steps": limit, "stop_on_success": success}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    return launch(path, runtime_root=tmp_path, catalog=FIXTURE / "catalog",
                  interpreter=sys.executable, run_dir=tmp_path / "runs", allow_stop=allow_stop,
                  timeout=20, execution_mode=execution_mode)


def test_realtime_ignores_explicit_step_cap_and_keeps_counting(tmp_path):
    with bounded(tmp_path, limit=50, success=False, execution_mode="realtime") as run:
        env = roboenv.connect(run.endpoint)
        for _ in range(3):
            env.device("controller").write([0.01])
        status = env.status()
        assert not status["terminal"]
        assert status["budget"]["used"] == 150 and status["budget"]["max"] is None
        env.device("controller").read()


def test_external_yaml_without_checkout_and_success_at_limit(tmp_path):
    with bounded(tmp_path, limit=50) as run:
        env = roboenv.connect(run.endpoint)
        assert env.status()["budget"]["used"] == 0
        assert "success" not in env.status() and "reason" not in env.status()
        env.device("controller").read()
        assert env.status()["budget"]["used"] == 0
        result = env.device("controller").write([0.3])
        assert result["simulation"]["terminal"] is True
        assert "success" not in result["simulation"] and "reason" not in result["simulation"]
        assert result["simulation"]["budget"]["used"] == 50
        with pytest.raises(RemoteError) as caught:
            env.device("controller").write([0.3])
        assert caught.value.code == "episode_finished"
        assert caught.value.details["effects"] == "none"
        assert "success" not in caught.value.details and "reason" not in caught.value.details
        assert "success" not in str(caught.value)
        assert "objects" not in env.status()
        terminal = json.loads((run.path.parent / "evidence/terminal.json").read_text())
        assert terminal["verdict"]["completed"] is True
        assert terminal["status"]["success"] is True
        assert terminal["status"]["reason"] == "success"
        owner = Client(run.record["socket"], session=run.record["session"],
                       generation=run.record["generation"])
        owner.allowed_ops = owner.allowed_ops | {"submit"}
        # A late stop may cancel buffers that the terminal frame already cleaned.
        # Refusals must leave the journal certifiable on either side of submit.
        for phase in ("before", "after"):
            with pytest.raises(RemoteError) as cancelled:
                owner.request("devices.cancel", {"execution_id": "ended-buffer"},
                              request_id=f"cancel-{phase}")
            assert cancelled.value.code == "episode_finished"
            assert cancelled.value.details["effects"] == "none"
            if phase == "before":
                verdict = owner.request("submit", request_id="verifier-submit")
        raw = (run.path.parent / "evidence/journal/actions.jsonl").read_bytes()
        validate = runpy.run_path(str(ROOT / "src/roboenv/assets/verify.py"))["validate_journal"]
        assert validate(raw, session=run.record["session"], submit_id="verifier-submit",
                        verdict=verdict)
        assert verdict == terminal["verdict"]
        stopped = env.stop()
        assert stopped["success"] is True and stopped["reason"] == "success"
        assert "success" not in env.status() and "reason" not in env.status()


def test_step_limit_freezes_and_stop_receipt_survives_worker_shutdown(tmp_path):
    with bounded(tmp_path, limit=100) as run:
        env = roboenv.connect(run.endpoint)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(env.device("controller").write, [0.01]) for _ in range(4)]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except RemoteError as exc:
                    assert exc.code == "episode_finished"
        assert len(results) == 2
        result = env.status()
        assert result["terminal"] and result["budget"]["used"] == 100
        assert "success" not in result and "reason" not in result
        stopped = stop(run.endpoint)
        assert stopped["reason"] == "step_limit" and stopped["success"] is False
        assert stopped["service_state"] == "stopped"
        assert stop(run.endpoint) == stopped
        assert status(run.endpoint)["simulation"] == {
            key: value for key, value in stopped.items() if key not in {"success", "reason"}
        }
    assert status(run.path)["state"] == "stopped"
    assert status(run.path)["simulation"]["budget"]["used"] == 100
    assert "success" not in status(run.path)["simulation"]
    assert "reason" not in status(run.path)["simulation"]


def test_stop_requires_explicit_owner_permission(tmp_path):
    with bounded(tmp_path, allow_stop=False) as run:
        with pytest.raises(RemoteError, match="owner-only"):
            stop(endpoint=run.endpoint, project=ROOT)
        assert Client(run.endpoint).request("health")["worker"]["alive"]


def test_early_stop_is_immutable_and_concurrent_stops_are_idempotent(tmp_path):
    with bounded(tmp_path) as run:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: stop(run.endpoint), range(3)))
        assert results[0] == results[1] == results[2]
        assert results[0]["reason"] == "stopped"
        assert results[0]["success"] is False
        assert results[0]["budget"]["used"] == 0
        assert "success" not in status(run.endpoint)["simulation"]
