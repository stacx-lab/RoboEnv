"""Harbor owns scheduling; the optional Kinex bridge validates isolated agent inputs."""

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from helpers import load_tool
from roboenv.harbor.kinex import KinexAgent, inspect_bundle
from roboenv.harbor.taskset import export_taskset, run_taskset
from test_taskset import ROOT, inputs


def bundle(path, *, extra=None, corrupt=False):
    contents = dict.fromkeys(
        ("bin/bun", "bin/uv", "kinex/bin/kinex", "kinex/integrations/roboenv/run.ts"), b"fixture"
    )
    if extra:
        contents[extra] = b"bad path"
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    identity = {
        "kinex_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    }
    if corrupt:
        contents["bin/bun"] = b"changed"
    contents["bundle.json"] = json.dumps(
        {"schema": "roboenv/kinex-agent-bundle/1", "files": hashes, "identity": identity}
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in contents.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return path


@pytest.mark.parametrize(
    "extra,corrupt", [("../escape", False), ("/absolute", False), (None, True)]
)
def test_bundle_rejects_unsafe_paths_and_changed_bytes_before_effects(tmp_path, extra, corrupt):
    path = bundle(tmp_path / "agent.tar.gz", extra=extra, corrupt=corrupt)
    with pytest.raises(ValueError):
        inspect_bundle(path)


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "unknown"},
        {"model_name": "openai/test"},
        {"mode": "develop"},
        {"setup_timeout_sec": float("inf")},
        {"tool": "../../escape"},
    ],
)
def test_invalid_agent_configuration_fails_before_setup(tmp_path, options):
    with pytest.raises(ValueError):
        KinexAgent(logs_dir=tmp_path, bundle=bundle(tmp_path / "agent.tar.gz"), **options)


def test_harbor_taskset_preserves_scheduler_settings_and_external_agent_config(
    tmp_path, monkeypatch
):
    wheel, deployments = inputs(tmp_path)
    destination = tmp_path / "taskset"
    export_taskset(
        ROOT, ROOT / "catalog/tasks", deployments, destination, wheel, backends=["libero"]
    )
    job = json.loads((destination / "job.json").read_text())
    job.update(n_attempts=2, n_concurrent_trials=3)
    (destination / "job.json").write_text(json.dumps(job))
    options = tmp_path / "agent.json"
    options.write_text(json.dumps({"bundle": "/explicit/agent.tar.gz", "mode": "no-model"}))
    monkeypatch.setattr("roboenv.harbor.taskset.shutil.which", lambda _: "/bin/harbor")

    def run(command, **kwargs):
        configured = JobConfig.model_validate_json(Path(command[3]).read_text())
        assert configured.n_attempts == 2 and configured.n_concurrent_trials == 3
        assert configured.retry.max_retries == 0
        assert configured.agents[0].import_path == "roboenv.harbor.kinex:KinexAgent"
        assert configured.agents[0].kwargs["mode"] == "no-model"
        assert all(task.path.is_absolute() for task in configured.tasks)
        assert kwargs["cwd"] == destination
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("roboenv.harbor.taskset.subprocess.run", run)
    assert run_taskset(destination, deployments, agent="kinex", agent_config=options) == 0
    assert json.loads((destination / "job.json").read_text()) == job


def test_zero_exit_without_execution_receipt_cannot_pass(tmp_path):
    agent = KinexAgent(logs_dir=tmp_path, bundle=bundle(tmp_path / "agent.tar.gz"))
    commands = []

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, *args):
            pass

        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"passed": True} if "integrity" in source else {}))

    context = AgentContext()
    with pytest.raises(ValueError, match="execution.json"):
        asyncio.run(agent.run("Fixture instruction", Environment(), context))
    assert context.metadata["completed"] is False
    assert any("prepare.py verify" in command for command in commands)


def test_harbor_report_requires_receipts_and_distinguishes_native_success(tmp_path):
    collect = load_tool("report_kinex_harbor").collect
    content = {
        "result.json": {"finished_at": "finished", "n_total_trials": 1},
        "trial/result.json": {"finished_at": "finished", "exception_info": None},
        "trial/agent/execution.json": {"passed": True, "mode": "no-model"},
        "trial/agent/integrity.json": {"passed": True},
        "trial/agent/runner.json": {"recording": {}},
        "trial/verifier/submit.json": {"valid": True, "trial_success": False},
    }
    for name, value in content.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    report = collect(tmp_path)
    assert report["passed"] and not report["trials"][0]["native_success"]
    assert not report["fresh_process_replay_verified"]
    assert not collect(tmp_path, require_task_success=True)["passed"]
    (tmp_path / "trial/agent/integrity.json").unlink()
    assert not collect(tmp_path)["passed"]
