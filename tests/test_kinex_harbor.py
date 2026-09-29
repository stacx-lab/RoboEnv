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

ENTRYPOINTS = (
    "bin/node",
    "bin/uv",
    "kinex/dist/application.js",
    "kinex/node_modules/@kinex/core/dist/index.js",
    "kinex/integrations/roboenv/run.ts",
)


def bundle(path, *, extra=None, corrupt=False, entrypoints=ENTRYPOINTS):
    contents = dict.fromkeys(entrypoints, b"fixture")
    if extra:
        contents[extra] = b"bad path"
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    identity = {
        "kinex_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    }
    if corrupt:
        contents["bin/node"] = b"changed"
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


def test_bundle_requires_node_kinex_and_the_bridge(tmp_path):
    assert inspect_bundle(bundle(tmp_path / "complete.tar.gz"))
    for missing in ENTRYPOINTS:
        entrypoints = tuple(name for name in ENTRYPOINTS if name != missing)
        path = bundle(tmp_path / "partial.tar.gz", entrypoints=entrypoints)
        with pytest.raises(ValueError, match="entrypoints"):
            inspect_bundle(path)
        path.unlink()


LOGINS = {
    "openai": {"type": "api_key", "key": "openai-secret"},
    "openai-codex": {"type": "oauth", "access": "codex-token", "expires": 1},
    "anthropic": {"type": "api_key", "key": "other-secret"},
}
SECRETS = ("openai-secret", "codex-token", "other-secret")
MODELS = ("openai/gpt-6", "openai-codex/gpt-6-astra")


def login(tmp_path, content=LOGINS):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(content))
    return {"KINEX_AUTH_JSON_PATH": str(path)}


def develop(tmp_path, **options):
    if "extra_env" not in options:
        options["extra_env"] = login(tmp_path)
    return {"mode": "develop", "model_name": "openai/gpt-6", "effort": "high", **options}


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


@pytest.mark.parametrize(
    "override",
    [
        {"model_name": "anthropic/claude"},
        {"model_name": "openai/"},
        {"model_name": "openai-codex/"},
        {"model_name": None},
        {"effort": None},
        {"effort": "off"},
        {"effort": "max"},
        {"extra_env": {}},
    ],
)
def test_invalid_development_configuration_fails_before_setup(tmp_path, monkeypatch, override):
    monkeypatch.delenv("KINEX_AUTH_JSON_PATH", raising=False)
    with pytest.raises(ValueError):
        KinexAgent(
            logs_dir=tmp_path,
            bundle=bundle(tmp_path / "agent.tar.gz"),
            **develop(tmp_path, **override),
        )


@pytest.mark.parametrize("model", MODELS)
def test_development_runs_the_selected_model_and_effort(tmp_path, model):
    agent = KinexAgent(
        logs_dir=tmp_path / "logs",
        bundle=bundle(tmp_path / "agent.tar.gz"),
        **develop(tmp_path, model_name=model),
    )
    agent.logs_dir.mkdir()
    uploads = {}

    class Environment:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target):
            uploads[target] = Path(source).read_text()

        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"passed": True, "mode": "develop"}))

    asyncio.run(agent.run("Fixture instruction", Environment(), AgentContext()))
    runner = json.loads(uploads[agent.root + "/runner.json"])
    assert runner["prompt"] == "Fixture instruction"
    provider, name = model.split("/")
    assert runner["model"] == {"provider": provider, "id": name, "effort": "high"}


def test_development_login_must_exist_for_the_selected_provider(tmp_path, monkeypatch):
    agent = bundle(tmp_path / "agent.tar.gz")
    # Like Harbor's own agents, the owner's process environment may name the login.
    monkeypatch.setenv("KINEX_AUTH_JSON_PATH", login(tmp_path)["KINEX_AUTH_JSON_PATH"])
    KinexAgent(logs_dir=tmp_path, bundle=agent, **develop(tmp_path, extra_env={}))
    # Each provider needs its own entry; a login for the other one is not used.
    for model in MODELS:
        provider = model.split("/")[0]
        other = "openai-codex" if provider == "openai" else "openai"
        for content in ({other: LOGINS[other]}, {provider: "key"}, []):
            with pytest.raises(ValueError, match="KINEX_AUTH_JSON_PATH"):
                KinexAgent(
                    logs_dir=tmp_path,
                    bundle=agent,
                    **develop(tmp_path, model_name=model, extra_env=login(tmp_path, content)),
                )


@pytest.mark.parametrize("model", MODELS)
def test_development_installs_only_the_selected_login_and_removes_it(tmp_path, model):
    agent = KinexAgent(
        logs_dir=tmp_path / "logs",
        bundle=bundle(tmp_path / "agent.tar.gz"),
        **develop(tmp_path, model_name=model),
    )
    agent.logs_dir.mkdir()
    commands, uploads = [], {}

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append((command, kwargs))
            failed = "runner.json >" in command
            return SimpleNamespace(return_code=int(failed), stdout="", stderr="")

        async def upload_file(self, source, target):
            uploads[target] = Path(source).read_text()

        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"passed": True, "mode": "develop"}))

    # A failed run still removes the login before the workspace is exported.
    with pytest.raises(RuntimeError, match="Kinex command exited 1"):
        asyncio.run(agent.run("Fixture instruction", Environment(), AgentContext()))
    provider = model.split("/")[0]
    assert json.loads(uploads[agent.auth + ".upload"]) == {provider: LOGINS[provider]}
    order = [command for command, _ in commands]
    install = order.index(
        f"install -m 600 {agent.auth}.upload {agent.auth} && rm {agent.auth}.upload"
    )
    run = next(index for index, command in enumerate(order) if "runner.json >" in command)
    remove = order.index(f"rm -f {agent.auth} {agent.auth}.upload")
    export = next(index for index, command in enumerate(order) if "workspace.tar.gz" in command)
    assert install < run < remove < export
    # The login never appears in a command, its environment or the trial's logs.
    logs = [path.read_text() for path in agent.logs_dir.rglob("*") if path.is_file()]
    assert all(options.get("user") is None for _, options in commands)
    assert not any(secret in str([logs, commands]) for secret in SECRETS)


def fake_kinex(root, *, installed="1.0.0"):
    def write(relative, content=""):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))

    agent = "node_modules/@earendil-works/pi-coding-agent"
    write("package.json", {"name": "kinex", "engines": {"node": ">=22.19.0"}})
    write(
        "package-lock.json",
        {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "kinex"},
                agent: {"version": "1.0.0"},
                agent + "/node_modules/nested": {"version": "2.0.0"},
                "node_modules/@esbuild/other-platform": {"version": "1.0.0", "optional": True},
                "node_modules/typescript": {"version": "5.9.3", "dev": True},
                "node_modules/@kinex/core": {"resolved": "packages/core", "link": True},
                "packages/core": {"version": "0.9.0"},
            },
        },
    )
    write("bin/kinex.mjs")
    write("dist/application.js")
    write(agent + "/package.json", {"version": installed})
    write(agent + "/dist/index.js")
    write(agent + "/node_modules/nested/package.json", {"version": "2.0.0"})
    write("node_modules/typescript/package.json", {"version": "5.9.3"})
    write("packages/core/package.json", {"version": "0.9.0"})
    write("packages/core/dist/index.js")
    write("packages/core/src/index.ts")
    write("packages/core/assets/python/kinex/__init__.py")
    write("packages/core/assets/python/kinex/__pycache__/cached.pyc")
    write(".git/config", "never packaged")
    (root / "node_modules/@kinex").mkdir()
    (root / "node_modules/@kinex/core").symlink_to("../../packages/core")
    return root


def executable(path, output):
    path.write_text(f"#!/bin/sh\necho {output}\n")
    path.chmod(0o755)
    return path


def test_packaged_kinex_contains_its_production_runtime_only(tmp_path):
    package = load_tool("package_kinex_agent").package
    node = executable(tmp_path / "node", "v22.19.0")
    result = package(
        fake_kinex(tmp_path / "kinex"), node, executable(tmp_path / "uv", "uv"), tmp_path / "a.tgz"
    )
    assert inspect_bundle(result["bundle"]) == {"kinex_sha256": result["kinex_sha256"]}
    with tarfile.open(result["bundle"]) as archive:
        names = set(archive.getnames())
    assert {
        "bin/node",
        "bin/uv",
        "kinex/package-lock.json",
        "kinex/node_modules/@kinex/core/package.json",
        "kinex/node_modules/@kinex/core/assets/python/kinex/__init__.py",
        "kinex/node_modules/@earendil-works/pi-coding-agent/node_modules/nested/package.json",
        "kinex/integrations/roboenv/run.ts",
    } <= names
    excluded = ("typescript", ".git", "/src/", "__pycache__", "kinex/packages/", "other-platform")
    assert not [name for name in names if any(part in name for part in excluded)]


@pytest.mark.parametrize("problem", ["unlocked", "old-node", "unbuilt", "external-link"])
def test_packaging_rejects_unlocked_old_unbuilt_or_linked_kinex(tmp_path, problem):
    package = load_tool("package_kinex_agent").package
    kinex = fake_kinex(tmp_path / "kinex", installed="1.0.1" if problem == "unlocked" else "1.0.0")
    if problem == "unbuilt":
        (kinex / "dist/application.js").unlink()
    if problem == "external-link":
        (tmp_path / "outside.js").write_text("")
        target = kinex / "node_modules/@earendil-works/pi-coding-agent/dist/linked.js"
        target.symlink_to(tmp_path / "outside.js")
    node = executable(tmp_path / "node", "v22.18.0" if problem == "old-node" else "v22.19.0")
    with pytest.raises(ValueError):
        package(kinex, node, executable(tmp_path / "uv", "uv"), tmp_path / "agent.tar.gz")
    assert not (tmp_path / "agent.tar.gz").exists()


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
