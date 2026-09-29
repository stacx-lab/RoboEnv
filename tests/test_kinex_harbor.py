"""Harbor owns scheduling; the optional Kinex bridge validates isolated agent inputs."""

import asyncio
import contextlib
import hashlib
import io
import json
import math
import runpy
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
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


def develop(**options):
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
        {"model_name": "openai-codex/gpt-6-astra"},
        {"model_name": "openai/"},
        {"model_name": None},
        {"effort": None},
        {"effort": "off"},
        {"effort": "max"},
    ],
)
def test_invalid_development_configuration_fails_before_setup(tmp_path, override):
    with pytest.raises(ValueError):
        KinexAgent(
            logs_dir=tmp_path, bundle=bundle(tmp_path / "agent.tar.gz"), **develop(**override)
        )


def test_development_runs_the_selected_model_and_effort(tmp_path):
    agent = KinexAgent(
        logs_dir=tmp_path / "logs", bundle=bundle(tmp_path / "agent.tar.gz"), **develop()
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
    assert runner["model"] == {"provider": "openai", "id": "gpt-6", "effort": "high"}


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


PROBE = ROOT / "integrations/kinex/probe"


class ProbeWorld:
    """RoboEnv 0.9 device semantics for the default no-model probe.

    Joint writes are timed segments in model order; the arm moves only while the
    world clock advances, as in a stepped world. Writes in any other form fail
    like RoboEnv's schema does.
    """

    def __init__(self, positions, *, limits=(-2.9, 2.9), moves=True, tau_s=0.4):
        self.positions, self.moves, self.tau_s = list(positions), moves, tau_s
        self.velocities = [0.0] * len(positions)
        self.goal = list(positions)
        self.time_s = 0.0
        self.joints = [
            {"name": f"joint{index}", "type": "revolute",
             "limits": {"min": limits[0], "max": limits[1]}}
            for index in range(len(positions))
        ]
        self.calls, self.finished = [], None

    def profile(self, name):
        assert name == "arm"
        return {"name": "arm", "type": "joint",
                "intrinsics": {"model": {"joints": self.joints}}}

    def device(self, name, action="read", **kwargs):
        self.calls.append((name, action, kwargs))
        if name == "camera" and action == "read":
            frame = f"/frames/{len(self.calls)}.png"
            return {"data": {"sample_time_s": self.time_s}, "file": frame}
        if name == "world" and action == "wait":
            duration = kwargs["duration_s"]
            if self.moves:
                decay = math.exp(-duration / self.tau_s)
                pairs = list(zip(self.positions, self.goal, strict=True))
                self.positions = [g + (p - g) * decay for p, g in pairs]
                self.velocities = [(g - p) * decay / self.tau_s for p, g in pairs]
            self.time_s += duration
            return {"data": {"time_s": self.time_s, "elapsed_s": duration}}
        if name != "arm":
            raise RuntimeError(f"unexpected device call {name}.{action}")
        if action == "read":
            return {"data": {"positions": list(self.positions),
                             "velocities": list(self.velocities), "faults": []}}
        if action == "write":
            if set(kwargs) != {"actions", "duration_s"}:
                raise RuntimeError("write requires actions and duration_s")
            actions, duration = kwargs["actions"], kwargs["duration_s"]
            if not actions or not duration > 0:
                raise RuntimeError("write requires a nonempty timed segment")
            for target in actions:
                assert len(target) == len(self.joints)
                assert all(math.isfinite(value) for value in target)
            self.goal = list(actions[-1])
            return {"data": {"execution_id": "segment", "elapsed_s": 0.0, "state": {}}}
        if action == "stop":
            return {"data": {"stopped": True}}
        raise RuntimeError(f"unsupported arm action {action}")

    def run(self, monkeypatch, arguments):
        class Finished(SystemExit):
            pass

        def finish(result=None, file=None):
            self.finished = {**result, "file": file}
            raise Finished(0)

        kinex = ModuleType("kinex")
        kinex.args = lambda: dict(arguments)
        kinex.device, kinex.profile, kinex.finish = self.device, self.profile, finish
        monkeypatch.setitem(sys.modules, "kinex", kinex)
        # Hold rounds also keep wall time for the recorder; skip it here.
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        with contextlib.suppress(Finished):
            runpy.run_path(str(PROBE / "tool.py"), run_name="__main__")


@pytest.mark.parametrize("start,direction", [(0.2, 1), (2.85, -1)])
def test_default_probe_moves_the_arm_with_a_timed_segment_and_world_waits(
    monkeypatch, start, direction
):
    world = ProbeWorld([start, -0.16, 0.0, -2.44, 0.0, 2.23, 0.79])
    world.run(monkeypatch, {"arm": "arm", "camera": "camera"})
    writes = [kwargs for name, action, kwargs in world.calls if action == "write"]
    assert len(writes) == 1 and writes[0]["duration_s"] > 0
    # A smooth segment from the measured configuration; other joints hold.
    assert writes[0]["actions"][-1][0] == pytest.approx(start + 0.1 * direction)
    held = [-0.16, 0.0, -2.44, 0.0, 2.23, 0.79]
    assert all(action[1:] == held for action in writes[0]["actions"])
    first_write = next(i for i, (_, action, _) in enumerate(world.calls) if action == "write")
    assert ("world", "wait") in [(name, action) for name, action, _ in world.calls[first_write:]]
    assert world.calls[-1][:2] == ("arm", "stop")
    result = world.finished
    # require_completion accepts only a returned completed: true.
    assert result["completed"] is True and result["reached"] is True
    assert result["after"]["positions"][0] - result["before"]["positions"][0] == pytest.approx(
        0.1 * direction, abs=0.01
    )
    assert result["file"].endswith(".png") and result["world_elapsed_s"] > 0


def test_default_probe_fails_and_stops_when_writes_do_not_move_the_arm(monkeypatch):
    world = ProbeWorld([0.0] * 7, moves=False)
    with pytest.raises(RuntimeError, match="did not converge"):
        world.run(monkeypatch, {"arm": "arm", "camera": "camera"})
    assert world.finished is None
    assert world.calls[-1][:2] == ("arm", "stop")


def test_default_probe_declares_its_devices_for_kinex():
    manifest = yaml.safe_load((PROBE / "tool.yaml").read_text())
    assert manifest["name"] == "integration_probe" and manifest["kind"] == "controller"
    assert set(manifest["devices"]) == {"joint", "world"}
    properties = manifest["parameters"]["properties"]
    assert manifest["parameters"]["required"] == ["arm", "camera"]
    assert {"arm", "camera", "world"} == set(properties) == set(manifest["device_inputs"])
    assert all(properties[name]["type"] == "string" for name in properties)
