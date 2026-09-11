"""Generation is deterministic, portable, answer-free, and checked by Harbor itself."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from harbor.models.job.config import JobConfig
from roboenv.harbor.export import export_task
from roboenv.harbor.taskset import export_taskset, run_taskset, validate_harbor_task
from roboenv.tasks.catalog import BACKENDS, discover, summary

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"test-only build input; never installed")
    mapping = {}
    for backend in BACKENDS:
        deployment = tmp_path / f"{backend}.json"
        depot = tmp_path / f"{backend}-assets"
        depot.mkdir()
        deployment.write_text(
            json.dumps(
                {
                    "schema": "roboenv/deployment/1",
                    "backend": backend,
                    "image": "sha256:" + "a" * 64,
                    "legacy_tree_required": False,
                    "mounts": [
                        {
                            "source": depot.name,
                            "target": "/opt/roboenv/runtime/assets",
                            "read_only": True,
                        }
                    ],
                }
            )
        )
        mapping[backend] = deployment.name
    deployments = tmp_path / "deployments.json"
    deployments.write_text(json.dumps(mapping))
    return wheel, deployments


def test_catalog_batch_is_standard_and_contains_no_answers_or_host_paths(tmp_path):
    wheel, deployments = inputs(tmp_path)
    destination = tmp_path / "generated"
    result = export_taskset(
        ROOT, ROOT / "catalog/tasks", deployments, destination, wheel, seeds=[0, 7]
    )
    assert result["references"] == 0
    assert result["tasks"] == 2 * summary(ROOT)["tasks"]
    assert result["harbor_loader_validated"] is True
    assert result["execution_verified"] is False
    config = JobConfig.model_validate_json((destination / "job.json").read_text())
    assert len(config.tasks) == result["tasks"]
    for item in config.tasks:
        assert not item.path.is_absolute()
        path = destination / item.path
        assert validate_harbor_task(path)["name"].startswith("roboenv/")
        assert not (path / "solution").exists()
        compose = yaml.safe_load((path / "environment/docker-compose.yaml").read_text())
        assert any("ROBOENV_DEPOT_" in v for v in compose["services"]["sim"]["volumes"])
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix != ".whl":
            assert str(tmp_path) not in path.read_text()
    with pytest.raises(ValueError, match="Oracle requires"):
        run_taskset(destination, deployments, agent="oracle")


def test_conflicts_are_detected_before_any_task_is_written(tmp_path):
    wheel, deployments = inputs(tmp_path)
    out = tmp_path / "generated"
    out.mkdir()
    (out / "job.json").write_text("user-owned configuration")
    with pytest.raises(ValueError, match="unmanaged output"):
        export_taskset(ROOT, ROOT / "catalog/tasks", deployments, out, wheel)
    assert not (out / "tasks").exists()
    assert (out / "job.json").read_text() == "user-owned configuration"


def test_runner_ignores_unselected_unprepared_deployments(tmp_path, monkeypatch):
    wheel, deployments = inputs(tmp_path)
    result_dir = tmp_path / "generated"
    export_taskset(
        ROOT, ROOT / "catalog/tasks", deployments, result_dir, wheel, backends=["libero"]
    )
    mapping = json.loads(deployments.read_text())
    mapping["unused"] = "not-prepared-yet.json"
    deployments.write_text(json.dumps(mapping))
    calls = []
    monkeypatch.setattr("roboenv.harbor.taskset.shutil.which", lambda executable: executable)
    monkeypatch.setattr(
        "roboenv.harbor.taskset.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    assert run_taskset(result_dir, deployments, agent="test-agent") == 0
    assert calls[0][1]["cwd"] == result_dir
    assert "ROBOENV_DEPOT_LIBERO_0" in calls[0][1]["env"]


def test_generator_discovers_when_source_is_omitted(tmp_path, monkeypatch):
    wheel, deployments = inputs(tmp_path)
    called = []

    def discover(project, out, backends):
        called.append(tuple(backends))
        shutil.copytree(ROOT / "catalog/tasks", Path(out) / "tasks")

    monkeypatch.setattr("roboenv.harbor.taskset.discover", discover)
    result = export_taskset(ROOT, None, deployments, tmp_path / "generated", wheel)
    assert called == [BACKENDS]
    assert result["references"] == 0


def test_project_reference_is_opt_in_and_not_an_official_certificate(tmp_path):
    wheel, _ = inputs(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    task = source / "task.yaml"
    shutil.copy2(ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml", task)
    (source / "solution.py").write_text("# project-authored test reference\n")
    out = tmp_path / "task"
    export_task(
        task,
        ROOT / "catalog",
        out,
        wheel,
        deployment_path=tmp_path / "robocasa.json",
        include_reference=True,
    )
    assert validate_harbor_task(out)
    note = json.loads((out / "solution/source.json").read_text())
    assert note["origin"] == "project-reference"
    assert note["official"] is False and note["replay_verified"] is False
    assert len(note["program_sha256"]) == 64
    (source / "solution-source.json").write_text('{"official":true}')
    with pytest.raises(ValueError, match="upstream URL and revision"):
        export_task(
            task,
            ROOT / "catalog",
            tmp_path / "bad",
            wheel,
            deployment_path=tmp_path / "robocasa.json",
            include_reference=True,
        )
    assert not (tmp_path / "bad").exists()


def test_missing_verifier_is_rejected_by_harbor(tmp_path):
    wheel, _ = inputs(tmp_path)
    out = tmp_path / "task"
    export_task(
        ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml",
        ROOT / "catalog",
        out,
        wheel,
        deployment_path=tmp_path / "robocasa.json",
    )
    (out / "tests/test.sh").rename(out / "tests/not-a-verifier.sh")
    with pytest.raises(ValueError, match="Harbor rejected"):
        validate_harbor_task(out)


@pytest.mark.skipif(
    os.environ.get("ROBOENV_TEST_UPSTREAMS") != "1",
    reason="optional full source-catalog test; needs pinned upstream checkouts, no GPU",
)
def test_every_discoverable_upstream_task_passes_harbor_loader(tmp_path):
    # Fixture image/wheel are deliberately non-executable: this tests every
    # generated definition against Harbor, not simulator installation or success.
    wheel, deployments = inputs(tmp_path)
    discovery = discover(ROOT, tmp_path / "source")
    result = export_taskset(
        ROOT, tmp_path / "source/tasks", deployments, tmp_path / "harbor", wheel
    )
    assert result["tasks"] == discovery["tasks"]
    assert set(result["backends"]) == set(BACKENDS)
    assert result["references"] == 0
    assert result["execution_verified"] is False
    print(json.dumps({"discovered": discovery, "harbor": result}, sort_keys=True))


def test_discovery_emits_namespaced_catalog_with_explicit_clock(tmp_path, monkeypatch):
    from roboenv.tasks import catalog

    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    monkeypatch.setitem(
        catalog.DISCOVERERS,
        "robocasa",
        lambda *a: [
            (catalog._task("robocasa", "robocasa.Example", {"task_name": "Example"}, "Do it"),
             ROOT / "README.md")
        ],
    )
    destination = tmp_path / "discovered"
    discover(ROOT, destination, ["robocasa"])
    task, path = catalog.load_catalog(destination / "tasks")["robocasa/Example"]
    assert path == destination / "tasks/robocasa/Example/task.yaml"
    assert task.backend_task == "robocasa.Example"
    assert task.backend_kwargs["realtime"] is True


def test_nested_catalog_duplicate_ids_are_rejected(tmp_path):
    source = ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml"
    for name in ("one", "two"):
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        shutil.copy2(source, directory / "task.yaml")
    from roboenv.tasks.catalog import load_catalog

    with pytest.raises(ValueError, match="duplicate task id"):
        load_catalog(tmp_path)


@pytest.mark.parametrize("identifier", [
    "robotwin/handover-mic", "robotwin/stack-blocks-three",
    "robodojo/general-pickup", "robodojo/stack-bowls",
])
def test_retired_primitive_reference_is_preserved_but_cannot_be_exported(tmp_path, identifier):
    import hashlib

    wheel, _ = inputs(tmp_path)
    task = ROOT / "catalog/tasks" / identifier / "task.yaml"
    reference = task.with_name("solution.py")
    provenance = json.loads(task.with_name("solution-source.json").read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == provenance["program_sha256"]
    destination = tmp_path / "retired"
    with pytest.raises(ValueError, match="reference program is retired"):
        export_task(
            task, ROOT / "catalog", destination, wheel,
            deployment_path=tmp_path / (identifier.split("/")[0] + ".json"),
            include_reference=True,
        )
    assert not destination.exists()
