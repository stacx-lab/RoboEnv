"""Task discovery, catalog loading and reference-program policy without a simulator."""

import ast
import builtins
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from roboenv.tasks import catalog
from roboenv.tasks.catalog import BACKENDS, discover, load_catalog
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_emits_namespaced_catalog_with_explicit_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    monkeypatch.setitem(
        catalog.DISCOVERERS,
        "robocasa",
        lambda *a: [
            (
                catalog._task("robocasa", "robocasa.Example", {"task_name": "Example"}, "Do it"),
                ROOT / "README.md",
            )
        ],
    )
    destination = tmp_path / "discovered"
    discover(ROOT, destination, ["robocasa"])
    task, path = load_catalog(destination / "tasks")["robocasa/Example"]
    assert path == destination / "tasks/robocasa/Example/task.yaml"
    assert task.backend_task == "robocasa.Example"
    assert "realtime" not in task.backend_kwargs
    assert yaml.safe_load(path.read_text())["execution_mode"] == "realtime"


def test_nested_catalog_duplicate_ids_are_rejected(tmp_path):
    source = ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml"
    for name in ("one", "two"):
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        shutil.copy2(source, directory / "task.yaml")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_catalog(tmp_path)


@pytest.mark.skipif(
    os.environ.get("ROBOENV_TEST_UPSTREAMS") != "1",
    reason="optional full source-catalog test; needs pinned upstream checkouts, no GPU",
)
def test_every_discoverable_upstream_task_resolves(tmp_path):
    # Every discovered definition must name a checked-in robot with a binding plan; this
    # tests the generated catalog, not simulator installation or task success.
    discovery = discover(ROOT, tmp_path / "source")
    inventory = load_catalog(tmp_path / "source/tasks")
    assert len(inventory) == discovery["tasks"]
    assert {task.backend for task, _ in inventory.values()} == set(BACKENDS)
    for task, path in inventory.values():
        resolved = resolve_task(path, catalog=ROOT / "catalog")
        assert resolved.task.id == task.id and resolved.bindings.devices
    print(json.dumps({"discovered": discovery, "resolved": len(inventory)}, sort_keys=True))


def test_catalog_references_run_in_replay_or_are_retired():
    # `roboenv replay` gives a reference program `env` and `np`; one written for an API that
    # is gone must be marked retired, with a reason, so validation skips it instead of
    # replaying a program that cannot run against device-only worlds.
    for reference in sorted((ROOT / "catalog/tasks").glob("*/*/solution.py")):
        note = reference.with_name("solution-source.json")
        provenance = json.loads(note.read_text()) if note.is_file() else {}
        assert isinstance(provenance, dict)
        assert type(provenance.get("retired", False)) is bool
        if provenance.get("retired") is True:
            assert isinstance(provenance.get("reason"), str) and provenance["reason"], reference
            continue
        tree = ast.parse(reference.read_text())
        defined = set(dir(builtins)) | {"env", "np", "__name__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        used = {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert not used - defined, (reference, sorted(used - defined))
