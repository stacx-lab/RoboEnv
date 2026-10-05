"""Task discovery and validation without an agent framework."""

import os
import shutil
from pathlib import Path

import pytest
import yaml
from roboenv.tasks.catalog import BACKENDS, discover, load_catalog
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_emits_namespaced_catalog_with_explicit_clock(tmp_path, monkeypatch):
    from roboenv.tasks import catalog

    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    monkeypatch.setitem(
        catalog.DISCOVERERS,
        "robocasa",
        lambda *a: [
            (
                catalog._task("robocasa", "Example", {"task_name": "Example"}, "Do it"),
                ROOT / "README.md",
            )
        ],
    )
    destination = tmp_path / "discovered"
    discover(ROOT, destination, ["robocasa"])
    task, path = catalog.load_catalog(destination / "tasks")["robocasa/example"]
    assert path == destination / "tasks/robocasa/example/task.yaml"
    assert task.backend_kwargs["task_name"] == "Example"
    assert "realtime" not in task.backend_kwargs
    assert yaml.safe_load(path.read_text())["execution_mode"] == "realtime"


def test_nested_catalog_duplicate_ids_are_rejected(tmp_path):
    source = ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml"
    for name in ("one", "two"):
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        shutil.copy2(source, directory / "task.yaml")
    from roboenv.tasks.catalog import load_catalog

    with pytest.raises(ValueError, match="duplicate task id"):
        load_catalog(tmp_path)


@pytest.mark.skipif(
    os.environ.get("ROBOENV_TEST_UPSTREAMS") != "1", reason="selected upstream check"
)
def test_every_discovered_task_resolves_without_agent_framework(tmp_path):
    receipt = discover(ROOT, tmp_path / "source")
    catalog = load_catalog(tmp_path / "source/tasks")
    assert len(catalog) == receipt["tasks"]
    assert {task.backend for task, _ in catalog.values()} == set(BACKENDS)
    assert receipt["execution_verified"] is False
    for task, path in catalog.values():
        resolved = resolve_task(path, project=ROOT)
        assert resolved.task.id == task.id
        assert resolved.bindings.devices
        assert not path.with_name("solution.py").exists()


@pytest.mark.parametrize("native,slug", [
    ("PickPlaceCounterToMicrowave", "pick-place-counter-to-microwave"),
    ("RGBDCamera", "rgbd-camera"), ("stack_blocks_three", "stack-blocks-three"),
    ("10_3", "10-3"), ("space...and__punctuation", "space-and-punctuation"),
])
def test_native_names_become_kebab_case(native, slug):
    from roboenv.tasks.catalog import task_name

    assert task_name(native) == slug


def test_discovery_reuses_names_and_is_repeatable(tmp_path, monkeypatch):
    from dataclasses import replace

    from roboenv.tasks import catalog

    examples = load_catalog(ROOT / "catalog/tasks")
    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    for backend in BACKENDS:
        entries = [(replace(task, id=f"{backend}/native-{index}", instruction="Native text."),
                    ROOT / "README.md")
                   for index, (task, _) in enumerate(examples.values()) if task.backend == backend]
        monkeypatch.setitem(catalog.DISCOVERERS, backend, lambda *a, entries=entries: entries)
    destination = tmp_path / "discovered"
    first = discover(ROOT, destination)
    before = {p.relative_to(destination): p.read_bytes()
              for p in destination.rglob("*") if p.is_file()}
    assert discover(ROOT, destination) == first
    assert before == {p.relative_to(destination): p.read_bytes()
                      for p in destination.rglob('*') if p.is_file()}
    exported = load_catalog(destination / "tasks")
    assert exported.keys() == examples.keys()
    for key, (task, path) in exported.items():
        assert task == examples[key][0]
        assert path.read_text() == yaml.safe_dump(task.as_dict(), sort_keys=False)
        assert resolve_task(path, project=ROOT).bindings.devices


def test_discovery_rejects_ambiguous_catalog_identity_before_export(tmp_path, monkeypatch):
    from dataclasses import replace

    from roboenv.tasks import catalog

    task, path = next(iter(load_catalog(ROOT / "catalog/tasks").values()))
    duplicate = replace(task, id=f"{task.backend}/another-name")
    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    monkeypatch.setattr(catalog, "load_catalog", lambda _: {
        task.id: (task, path), duplicate.id: (duplicate, path),
    })
    destination = tmp_path / "discovered"
    with pytest.raises(ValueError, match="ambiguous catalog native task"):
        discover(ROOT, destination)
    assert not destination.exists()


def test_discovery_rejects_normalized_name_collision_before_export(tmp_path, monkeypatch):
    from roboenv.tasks import catalog

    monkeypatch.setattr(catalog, "load_lock", lambda _: (ROOT, {"sources": {}}))
    monkeypatch.setattr(catalog, "status", lambda *a: {"ready": True, "sources": []})
    monkeypatch.setitem(catalog.DISCOVERERS, "robocasa", lambda *a: [
        (catalog._task("robocasa", name, {"task_name": name}, "Test."), ROOT / "README.md")
        for name in ("PickPlace", "pick_place")
    ])
    destination = tmp_path / "discovered"
    with pytest.raises(ValueError, match="duplicate discovered task: robocasa/pick-place"):
        discover(ROOT, destination, ["robocasa"])
    assert not destination.exists()
