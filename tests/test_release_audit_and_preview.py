"""Core audit and private snapshot scope cannot silently broaden or skip failures."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packages():
    return {
        "roboenv": {
            "name": "roboenv",
            "version": "0.2.0",
            "requires": ['validation>=1; extra == "server"'],
        },
        "roboenv-runtime": {
            "name": "roboenv-runtime",
            "version": "0.2.0",
            "requires": ["roboenv[server]==0.2.0"],
        },
        "validation": {"name": "validation", "version": "1.0", "requires": []},
    }


def test_core_closure_includes_server_extra_and_rejects_dev_environment(monkeypatch):
    audit = load_tool("audit_core_release", monkeypatch)
    installed = packages()
    assert audit.dependency_closure(installed) == ["validation"]
    installed["unrelated-tool"] = {"name": "unrelated-tool", "version": "1", "requires": []}
    with pytest.raises(ValueError, match="non_core"):
        audit.dependency_closure(installed)


def test_core_closure_fails_on_missing_or_incompatible_dependency(monkeypatch):
    audit = load_tool("audit_core_release", monkeypatch)
    installed = packages()
    installed["validation"]["version"] = "0.1"
    with pytest.raises(ValueError, match="version_mismatch"):
        audit.dependency_closure(installed)
    del installed["validation"]
    with pytest.raises(ValueError, match="unresolved"):
        audit.dependency_closure(installed)


def test_sbom_replaces_only_local_candidate_locations_with_hashes(monkeypatch):
    audit = load_tool("audit_core_release", monkeypatch)
    bom = {
        "components": [
            {
                "name": "roboenv",
                "externalReferences": [
                    {"type": "distribution", "url": "file:" + "/tmp/private-wheel.whl"},
                    {"type": "website", "url": "https://example.org/project"},
                ],
            }
        ],
    }
    result = audit.sanitize_bom(
        bom,
        {
            "roboenv": {"name": "roboenv-0.2.0-py3-none-any.whl", "sha256": "a" * 64},
        },
    )
    component = result["components"][0]
    assert component["hashes"] == [{"alg": "SHA-256", "content": "a" * 64}]
    assert component["externalReferences"] == [
        {"type": "website", "url": "https://example.org/project"},
    ]
    assert "private-wheel" not in json.dumps(result)


def test_advisory_collection_failure_cannot_pass(monkeypatch, tmp_path):
    audit = load_tool("audit_core_release", monkeypatch)
    monkeypatch.setattr(
        audit,
        "run",
        lambda *args, **kwargs: type(
            "Failed",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "",
            },
        )(),
    )
    with pytest.raises(ValueError, match="advisory_query_failed"):
        audit.audit(packages(), ["validation"], {}, Path(sys.executable), tmp_path, tmp_path)


def test_snapshot_excludes_history_generated_tasks_and_private_work(monkeypatch, tmp_path):
    snapshot = load_tool("prepare_source_preview", monkeypatch)
    (tmp_path / "README.md").write_text("# Example\n")
    (tmp_path / "AGENTS.md").write_text("# Working here\n")
    (tmp_path / "upstreams.lock.json").write_text(
        json.dumps(
            {
                "sources": {
                    "engine": {
                        "path": "third_party/engine",
                        "revision": "a" * 40,
                        "url": "https://github.com/example/engine.git",
                    }
                },
            }
        )
    )
    for private in (
        ".git/HEAD",
        ".cache/note.txt",
        "generated/task/task.yaml",
        "benchmarks/manipulation/tasks/old/task.yaml",
        "deployments/local/map.json",
    ):
        path = tmp_path / private
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not part of the snapshot")
    inventory = snapshot.manifest(tmp_path)
    assert {item["path"] for item in inventory["files"]} == {
        "README.md",
        "AGENTS.md",
        "upstreams.lock.json",
    }
    assert inventory["destination"]["visibility"] == "private"
    assert inventory["core_license_declared"] is False
    assert inventory["history_included"] is False
    assert len(inventory["gitlinks"]) == 1


def test_snapshot_refuses_unexpected_files_and_existing_destination(monkeypatch, tmp_path):
    snapshot = load_tool("prepare_source_preview", monkeypatch)
    (tmp_path / "scratch-note.txt").write_text("not approved source")
    with pytest.raises(ValueError, match="unexpected"):
        snapshot.manifest(tmp_path)
    monkeypatch.setattr(
        snapshot,
        "inspect",
        lambda *args, **kwargs: {
            "private_source_preview_ready": True,
        },
    )
    monkeypatch.setattr(snapshot, "manifest", lambda root: {"files": []})
    with pytest.raises(ValueError, match="already_exists"):
        snapshot.prepare(tmp_path, tmp_path)


def test_source_preview_permission_is_not_package_release_approval(monkeypatch):
    preflight = load_tool("release_preflight", monkeypatch)
    monkeypatch.setattr(preflight, "check_sources", lambda *args: None)
    result = preflight.inspect(ROOT, mode="source-preview")
    assert result["passed"]
    assert result["private_source_preview_ready"]
    assert not result["packaging_ready"]
    assert not preflight.inspect(ROOT, mode="release")["passed"]


def test_source_preview_still_requires_upstream_verification(monkeypatch):
    preflight = load_tool("release_preflight", monkeypatch)
    result = preflight.inspect(ROOT, mode="source-preview")
    assert not result["passed"]
    assert not result["private_source_preview_ready"]
