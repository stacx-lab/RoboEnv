"""Task bundles use native images only and never include references implicitly."""

import json
from pathlib import Path

import pytest
import yaml
from roboenv.harbor import Bundle, export_task
from roboenv.taskset import validate_harbor_task

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"test-only build input; never installed")
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "robocasa",
                "image": "sha256:" + "a" * 64,
                "legacy_tree_required": False,
                "mounts": [],
            }
        )
    )
    return ROOT / "catalog/tasks/robocasa-arrange-vegetables/task.yaml", deployment, wheel


def test_native_task_is_harbor_format_without_a_solution(tmp_path):
    task, deployment, wheel = inputs(tmp_path)
    out = tmp_path / "task"
    export_task(task, ROOT / "catalog", out, None, wheel, deployment_path=deployment)
    assert validate_harbor_task(out)["schema_version"] == "1.4"
    assert not (out / "solution").exists()
    assert (out / "tests/test.sh").stat().st_mode & 0o111
    compose = yaml.safe_load((out / "environment/docker-compose.yaml").read_text())
    assert compose["services"]["sim"]["image"] == "sha256:" + "a" * 64
    assert str(tmp_path) not in (out / "environment/docker-compose.yaml").read_text()
    before = (out / "bundle.json").read_bytes()
    export_task(task, ROOT / "catalog", out, None, wheel, deployment_path=deployment)
    assert before == (out / "bundle.json").read_bytes()
    assert json.loads(before)["certified"] is False


def test_host_source_tree_deployments_are_rejected(tmp_path):
    task, _, wheel = inputs(tmp_path)
    with pytest.raises(ValueError, match="native deployment"):
        export_task(task, ROOT / "catalog", tmp_path / "task", tmp_path, wheel)


def test_stale_reference_cannot_survive_regeneration(tmp_path):
    bundle = Bundle(tmp_path / "task")
    bundle.add("instruction.md", "test")
    bundle.add("solution/solve.sh", "test")
    bundle.commit({})
    replacement = Bundle(bundle.root)
    replacement.add("instruction.md", "test")
    with pytest.raises(ValueError, match="stale managed"):
        replacement.commit({})


def test_manifest_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    out = tmp_path / "task"
    out.mkdir()
    (out / "bundle.json").symlink_to(outside)
    bundle = Bundle(out)
    bundle.add("instruction.md", "test")
    with pytest.raises(ValueError, match="manifest"):
        bundle.commit({})
