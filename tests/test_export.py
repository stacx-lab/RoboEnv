"""Task bundles use native images only and never include references implicitly."""

import json
from pathlib import Path

import pytest
import tomllib
import yaml
from roboenv.harbor.export import export_task
from roboenv.harbor.taskset import validate_harbor_task
from roboenv.tasks.artifacts import Bundle

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
    return ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml", deployment, wheel


def test_native_task_is_harbor_format_without_a_solution(tmp_path):
    task, deployment, wheel = inputs(tmp_path)
    out = tmp_path / "task"
    export_task(task, ROOT / "catalog", out, wheel, deployment_path=deployment)
    assert validate_harbor_task(out)["schema_version"] == "1.4"
    assert (out / "environment/sim/robot/bindings.yaml").is_file()
    assert not list((out / "environment/definition").rglob("bindings.yaml"))
    assert not (out / "solution").exists()
    assert (out / "tests/test.sh").stat().st_mode & 0o111
    compose = yaml.safe_load((out / "environment/docker-compose.yaml").read_text())
    assert compose["services"]["sim"]["image"] == "sha256:" + "a" * 64
    config = tomllib.loads((out / "task.toml").read_text())
    assert config["agent"]["user"] == "agent"
    assert config["verifier"]["user"] == "root"
    main, sim = compose["services"]["main"], compose["services"]["sim"]
    assert main["environment"]["ROBOENV_ENDPOINT"] == "http://sim:8123"
    assert main["depends_on"]["sim"]["condition"] == "service_healthy"
    assert sim["environment"]["ROBOENV_HTTP_HOST"] == "0.0.0.0"
    assert sim["environment"]["ROBOENV_FRESH_SESSION"] == "1"
    assert sim["environment"]["ROBOENV_SERVER_ROBOT"] == "/opt/serve/robot"
    assert sim["environment"]["ROBOENV_SOCKET"] == "/run/roboenv/sim.sock"
    assert sim["environment"]["ROBOENV_PUBLISH_DEVICES"]
    assert "ports" not in sim
    assert main["cap_drop"] == ["ALL"]
    assert main["security_opt"] == ["no-new-privileges:true"]
    assert not (out / "environment/definition/robot/devices").exists()
    assert (out / "environment/sim/environment.json").is_file()
    assert list((out / "environment/sim/robot/devices").glob("*/driver.py"))
    assert str(tmp_path) not in (out / "environment/docker-compose.yaml").read_text()
    before = (out / "bundle.json").read_bytes()
    export_task(task, ROOT / "catalog", out, wheel, deployment_path=deployment)
    assert before == (out / "bundle.json").read_bytes()
    assert json.loads(before)["certified"] is False


def test_export_requires_a_native_deployment(tmp_path):
    task, _, wheel = inputs(tmp_path)
    with pytest.raises(ValueError, match="native deployment"):
        export_task(task, ROOT / "catalog", tmp_path / "task", wheel)


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
