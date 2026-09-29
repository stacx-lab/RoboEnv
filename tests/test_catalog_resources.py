"""Public catalogs resolve private assembly defaults from installed resources."""

import shutil
from pathlib import Path

import pytest
import yaml
from helpers import load_tool
from roboenv.devices.bindings.spec import BindingPlan, binding_resource
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"


def test_public_catalog_contains_only_robot_and_task_definitions():
    assert {p.name for p in CATALOG.iterdir()} == {"robots", "tasks"}
    assert not list(CATALOG.rglob("bindings.yaml"))


@pytest.mark.parametrize("path", sorted((CATALOG / "tasks").rglob("task.yaml")))
def test_tasks_resolve_from_a_copied_public_catalog(tmp_path, path):
    original = resolve_task(path, catalog=CATALOG)
    copied = tmp_path / "catalog"
    shutil.copytree(original.robot, copied / "robots" / original.task.robot)
    task = copied / path.relative_to(CATALOG)
    task.parent.mkdir(parents=True)
    shutil.copy2(path, task)
    resolved = resolve_task(task, catalog=copied)
    assert resolved.bindings == original.bindings
    assert resolved.task.execution_mode == "realtime"
    registry = default_registry()
    assert "world" in {spec.name for spec in resolved.bindings.devices}
    for spec in resolved.bindings.devices:
        assert spec.binding in registry._factories
        assert spec.definition(resolved.robot).name == spec.name
    assert not (copied / "bindings").exists()


def test_owner_bundle_and_explicit_bindings_override_packaged_defaults(tmp_path):
    source = CATALOG / "robots/libero/libero-panda-kitchen"
    robot = tmp_path / source.name
    shutil.copytree(source, robot)
    default = BindingPlan.load(robot, backend="libero")
    arm = next(spec for spec in default.devices if spec.name == "arm")
    bundle = {**default.as_dict(), "devices": {"arm": arm.as_dict()}}
    (robot / "bindings.yaml").write_text(yaml.safe_dump(bundle))
    assert len(BindingPlan.load(robot, backend="libero").devices) == 1
    with pytest.raises(ValueError, match="backend mismatch"):
        BindingPlan.load(robot, backend="robocasa")
    (robot / "bindings.yaml").write_text("invalid bundle")
    explicit = BindingPlan.load(robot, backend="libero", devices=default.as_dict()["devices"])
    assert explicit.devices == default.devices
    with pytest.raises(ValueError, match="malformed binding plan"):
        BindingPlan.load(robot, backend="libero")


def test_missing_or_unsafe_defaults_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="No default bindings"):
        BindingPlan.load(tmp_path / "unknown-robot", backend="libero")
    with pytest.raises(ValueError, match="invalid backend"):
        binding_resource(tmp_path / "robot", backend="../libero")


def test_native_generator_separates_public_catalog_and_internal_bindings(tmp_path):
    generator = load_tool("generate_native_rigs")
    catalog = tmp_path / "catalog"
    facts = {"world": {"type": "world", "intrinsics": {}}}
    generator.write_rig(catalog, "test-robot", "test", "test", facts)
    assert {p.name for p in catalog.iterdir()} == {"robots"}
    bindings = tmp_path / "defaults"
    generator.write_rig(catalog, "test-robot", "test", "test", facts, bindings_out=bindings)
    plan = yaml.safe_load((bindings / "test/test-robot.yaml").read_text())
    assert plan["devices"]["world"] == {"binding": "test.device/1", "entity": "world"}
    assert {p.name for p in catalog.iterdir()} == {"robots"}


def test_all_robot_variants_have_registered_bindings():
    registry = default_registry()
    for workspace in sorted((CATALOG / "robots").rglob("workspace.yaml")):
        robot = workspace.parent
        for spec in BindingPlan.load(robot, backend=robot.parent.name).devices:
            assert spec.binding in registry._factories
            assert spec.definition(robot).name == spec.name
