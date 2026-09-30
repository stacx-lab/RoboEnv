"""Public catalogs resolve private assembly defaults from installed resources."""

import runpy
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


def test_go2_arm_driver_converts_native_frame_poses():
    driver = CATALOG / "robots/isaaclab/isaaclab-go2-d1/devices/arm/driver.py"
    normalize = runpy.run_path(str(driver))["kinex_reading"]
    state = {"positions": [0.0], "frame_poses": {"flange": {
        "translation_m": [1.0, 2.0, 3.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }}}
    assert normalize(state)["frame_poses"]["flange"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    assert state["frame_poses"]["flange"]["translation_m"] == [1.0, 2.0, 3.0]
    with pytest.raises(KeyError, match="rotation_xyzw"):
        normalize({"frame_poses": {"flange": {"translation_m": [1.0, 2.0, 3.0]}}})


def test_public_catalog_contains_only_robot_and_task_definitions():
    assert {p.name for p in CATALOG.iterdir()} == {"robots", "tasks"}
    assert not list(CATALOG.rglob("bindings.yaml"))


def test_tasks_resolve_from_a_copied_public_catalog(tmp_path):
    copied = tmp_path / "catalog"
    shutil.copytree(CATALOG, copied)
    for path in sorted((CATALOG / "tasks").rglob("task.yaml")):
        original = resolve_task(path, catalog=CATALOG)
        resolved = resolve_task(copied / path.relative_to(CATALOG), catalog=copied)
        assert resolved.bindings == original.bindings
        assert resolved.task.execution_mode == "realtime"
        assert "world" in {spec.name for spec in resolved.bindings.devices}
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


def test_copied_go2_catalog_keeps_bindings_private_and_drivers_portable(tmp_path):
    from roboenv.tasks.catalog import load_catalog

    source = CATALOG / "robots/isaaclab/isaaclab-go2-d1"
    catalog = tmp_path / "catalog"
    rig = catalog / "robots/isaaclab/isaaclab-go2-d1"
    shutil.copytree(source, rig)
    shutil.copytree(CATALOG / "tasks/isaaclab", catalog / "tasks/isaaclab")
    tasks = load_catalog(catalog / "tasks")
    assert set(tasks) == {"isaaclab/go2-d1-control", "isaaclab/go2-d1-reach",
                          "isaaclab/go2-d1-pick-place"}
    for task, _ in tasks.values():
        assert task.execution_mode == "realtime"
        assert "realtime" not in task.backend_kwargs
    assert list(rig.rglob("_roboenv.py")) == [rig / "devices/_roboenv.py"]
    base_control = yaml.safe_load((rig / "devices/base/profile.yaml").read_text())[
        "intrinsics"
    ]["control"]
    arm_control = yaml.safe_load((rig / "devices/arm/profile.yaml").read_text())[
        "intrinsics"
    ]["control"]
    assert "execution" not in base_control
    assert arm_control == {"mode": "position", "limit_handling": "native"}
    plan = BindingPlan.load(rig, backend="isaaclab")
    assert {d.name for d in plan.devices} == {
        "base", "arm", "gripper", "camera_head", "camera_wrist", "world"
    }
    assert binding_resource(rig, backend="isaaclab").is_file()
    assert not (catalog / "bindings").exists()
    rig.rename(rig.with_name("unknown-go2"))
    with pytest.raises(ValueError, match="No default bindings"):
        BindingPlan.load(rig.with_name("unknown-go2"), backend="isaaclab")
