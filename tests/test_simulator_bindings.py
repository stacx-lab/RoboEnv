"""Explicit legacy controller compatibility; default native bindings are tested separately."""

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.model import DeviceError
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.bindings.backend import BackendControllers
from roboenv_runtime.bindings.simulators import binding_index
from roboenv_runtime.runtime.worker import SimWorker
from service_fixtures import legacy_registry

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = sorted((ROOT / "tests/fixtures/legacy-native-catalogs").iterdir())


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda p: p.name)
def test_every_catalog_slot_has_a_registered_non_catalog_implementation(robot):
    value = yaml.safe_load((robot / "bindings.yaml").read_text())
    backend, embodiment = value["backend"], value["embodiment"]
    plan = BindingPlan.load(robot, backend=backend)
    controllers = {}
    for spec in plan.devices:
        assert spec.binding == backend + ".device/1"
        record = binding_index()[f"{backend}:{embodiment}:{spec.entity}"]
        module = importlib.import_module("roboenv_runtime.bindings.backends." + record["module"])
        for dependencies in module.REQUIRES.values():
            controllers.update({name: lambda: None for name in dependencies})
    context = BindingContext(backend, embodiment, object(), public_tools=controllers)
    service = legacy_registry().assemble(plan, robot, context)
    try:
        for spec in plan.devices:
            description = service.describe(spec.name)
            assert description["implementation"] == "backend-controller-binding"
            for action in ("stop", "ik", "fk", "unload", "policy_run"):
                assert not description["actions"].get(action, {}).get("supported", False)
        assert not list((robot / "devices").glob("*/driver.py"))
    finally:
        service.close()


def robodojo_service(controllers):
    robot = ROOT / "tests/fixtures/legacy-native-catalogs/robodojo-arx-x5"
    plan = BindingPlan.load(robot, backend="robodojo")
    context = BindingContext("robodojo", "arx_x5", object(), public_tools=controllers)
    return legacy_registry().assemble(plan, robot, context)


def test_missing_controller_is_reported_before_execution():
    service = robodojo_service({})
    try:
        assert service.describe("arm")["actions"]["write"]["supported"] is False
        with pytest.raises(DeviceError, match="unavailable"):
            service.call({"device": "arm", "action": "write", "args": {"target": [0, 0, 0]}})
    finally:
        service.close()


def test_cartesian_write_converts_xyzw_and_preserves_counted_controller_calls():
    calls = []
    service = robodojo_service(
        {
            "goto_pose": lambda xyz, quat: calls.append((xyz, quat)),
            "get_gripper_pose": lambda: [0.2, 0.1, 0.3],
        }
    )
    try:
        result = service.call(
            {"device": "arm", "action": "write", "args": {"target": [0.2, 0.1, 0.3, 0, 0, 0, 1]}}
        )
        assert calls == [([0.2, 0.1, 0.3], [1, 0, 0, 0])]
        assert [entry["args"]["name"] for entry in result["calls"]] == [
            "goto_pose", "get_gripper_pose",
        ]
        assert all(entry["op"] == "binding.controller" for entry in result["calls"])
        with pytest.raises(DeviceError):
            service.call({"device": "arm", "action": "write", "args": {"target": [0, 0]}})
        assert len(calls) == 1
    finally:
        service.close()


def test_controllers_cannot_execute_during_binding_construction():
    controller = BackendControllers(
        BindingContext("test", "test", object(), public_tools={"move": lambda: None})
    )
    with pytest.raises(DeviceError, match="setup"):
        controller.call("move")


def test_legacy_device_accounting_does_not_enable_public_primitives(tmp_path, monkeypatch):
    monkeypatch.setattr("roboenv_runtime.bindings.assembly.default_registry", legacy_registry)

    class ArmWorld(FakeShell):
        name = "robodojo"
        embodiment = "arx_x5"

        def tools(self):
            return {"goto_pose": self.goto_pose, "get_gripper_pose": lambda: list(self.pos)}

        def goto_pose(self, xyz, quat=None):
            return self.push(dx=xyz[0] - self.pos[0])

    worker = SimWorker(
        ArmWorld(),
        task_id="test",
        seed=0,
        run_dir=tmp_path,
        device_root=ROOT / "tests/fixtures/legacy-native-catalogs/robodojo-arx-x5",
    )
    try:
        worker.dispatch(
            "devices.call", {"device": "arm", "action": "write", "args": {"target": [0.1, 0, 0]}}
        )
        assert worker.basic_call_count == 2  # motion plus pose query, each counted once
        from roboenv_runtime.runtime.worker import WorkerRPCError

        with pytest.raises(WorkerRPCError, match="basic device"):
            worker.dispatch("call", {"name": "goto_pose", "kwargs": {"xyz": [0.2, 0, 0]}})
        assert worker.basic_call_count == 2
        assert worker.adapter.budget().used == 50
    finally:
        worker.close()


def piper_world_service(*, functional_controller=True, failed_pose=False):
    root = ROOT / "catalog/robots/robotwin/robotwin-dual-piper"
    plan = BindingPlan.load(root, backend="robotwin")
    plan = replace(plan, devices=tuple(spec for spec in plan.devices if spec.name == "world"))
    functional_calls = []

    def pose(name):
        if failed_pose:
            raise RuntimeError("pose backend unavailable")
        return [0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0]

    def functional(name, point):
        functional_calls.append(name)
        if name == "table":
            raise AttributeError("plain scene entity has no functional point")
        return [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]

    controllers = {
        "list_objects": lambda: ["microphone", "table"],
        "get_object_pose": pose,
        "get_gripper_pose_left": lambda: [0.0, 0.0, 0.8],
        "get_gripper_pose_right": lambda: [0.0, 0.0, 0.8],
    }
    if functional_controller:
        controllers["get_object_functional_pose"] = functional
    observations = {
        "observe": {"budget": {"used": 0, "max": 60_000, "unit": "api_calls"}},
        "score.peek": {"reward": 0.0, "submitted": False},
    }
    context = BindingContext(
        "robotwin",
        "piper",
        object(),
        public_dispatch=lambda op, args: observations[op],
        public_tools=controllers,
    )
    # Retain regression coverage for the independently retained primitive handler.
    # The default RoboTwin registry now uses a sensor-only native world.
    from roboenv.devices.bindings.registry import BindingRegistry
    from roboenv_runtime.bindings.simulators import bind_simulator_device

    registry = BindingRegistry()
    registry.register("robotwin.device/1", bind_simulator_device)
    return registry.assemble(plan, root, context), functional_calls


@pytest.mark.parametrize("functional_controller", [False, True])
def test_piper_world_read_does_not_assume_every_scene_entity_has_functional_points(
    functional_controller,
):
    service, functional_calls = piper_world_service(functional_controller=functional_controller)
    try:
        assert service.describe("world")["actions"]["read"]["supported"]
        for action in ("read", "state"):
            response = service.call({"device": "world", "action": action, "args": {}})
            assert response["read_only"]
            assert set(response["data"]["objects"]) == {"microphone", "table"}
            assert all(entry["ok"] for entry in response["calls"])
        assert functional_calls == []
    finally:
        service.close()


def test_piper_world_read_still_propagates_real_pose_failures():
    service, _ = piper_world_service(failed_pose=True)
    try:
        with pytest.raises(DeviceError, match="pose backend unavailable") as error:
            service.call({"device": "world", "action": "read", "args": {}})
        assert any(entry.get("ok") is False for entry in error.value.details["calls"])
    finally:
        service.close()
