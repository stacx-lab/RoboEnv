"""Opt-in Cartesian arm proposal: real binding/service path, fake physics only."""

import ast
import math
from pathlib import Path

import pytest
import yaml
from roboenv.bindings.registry import BindingContext
from roboenv.bindings.spec import BindingPlan
from roboenv.device_model import DeviceError, ExecutionContext
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.bindings import default_registry
from roboenv_runtime.bindings.arm_ports import PORTS, PROPOSAL_BINDING, bind_cartesian_arm
from roboenv_runtime.drivers.arm import ArmDriver, Pose
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError

ROOT = Path(__file__).resolve().parents[1]


def definition(tmp_path):
    root = tmp_path / "robot"
    profile = root / "devices/arm/profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        yaml.safe_dump(
            {
                "name": "arm",
                "type": "manipulator",
                "access": {
                    "control_mode": "cartesian_pose",
                    "driver": {"actions": ["read", "state", "write", "displace"]},
                },
            }
        )
    )
    return root


def plan(root, backend, entity, *, config=None):
    return BindingPlan.load(
        root,
        backend=backend,
        devices={
            "arm": {
                "binding": PROPOSAL_BINDING,
                "entity": entity,
                "config": config or {},
            }
        },
    )


def setup_service(
    tmp_path,
    backend="robotwin",
    embodiment="piper",
    entity="left_arm",
    *,
    missing=(),
    motion_error=False,
    undershoot=False,
):
    root = definition(tmp_path)
    selected = plan(root, backend, entity)
    state = {"position": [0.1, 0.2, 0.3], "wxyz": [1.0, 0.0, 0.0, 0.0]}
    calls = []
    side = entity.removesuffix("_arm")

    def read(*args):
        calls.append(("read", args))
        if backend == "robodojo":
            assert args == (side,)
        else:
            assert args == ()
        if backend == "libero":
            # TCP is deliberately different from wrist and fingertips.
            return {
                "tcp": state["position"],
                "quat_wxyz": state["wxyz"],
                "wrist": [9, 9, 9],
                "fingertips": [8, 8, 8],
            }
        return state["position"] + state["wxyz"]

    def move(*args):
        calls.append(("move", args))
        if backend == "robodojo":
            assert args[0] == side
            args = args[1:]
        assert len(args) == 2
        position, orientation = args
        if not undershoot:
            state.update(position=list(position), wxyz=list(orientation))
        if motion_error:
            raise RuntimeError("controller failed after movement")
        return {"backend": backend}

    read_name = "get_hand_pose" if backend == "libero" else "get_gripper_pose"
    move_name = "goto_pose"
    if backend == "robotwin":
        read_name += "_" + side
        move_name += "_" + side
    tools = {read_name: read, move_name: move}
    tools = {name: value for name, value in tools.items() if name not in missing}
    context = BindingContext(backend, embodiment, object(), public_tools=tools)
    service = default_registry().assemble(selected, root, context)
    return service, state, calls, context, selected, root


def call(service, action, args=None):
    return service.call({"device": "arm", "action": action, "args": args or {}})


@pytest.mark.parametrize("backend,embodiment,entity", sorted(PORTS))
def test_same_driver_contract_covers_every_explicit_arm_selector(
    tmp_path, backend, embodiment, entity
):
    service, state, calls, *_ = setup_service(tmp_path, backend, embodiment, entity)
    with_service = service.describe("arm")
    assert with_service["implementation"] == "shared-cartesian-arm-proposal"
    frame = "robot_base" if backend == "libero" else "world"
    observed = call(service, "read")
    assert observed["read_only"] is True
    assert observed["data"]["position_m"] == [0.1, 0.2, 0.3]
    assert observed["data"]["frame"] == frame
    calls.clear()
    result = call(service, "write", {"frame": frame, "target": [0.4, 0.5, 0.6, 0, 0, 1, 0]})
    assert state["wxyz"] == [0, 0, 0, 1]
    assert state["position"] == [0.4, 0.5, 0.6]
    assert [entry[0] for entry in calls] == ["move", "read"]
    assert len(result["calls"]) == 2
    assert result["data"]["target_reached"] is True
    assert result["data"]["observed"]["quaternion_xyzw"] == [0, 0, 1, 0]
    assert result["data"]["backend_report"] == {"backend": backend}
    service.close()
    assert state["position"] == [0.4, 0.5, 0.6]


def test_xyz_preserves_orientation_and_displacement_is_shared_logic(tmp_path):
    service, state, calls, *_ = setup_service(tmp_path)
    state["wxyz"] = [0, 0, 1, 0]
    call(service, "write", {"frame": "world", "target": [0.2, 0.3, 0.4]})
    assert state["wxyz"] == [0, 0, 1, 0]
    assert [entry[0] for entry in calls] == ["read", "move", "read"]
    calls.clear()
    result = call(service, "displace", {"frame": "world", "delta": [0.1, -0.1, 0]})
    assert state["position"] == pytest.approx([0.3, 0.2, 0.4])
    assert [entry[0] for entry in calls] == ["read", "move", "read"]
    assert len(result["calls"]) == 3


@pytest.mark.parametrize(
    "args",
    [
        {"frame": "robot_base", "target": [0, 0, 0]},
        {"frame": "world", "target": [0, 0]},
        {"frame": "world", "target": [0, 0, 0, 0]},
        {"frame": "world", "target": [0, 0, 0, 0, 0, 0, 0]},
        {"frame": "world", "target": [0, 0, 0, 1, 1, 1, 1]},
        {"frame": "world", "target": [True, 0, 0]},
        {"frame": "world", "target": ["0", 0, 0]},
        {"frame": "world", "target": [math.nan, 0, 0]},
        {"frame": "world", "target": [math.inf, 0, 0]},
        {"frame": "world", "target": [0, 0, 0], "planner": "replacement"},
        {"target": [0, 0, 0]},
    ],
)
def test_invalid_requests_have_no_backend_calls(tmp_path, args):
    service, _, calls, *_ = setup_service(tmp_path)
    with pytest.raises(DeviceError):
        call(service, "write", args)
    assert calls == []


def test_motion_failure_is_not_retried_and_keeps_unknown_effects(tmp_path):
    service, state, calls, *_ = setup_service(tmp_path, motion_error=True)
    with pytest.raises(DeviceError) as caught:
        call(service, "write", {"frame": "world", "target": [0.4, 0.5, 0.6, 0, 0, 0, 1]})
    assert state["position"] == [0.4, 0.5, 0.6]
    assert [entry[0] for entry in calls] == ["move"]
    assert caught.value.details["effects"] == "unknown"
    assert caught.value.details["calls"][0]["ok"] is False


def test_undershoot_is_measured_not_reported_as_target_reached(tmp_path):
    service, _, calls, *_ = setup_service(tmp_path, undershoot=True)
    result = call(service, "write", {"frame": "world", "target": [0.9, 0.8, 0.7, 0, 0, 0, 1]})
    assert result["data"]["target_reached"] is False
    assert result["data"]["position_error_m"] > 0.1
    assert [entry[0] for entry in calls] == ["move", "read"]


def test_missing_motion_controller_leaves_read_available(tmp_path):
    service, _, calls, *_ = setup_service(tmp_path, missing=("goto_pose_left",))
    descriptor = service.describe("arm")
    assert descriptor["actions"]["read"]["supported"] is True
    assert descriptor["actions"]["write"]["supported"] is False
    with pytest.raises(DeviceError, match="unavailable"):
        call(service, "write", {"frame": "world", "target": [0, 0, 0]})
    assert calls == []


def test_readonly_context_cannot_drive_motion(tmp_path):
    service, _, calls, *_ = setup_service(tmp_path)
    device = service._devices["arm"]
    with pytest.raises(DeviceError, match="mutating"):
        device.execute(
            "write",
            {"frame": "world", "target": [0, 0, 0, 0, 0, 0, 1]},
            ExecutionContext(read_only=True),
        )
    assert calls == []


@pytest.mark.parametrize(
    "backend,embodiment,entity",
    [
        ("behavior", "r1pro", "arm"),
        ("robotwin", "unknown", "left_arm"),
        ("robodojo", "arx_x5", "arm"),
        ("robotwin", "piper", "robot.left.__dict__"),
    ],
)
def test_unknown_symbolic_and_ambiguous_entities_are_rejected(
    tmp_path, backend, embodiment, entity
):
    root = definition(tmp_path)
    selected = plan(root, backend, entity)
    with pytest.raises(ValueError, match="no reviewed"):
        default_registry().assemble(
            selected, root, BindingContext(backend, embodiment, object(), public_tools={})
        )


def test_unknown_config_and_invalid_tolerances_are_rejected(tmp_path):
    _, _, _, context, _, root = setup_service(tmp_path)
    for config in (
        {"planner": "anything"},
        {"position_tolerance_m": 0},
        {"orientation_tolerance_rad": True},
    ):
        selected = plan(root, "robotwin", "left_arm", config=config)
        with pytest.raises(ValueError):
            bind_cartesian_arm(selected.devices[0], root, context)


def test_driver_has_no_backend_imports_and_default_catalog_is_unchanged():
    tree = ast.parse((ROOT / "runtime/src/roboenv_runtime/drivers/arm.py").read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("bindings" in name or "adapters" in name for name in imports)
    for path in (ROOT / "catalog/robots").glob("*/bindings.yaml"):
        assert PROPOSAL_BINDING not in path.read_text()


def test_quaternion_double_cover_does_not_change_rotation():
    class Port:
        def move_pose(self, pose, execution):
            self.pose = Pose(pose.position, tuple(-v for v in pose.quaternion_xyzw))

        def read_pose(self, execution):
            return self.pose

    from roboenv_runtime.drivers.arm import ArmSemantics

    port = Port()
    port.semantics = ArmSemantics("world", "tool", "tool", "test-only")
    result = ArmDriver(port).write(
        {"frame": "world", "target": [0, 0, 0, 0, 0, 0, 1]}, ExecutionContext(read_only=False)
    )
    assert result["orientation_error_rad"] == 0
    assert result["target_reached"] is True


def test_proposal_uses_existing_worker_budget_and_rebind_lifecycle(tmp_path):
    class ArmWorld(FakeShell):
        name, embodiment = "robotwin", "piper"

        def tools(self):
            return {"get_gripper_pose_left": self.read_pose, "goto_pose_left": self.goto_pose}

        def read_pose(self):
            return [self.pos[0], 0, 0, 1, 0, 0, 0]

        def goto_pose(self, xyz, quat):
            return self.push(dx=xyz[0] - self.pos[0])

    root = definition(tmp_path)
    selected = plan(root, "robotwin", "left_arm")
    (root / "bindings.yaml").write_text(yaml.safe_dump(selected.as_dict()))
    worker = SimWorker(
        ArmWorld(), task_id="test", seed=0, run_dir=tmp_path / "run", device_root=root
    )
    command = {
        "device": "arm",
        "action": "write",
        "args": {"frame": "world", "target": [0.1, 0, 0, 0, 0, 0, 1]},
    }
    try:
        worker.dispatch("devices.call", command)
        assert worker.basic_call_count == 2
        assert worker.adapter.budget().used == 50
        old = worker.devices
        worker.dispatch("reset", {"seed": 1})
        assert old is not worker.devices
        assert worker.adapter.pos[0] == 0
        with pytest.raises(DeviceError, match="closed"):
            old.call(command)
        worker.dispatch("devices.call", command)
        worker.dispatch("submit", {})
        with pytest.raises(WorkerRPCError, match="frozen"):
            worker.dispatch("devices.call", command)
    finally:
        worker.close()
