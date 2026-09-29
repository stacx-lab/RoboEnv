"""Native limits remain in the simulator; grippers use explicit profile bounds."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_exported_actuators_delegate_limits_and_keep_only_known_ratings():
    count = 0
    for path in (ROOT / "catalog/robots").rglob("profile.yaml"):
        if "fakeshell" in path.parts:
            continue
        profile = yaml.safe_load(path.read_text())
        control = profile["intrinsics"].get("control")
        if control is None:
            continue
        if profile["type"] == "gripper":
            assert "limit_handling" not in control, path
            if control["mode"] == "position":
                assert control["bounds"] == {"min": 0, "max": 1}, path
        else:
            assert control["limit_handling"] == "native", path
        count += 1
        if any(backend in path.parts for backend in ("robotwin", "behavior")):
            for joint in profile["intrinsics"].get("model", {}).get("joints", []):
                assert joint["max_velocity"] is None
                assert joint["max_effort"] is None
    assert count > 30


GRIPPERS = sorted((ROOT / "catalog/robots").glob("*/*/devices/*gripper/profile.yaml"))


@pytest.mark.parametrize("path", GRIPPERS, ids=lambda p: "/".join(p.parts[-5:-1]))
def test_gripper_profile_bounds_are_enforced_before_any_effect(path, monkeypatch):
    from concurrent.futures import Future

    from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
    from roboenv.devices.service import DeviceService
    from roboenv_runtime.bindings.motion import control_binding

    profile = yaml.safe_load(path.read_text())
    control = profile["intrinsics"]["control"]
    low = control["bounds"]["min"] if control["mode"] == "position" else -1
    high = control["bounds"]["max"] if control["mode"] == "position" else 1
    submitted = []

    @control_binding
    def bind(spec, root, context):
        return BoundDevice(
            DeviceDefinition("gripper", "gripper", profile["intrinsics"]),
            {
                "read": Action(lambda args, ctx: {"faults": []}),
                "write": Action(lambda args, ctx: submitted.append(args["target"])),
                "stop": Action(lambda args, ctx: {"stopped": True}),
            },
            implementation="gripper-limits-cpu",
        )

    def schedule(iterator, cancel):
        list(iterator)
        result = Future()
        result.set_result(None)
        return result

    device = bind(None, None, SimpleNamespace(native=SimpleNamespace(
        control_interval_s=.02, budget=lambda: SimpleNamespace(exhausted=False),
    )))
    service = DeviceService([device], motion_scheduler=schedule)

    def call(actions):
        return service.call({"device": "gripper", "action": "write",
                             "args": {"actions": actions, "duration_s": .06}})

    call([low, .5, high])
    expected = [low, .5, high] if control["mode"] == "position" else [-1, 1, 1]
    assert submitted == [[value] for value in expected]
    for value in (low - 1e-6, high + 1e-6, True, float("nan"), float("inf"), [0]):
        before = list(submitted)
        with pytest.raises(DeviceError):
            call([.5, value])
        assert submitted == before

    library = ROOT.parent / "kinex/packages/core/assets/python"
    if library.is_dir():
        monkeypatch.syspath_prepend(str(library))
        from kinex.scalar_control import command

        assert command(profile, {"actions": [low, .5, high], "duration_s": .06}) == {
            "actions": expected, "duration_s": .06,
        }
        for value in (low - 1e-6, high + 1e-6, True, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                command(profile, {"actions": [.5, value], "duration_s": .06})


def native_robot_class():
    # Execute the pinned upstream methods without importing its planners or GPU stack.
    import ast

    import numpy as np

    source = ROOT / "third_party/RoboTwin/envs/robot/robot.py"
    if not source.is_file():
        pytest.skip("requires the pinned RoboTwin checkout")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Robot")
    methods = {"set_arm_joints", "set_gripper", "get_normal_real_gripper_val"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Robot"]


@pytest.mark.parametrize("fraction,expected", [(-5, 0), (.4, .4), (5, 1)])
def test_robotwin_uses_native_gripper_limits_each_physics_substep(fraction, expected):
    pytest.importorskip("sapien")
    from roboenv_runtime.adapters.robotwin.devices import RoboTwinDevices

    class Joint:
        def __init__(self):
            self.drive_target = .01
            self.velocity = 0
            self.history = []

        def get_drive_target(self):
            return [self.drive_target]

        def set_drive_target(self, value):
            self.drive_target = value

        def set_drive_velocity_target(self, value):
            self.velocity = value
            self.history.append((self.drive_target, value))

    robot = native_robot_class()()
    robot._entity_qf = lambda _: None
    for side in ("left", "right"):
        setattr(robot, side + "_entity", object())
        setattr(robot, side + "_arm_joints", [Joint()])
        setattr(robot, side + "_gripper", [(Joint(), 1, 0)])
        setattr(robot, side + "_gripper_scale", (.01, .05))
    env = RoboTwinDevices.__new__(RoboTwinDevices)
    env.task = SimpleNamespace(robot=robot, scene=SimpleNamespace(step=lambda: None),
                               check_success=lambda: False)
    goals = {"left_arm": [100], "right_arm": [-100],
             "left_gripper": [fraction], "right_gripper": [fraction]}
    env.control = SimpleNamespace(targets=goals, frame=0)
    env.gripper_holds = {}
    env._sim_step_count, env._native_success = 0, False
    env._apply_targets(goals)
    left, right = robot.left_gripper[0][0], robot.right_gripper[0][0]
    assert left.history == right.history == []  # staging or a peer stop must not ramp fingers
    assert robot.left_arm_joints[0].drive_target == 100
    assert robot.right_arm_joints[0].drive_target == -100
    env._advance()
    assert env._sim_step_count == 10
    assert len(left.history) == len(right.history) == 10
    assert left.drive_target == pytest.approx(.01 + expected * .04)
    assert left.history[0] == pytest.approx((.01 + min(expected, .1) * .04,
                                            min(expected, .1) * .04 * .05))
    assert robot.left_gripper_val == expected
    # Default upstream epsilon limits opening increments, not closing increments.
    goals["left_gripper"] = [0]
    env._advance()
    assert left.history[10] == pytest.approx((.01, -expected * .04 * .05))
    # A hold preserves measured overshoot, bypassing command bounds without affecting its peer.
    env.gripper_holds["left_gripper"] = [.05001]
    peer_history = list(right.history)
    env._apply_targets(goals)
    assert left.drive_target == .05001 and left.velocity == 0
    assert right.history == peer_history
    env._advance()
    assert left.drive_target == .05001
