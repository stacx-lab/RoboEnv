"""Exercise native stop implementations without installing simulator dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.device_control import DeviceControl

ROOT = Path(__file__).resolve().parents[1] / "runtime/src/roboenv_runtime/adapters"


def stop_method(backend, cls, method):
    # Only this boundary method is needed; importing RoboTwin would require SAPIEN.
    source = ROOT / backend
    definition = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in definition.body if isinstance(n, ast.FunctionDef)
                    and n.name == method)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method]


@pytest.mark.parametrize("backend,cls,method,name", [
    ("robocasa/backend.py", "RoboCasaEnv", "stop_part", "right_gripper"),
    ("libero/backend.py", "LiberoEnv", "stop_part", "gripper"),
    ("robotwin/devices.py", "RoboTwinDevices", "device_stop", "left_gripper"),
    ("robodojo/devices.py", "RoboDojoDevices", "device_stop", "left_gripper"),
    ("behavior/devices.py", "BehaviorDevices", "device_stop", "left_gripper"),
    ("isaaclab/go2_d1/environment.py", "Go2Env", "stop", "gripper"),
])
def test_stop_retains_goal_despite_different_measured_aperture(backend, cls, method, name):
    stop = stop_method(backend, cls, method)
    goal = np.array([-.8, .8])
    gripper = SimpleNamespace(current_action=goal.copy())
    controller = SimpleNamespace(set_goal=lambda value: applied.append(value.copy()))
    applied = []
    world = SimpleNamespace(
        _stream_commands={name: np.array([1.])}, _gripper_command=1.,
        _robot=SimpleNamespace(part_controllers={name: controller}, gripper={"right": gripper}),
        _joint_controller=lambda: controller, command_log=[], _step_count=0,
        targets={name: goal.copy()}, events=[], steps=0,
        read_device=lambda _: pytest.fail("stop must not replace preload with measurements"),
        _measured_target=lambda _: pytest.fail("stop must not read measured gripper position"),
    )
    world.control = DeviceControl(
        {name: [(0, 1)], "peer": [(0, 1)]}, {name: [.1], "peer": [.8]},
        lambda targets: applied.append(targets), lambda: None,
        exhausted=lambda: False, interval_s=.05,
    )
    for _ in range(2):
        stop(world, name)
    np.testing.assert_array_equal(gripper.current_action, goal)
    np.testing.assert_array_equal(world.targets[name], goal)
    assert world.control.targets == {name: [.1], "peer": [.8]}
    assert world.control.frame == 0
    if backend.startswith("robocasa"):
        assert world._stream_commands[name] == [0]
        np.testing.assert_array_equal(applied[-1], goal)
    elif backend.startswith("libero"):
        assert world._gripper_command == 0
    elif backend.startswith(("robotwin", "robodojo", "behavior")):
        world.control.write(name, [.9])
        stop(world, name)
        assert world.control.targets[name] == [.9]
