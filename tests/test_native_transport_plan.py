"""Transport acceptance must use real sensor inventory and effective task instances."""

from types import SimpleNamespace

import pytest
from helpers import load_tool
from roboenv.tasks.model import Task


def task(backend):
    return Task(
        "probe",
        backend,
        "probe",
        "robot",
        "Probe devices.",
        backend_kwargs={"scene_model": "fixture"},
    )


def test_behavior_requires_distinct_instances_without_mutating_task():
    tool = load_tool("check_native_transport")
    original = task("behavior")
    for instances in (None, (0, 0), (-1, 2), (True, 2)):
        with pytest.raises(ValueError, match="activity instance IDs"):
            tool.select_cases(original, (0, 0), instances)
    cases = tool.select_cases(original, (0, 0), (2, 5))
    assert [case.backend_kwargs["activity_instance_id"] for case in cases] == [2, 5]
    assert all(case.seed == 0 for case in cases)
    assert all(case.backend_kwargs["scene_model"] == "fixture" for case in cases)
    assert original.backend_kwargs == {"scene_model": "fixture"}


@pytest.mark.parametrize("backend", ["robotwin", "robodojo"])
def test_source_simulators_keep_explicit_seed_selection(backend):
    tool = load_tool("check_native_transport")
    assert [case.seed for case in tool.select_cases(task(backend), (3, 7))] == [3, 7]
    with pytest.raises(ValueError, match="distinct seeds"):
        tool.select_cases(task(backend), (3, 3))
    with pytest.raises(ValueError, match="only for BEHAVIOR"):
        tool.select_cases(task(backend), instances=(0, 1))


def test_camera_discovery_supports_behavior_and_rejects_non_sensor():
    tool = load_tool("check_native_transport")
    inventory = [
        {"name": "left_arm", "type": "joint"},
        {"name": "camera_1", "type": "rgbd"},
        {"name": "camera_0", "type": "rgbd"},
    ]
    client = SimpleNamespace(devices=lambda: {"devices": inventory})
    assert tool.camera_name(client) == "camera_0"
    assert tool.camera_name(client, "camera_1") == "camera_1"
    with pytest.raises(ValueError, match="not a native color camera"):
        tool.camera_name(client, "left_arm")
    inventory.append({"name": "head_camera", "type": "rgb"})
    assert tool.camera_name(client) == "head_camera"
    inventory[:] = inventory[:1]
    with pytest.raises(ValueError, match="require a native color camera"):
        tool.camera_name(client)


def test_transport_target_moves_away_from_a_joint_limit():
    tool = load_tool("check_native_transport")
    positions = [1.02, 0.0]
    joints = [{"limits": {"min": -1.0, "max": 1.0}}, {"limits": {"min": 0.0, "max": 0.0}}]
    descriptor = {
        "intrinsics": {
            "model": {"joints": joints},
            "kinematics": {
                "frame": "flange",
                "fk": "numerical",
                "feedback": "driver",
                "frame_to_reported": [0, 0, 0, 0, 0, 0, 1],
                "ik": None,
            },
        }
    }
    device = SimpleNamespace(
        read=lambda: {"data": {"positions": positions}}, describe=lambda: descriptor
    )
    client = SimpleNamespace(device=lambda name: device)
    assert tool.target(client, "left_arm", 0.1) == [0.9, 0.0]
    assert positions == [1.02, 0.0]
    joints[0]["limits"] = {"min": 0.0, "max": 0.0}
    with pytest.raises(ValueError, match="No joint has room"):
        tool.target(client, "left_arm", 0.1)
