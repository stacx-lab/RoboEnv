"""Task scoring rejects shortcuts, drops, premature release and stale holds."""

import copy
import itertools

import numpy as np
import pytest
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import (
    CONTROL,
    OBJECT_SIZE,
    PICK_PLACE,
    REACH,
    REACH_POSITION,
    REACH_QUAT,
    TaskState,
)


def reach_sample(**changes):
    sample = dict(
        fallen=False,
        base_position=[1.0, 0.0, 0.3],
        base_speed=0.0,
        tcp_position=REACH_POSITION,
        tcp_quaternion=REACH_QUAT,
    )
    return {**sample, **changes}


def payload_sample(x=1.25, z=0.2, base_x=0.88, **changes):
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * np.array(OBJECT_SIZE) / 2 + [
        x,
        0,
        z,
    ]
    sample = dict(
        fallen=False,
        base_position=[base_x, 0.0, 0.3],
        base_speed=0.0,
        object_position=[x, 0, z],
        object_corners=corners,
        finger_forces=[2.0, 2.0],
        object_tcp_distance=0.01,
        object_speed=0.0,
        object_angular_speed=0.0,
        gripper_opening=0.3,
    )
    return {**sample, **changes}


def update(state, sample, count):
    for _ in range(count):
        state.update(sample)


def test_reach_requires_base_motion_orientation_and_continuous_hold():
    s = TaskState(REACH)
    update(s, reach_sample(base_position=[0, 0, 0.3]), 60)
    assert not s.completed
    update(s, reach_sample(tcp_quaternion=[0, 0, 0, 1]), 60)
    assert not s.completed
    update(s, reach_sample(), 49)
    assert not s.completed
    s.update(reach_sample(base_speed=0.1))
    assert s.hold == 0
    update(s, reach_sample(), 50)
    assert s.completed and s.done
    assert "stage" not in s.snapshot()
    before = copy.deepcopy(s.snapshot())
    s.update(reach_sample(fallen=True))
    assert s.snapshot() == before


def test_pick_place_requires_bilateral_lift_carry_release_and_settle():
    s = TaskState(PICK_PLACE)
    update(s, payload_sample(z=0.03), 60)
    assert s.pickup_base is None
    update(s, payload_sample(finger_forces=[2, 0]), 60)
    assert s.pickup_base is None
    update(s, payload_sample(), 10)
    assert s.pickup_base is not None and not s.placement_started
    # Reaching the destination with the arm alone is insufficient.
    update(s, payload_sample(x=2.5, z=0.06), 5)
    assert s.pickup_base is not None and not s.placement_started
    update(s, payload_sample(x=2.5, z=0.06, base_x=2.13), 1)
    assert s.placement_started
    release = payload_sample(x=2.5, z=0.03, base_x=2.13, finger_forces=[0, 0], gripper_opening=1.0)
    update(s, {**release, "object_speed": 0.1}, 60)
    assert not s.completed
    update(s, release, 49)
    assert not s.completed
    update(s, release, 1)
    assert s.completed and s.carried_distance >= 0.6
    assert "stage" not in s.snapshot()


@pytest.mark.parametrize("failure", [{"finger_forces": [0, 0]}, {"z": 0.03}])
def test_transport_drop_is_terminal_and_cannot_be_recovered(failure):
    s = TaskState(PICK_PLACE)
    update(s, payload_sample(), 10)
    update(s, payload_sample(**failure), 11)
    assert s.failure == "object_dropped" and s.done
    update(s, payload_sample(x=2.5, z=0.03, base_x=2.13, gripper_opening=1.0), 100)
    assert not s.completed


def test_destination_checks_entire_object_and_no_success_before_pickup():
    s = TaskState(PICK_PLACE)
    release = payload_sample(x=2.5, z=0.03, base_x=2.13, finger_forces=[0, 0], gripper_opening=1.0)
    update(s, release, 100)
    assert not s.completed and s.pickup_base is None
    update(s, payload_sample(), 10)
    update(s, payload_sample(x=2.5, z=0.06, base_x=2.13), 1)
    update(s, payload_sample(x=2.64, z=0.03, base_x=2.13), 1)
    assert s.failure == "object_left_destination"


@pytest.mark.parametrize("name", [REACH, PICK_PLACE])
def test_fall_prevents_success(name):
    s = TaskState(name)
    s.update({"fallen": True})
    assert s.failure == "robot_fallen" and not s.completed


def test_reach_goal_is_outside_even_a_conservative_fixed_base_workspace():
    resolved = resolve_task("isaaclab/" + REACH)
    arm = next(d for d in resolved.bindings.devices if d.name == "arm")
    profile = arm.resolved_profile(resolved.robot)
    # Triangle inequality gives an upper bound for every joint configuration.
    mount = profile["extrinsics"]["mount"]["translation_m"]
    chain = profile["intrinsics"]["model"]["chain"]
    bound = np.linalg.norm(mount) + sum(np.linalg.norm(j["origin"][:3]) for j in chain) + 0.105
    assert bound < np.linalg.norm(REACH_POSITION[:2]) - 0.025


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("object_dropped", []),
        ("object_left_destination", []),
        ("robot_fallen", ["fallen"]),
        (None, []),
    ],
)
def test_public_robot_feedback_does_not_expose_task_verdict(fault, expected):
    from types import SimpleNamespace

    from roboenv_runtime.bindings.backends.isaaclab.go2_d1 import read_robot_state

    world = SimpleNamespace(
        read_device=lambda name: {"faults": [fault] if fault else [], "positions": [0.1]}
    )
    assert read_robot_state(world, "arm") == {"faults": expected, "positions": [0.1]}


@pytest.mark.parametrize("name", [CONTROL, REACH, PICK_PLACE])
def test_task_state_and_snapshot_have_no_stage(name):
    state = TaskState(name)
    assert not hasattr(state, "stage")
    assert "stage" not in state.snapshot()
    state.update({"fallen": True})
    assert "stage" not in state.snapshot()


def test_placement_latch_preserves_destination_failure_after_block_is_raised():
    state = TaskState(PICK_PLACE)
    update(state, payload_sample(), 10)
    state.update(payload_sample(x=2.5, z=0.06, base_x=2.13))
    state.update(payload_sample(x=2.5, z=0.2, base_x=2.13))
    assert state.placement_started and state.failure is None
    state.update(payload_sample(x=2.7, z=0.2, base_x=2.13))
    assert state.failure == "object_left_destination"
