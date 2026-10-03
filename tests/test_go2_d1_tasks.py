"""Task scoring rejects shortcuts, drops, premature release and stale holds."""

import copy
import itertools

import numpy as np
import pytest
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import (
    CONTROL,
    PICK_PLACE,
    REACH,
    TaskState,
    definition,
)

REACH_POSITION = definition(REACH)["reach_position"]
REACH_QUAT = definition(REACH)["reach_quaternion_xyzw"]
OBJECT_SIZE = definition(PICK_PLACE)["object_size"]


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
        ("robot_fallen", [{"code": "fallen"}]),
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


def test_custom_reach_spec_controls_goal_tolerance_hold_and_snapshot():
    from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import catalog_definition

    spec = catalog_definition(REACH)
    spec.update(
        reach_position=[2.0, 0.0, 0.8],
        reach_quaternion_xyzw=[0, 0, 0, 1],
        reach_position_tolerance_m=0.01,
        hold_steps=2,
    )
    state = TaskState(REACH, spec, [0, 0, 0.2])
    update(state, reach_sample(), 50)
    assert not state.completed
    sample = reach_sample(tcp_position=[2.02, 0, 0.8], tcp_quaternion=[0, 0, 0, 1])
    update(state, sample, 2)
    assert not state.completed
    sample["tcp_position"] = [2.005, 0, 0.8]
    state.update(sample)
    assert not state.completed
    state.update(sample)
    assert state.completed
    spec["reach_position"][0] = 99
    assert state.snapshot()["spec"]["reach_position"] == [2.0, 0.0, 0.8]
    assert state.snapshot()["spec"]["tcp_in_flange_m"] == [0, 0, 0.2]


def test_custom_destination_and_pick_hold_are_used():
    from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import catalog_definition

    spec = catalog_definition(PICK_PLACE)
    spec.update(destination=[3, 0, 0], destination_half_size_m=0.1, lift_hold_steps=2, hold_steps=2)
    state = TaskState(PICK_PLACE, spec)
    update(state, payload_sample(), 2)
    assert state.pickup_base is not None
    release = payload_sample(x=3, z=0.03, base_x=2.5, finger_forces=[0, 0], gripper_opening=1)
    update(state, release, 2)
    assert state.completed


@pytest.mark.parametrize(
    "name,changes",
    [
        (REACH, {"reach_position": [1, 2]}),
        (REACH, {"reach_position": [1, float("nan"), 3]}),
        (REACH, {"reach_quaternion_xyzw": [0, 0, 0, 0]}),
        (REACH, {"hold_steps": True}),
        (REACH, {"hold_steps": 1.5}),
        (REACH, {"reach_position_tolerance_m": -1}),
        (REACH, {"unknown": 1}),
        (PICK_PLACE, {"object_size": [0, 1, 1]}),
        (PICK_PLACE, {"object_mass_kg": 6}),
        (PICK_PLACE, {"minimum_gripper_opening": 1}),
        (PICK_PLACE, {"settled_bottom_min_m": 1}),
        (PICK_PLACE, {"destination": [1, 0, 1]}),
    ],
)
def test_invalid_task_spec_is_rejected_before_native_start(monkeypatch, name, changes):
    from roboenv_runtime.adapters.isaaclab.go2_d1 import native
    from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env
    from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import catalog_definition

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid configuration started Isaac")

    monkeypatch.setattr(native, "Native", forbidden)
    with pytest.raises(ValueError):
        Go2Env(task_name=name, task_spec={**catalog_definition(name), **changes})


def test_resolved_configuration_is_portable_and_uses_selected_robot(tmp_path, monkeypatch):
    import shutil

    import yaml
    from roboenv_runtime.adapters.isaaclab.go2_d1 import tasks

    original = resolve_task("isaaclab/" + REACH)
    catalog = tmp_path / "catalog"
    shutil.copytree(original.robot, catalog / "robots" / original.task.robot)
    target = catalog / "tasks/isaaclab" / REACH / "task.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(original.path, target)
    profile = catalog / "robots" / original.task.robot / "devices/gripper/profile.yaml"
    data = yaml.safe_load(profile.read_text())
    data["intrinsics"]["frames"]["tcp"]["translation_m"] = [0, 0, 0.2]
    profile.write_text(yaml.safe_dump(data))
    resolved = resolve_task("isaaclab/" + REACH, catalog=catalog)

    def forbidden(*args):
        pytest.fail("Portable configuration attempted to read the source catalog")

    monkeypatch.setattr(tasks, "catalog_definition", forbidden)
    monkeypatch.setattr(tasks, "catalog_tcp", forbidden)
    kwargs = resolved.task.backend_kwargs
    spec = definition(REACH, kwargs["task_spec"], kwargs["tcp_in_flange_m"])
    assert spec["tcp_in_flange_m"] == [0, 0, 0.2]
    assert spec["reach_position"] == REACH_POSITION


def test_scene_measurements_use_configured_tcp_and_object_dimensions():
    from types import SimpleNamespace

    from roboenv_runtime.adapters.isaaclab.go2_d1.tasks import TaskScene, catalog_definition

    config = catalog_definition(PICK_PLACE)
    config["object_size"] = [0.1, 0.2, 0.3]
    scene = TaskScene.__new__(TaskScene)
    scene.name = PICK_PLACE
    scene.spec = definition(PICK_PLACE, config, [0, 0, 0.2])
    samples = []
    scene.state = SimpleNamespace(update=samples.append)
    scene.forces = [1, 2]
    scene.object = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pose_w=np.array([1.0, 0, 0.15, 0, 0, 0, 1]),
            root_lin_vel_w=np.zeros(3),
            root_ang_vel_w=np.zeros(3),
        )
    )
    scene.native = SimpleNamespace(
        root_pose=lambda: np.array([0.0, 0, 0, 0, 0, 0, 1]),
        model=SimpleNamespace(forward=lambda q: {"d1_Link6": np.eye(4)}),
        joints=lambda: {"d1_Joint7_1": 0.03, "d1_Joint7_2": -0.03},
        array=lambda v: v,
        robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=np.zeros(3))),
    )
    scene.update(False)
    assert np.allclose(samples[0]["tcp_position"], [0, 0, 0.2])
    corners = samples[0]["object_corners"]
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0), config["object_size"])
    assert corners[:, 2].min() == pytest.approx(0)
