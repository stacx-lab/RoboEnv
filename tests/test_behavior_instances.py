"""Instance selection and native reset ordering, without booting Isaac."""

import json
import sys
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.behavior.instances import restore_instance


def fixture(monkeypatch, tmp_path, *, generic=True):
    events = []
    pose_key = "Robot" if generic else "r1pro"
    state = {
        "radio.n.01_1": {"pose": [1, 2, 3]},
        "robot_poses": {pose_key: [{"position": [4, 5, 6], "orientation": [0, 0, 0, 1]}]},
    }
    path = tmp_path / "instance.json"
    path.write_text(json.dumps(state))
    entity = SimpleNamespace(
        load_state=lambda value, serialized: events.append(("load", value, serialized)),
        keep_still=lambda: events.append("still"),
    )
    robot = SimpleNamespace(
        model="r1pro",
        set_position_orientation=lambda *pose: events.append(("pose", pose)),
        keep_still=lambda: events.append("robot_still"),
    )
    task = SimpleNamespace(
        scene_name="house", activity_name="turning_on_radio", activity_definition_id=0,
        get_cached_activity_scene_filename=lambda *args: "instance_" + str(args[-1]),
        object_scope={"radio.n.01_1": entity, "agent.n.01_1": robot, "water.n.06_1": None},
    )
    scene = SimpleNamespace(
        write_task_metadata=lambda **kwargs: events.append("metadata"),
        update_initial_file=lambda: events.append("snapshot"),
        reset=lambda: events.append("reset"),
    )
    sim = SimpleNamespace(
        update_handles=lambda: events.append("handles"),
        step_physics=lambda: events.append("physics"),
    )
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(sim=sim))
    monkeypatch.setitem(sys.modules, "omnigibson.utils.asset_utils", SimpleNamespace(
        get_task_instance_path=lambda *args, **kwargs: (
            events.append(("path", args, kwargs)) or str(path)
        )
    ))
    monkeypatch.setitem(sys.modules, "omnigibson.utils.bddl_utils", SimpleNamespace(
        is_system_bddl_inst=lambda key: key.startswith("water")
    ))
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", SimpleNamespace(
        recursively_convert_to_torch=lambda value: value
    ))
    return SimpleNamespace(task=task, scene=scene), robot, path, events


@pytest.mark.parametrize("generic", [True, False])
def test_selected_state_and_robot_pose_precede_settling_and_snapshot(
    monkeypatch, tmp_path, generic
):
    env, robot, path, events = fixture(monkeypatch, tmp_path, generic=generic)
    report = restore_instance(env, robot, 7)
    assert events[0] == (
        "path", ("house", "house_task_turning_on_radio_instances/instance_7-tro_state"),
        {"mode": "train"},
    )
    assert events[1][0] == "load" and events[1][2] is False
    assert events[2] == ("pose", ([4, 5, 6], [0, 0, 0, 1]))
    assert events[3:5] == ["metadata", "handles"]
    assert events[5:-2] == ["physics", "still", "robot_still"] * 25
    assert events[-2:] == ["snapshot", "reset"]
    assert report["split"] == "train" and report["settling_physics_steps"] == 25
    assert len(report["sha256"]) == 64


def test_missing_instance_is_explicit_and_does_not_modify_world(monkeypatch, tmp_path):
    env, robot, path, events = fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(sys.modules["omnigibson.utils.asset_utils"],
                        "get_task_instance_path", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="training instance 7"):
        restore_instance(env, robot, 7)
    assert not events


def test_unknown_instance_entities_are_rejected_before_effects(monkeypatch, tmp_path):
    env, robot, path, events = fixture(monkeypatch, tmp_path)
    data = json.loads(path.read_text())
    data["unexpected.n.01_1"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown task entities"):
        restore_instance(env, robot, 7)
    assert len(events) == 1 and events[0][0] == "path"


def test_close_clears_world_once_and_defers_application_exit(monkeypatch):
    import atexit

    from roboenv_runtime.adapters.behavior.devices import BehaviorDevices

    events = []

    def shutdown():
        events.append("shutdown")

    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(
        clear=lambda: events.append("clear"), shutdown=shutdown
    ))
    monkeypatch.setattr(atexit, "register", lambda callback: events.append(callback))
    world = BehaviorDevices.__new__(BehaviorDevices)
    world.env = world.robot = world.control = object()
    world.close()
    world.close()
    assert events == [shutdown, "clear"]
    assert world.env is world.robot is world.control is None
