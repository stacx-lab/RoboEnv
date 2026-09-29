"""Contact geometry against pinned source facets, independent of simulator boot."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from roboenv_runtime.adapters.gripper_geometry import ParallelGripper
from roboenv_runtime.adapters.urdf_geometry import fixed_transform, matrix, origin
from roboenv_runtime.bindings.backends.libero.devices import same_facts

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/gripper_contacts.json").read_text())
RIGS = {
    "piper": "robotwin/robotwin-dual-piper",
    "aloha": "robotwin/robotwin-aloha-agilex",
    "x5": "robodojo/robodojo-arx-x5",
    "r1pro": "behavior/behavior-r1pro",
}


def build(tmp_path, case):
    urdf = tmp_path / "gripper.urdf"
    urdf.write_text(case["urdf"])
    return ParallelGripper(urdf, case["names"], case["flange"], case["model"]), urdf


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["model"])
def test_contacts_stay_on_source_surfaces_at_asymmetric_openings(tmp_path, case):
    geometry, urdf = build(tmp_path, case)
    joints = {joint.get("name"): joint for joint in ET.parse(urdf).getroot().findall("joint")}
    sign = 1
    for positions in ([0, 0], [sign * 0.01, sign * 0.03], [sign * 0.035, sign * 0.015]):
        reading = geometry.read(positions)
        contacts = []
        for index, (name, triangle) in enumerate(
            zip(case["names"], case["triangles"], strict=True)
        ):
            joint = joints[name]
            transform = fixed_transform(urdf, case["flange"], joint.find("parent").get("link"))
            transform = transform @ matrix(origin(joint))
            displacement = np.eye(4)
            displacement[:3, 3] = (
                np.fromstring(joint.find("axis").get("xyz"), sep=" ") * positions[index]
            )
            transform = transform @ displacement
            contact = geometry.tcp @ matrix(reading["frame_poses"][f"pad_{index + 1}"])
            local = np.linalg.inv(transform) @ contact
            triangle = np.asarray(triangle)
            edges = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
            uv = np.linalg.lstsq(edges, local[:3, 3] - triangle[0], rcond=None)[0]
            np.testing.assert_allclose(triangle[0] + edges @ uv, local[:3, 3], atol=1e-9)
            assert uv.min() >= -1e-6 and uv.sum() <= 1 + 1e-6
            normal = np.cross(edges[:, 0], edges[:, 1])
            np.testing.assert_allclose(local[:3, 2], normal / np.linalg.norm(normal), atol=1e-7)
            contacts.append(contact[:3, 3])
        closing = geometry.intrinsics["geometry"]["closing_axis"]
        expected = max(0, np.dot(contacts[1] - contacts[0], closing))
        assert reading["aperture_m"] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(geometry.coefficients, [sign, sign], atol=1e-9)
    zero = geometry.read([0, 0])["frame_poses"]
    np.testing.assert_allclose((np.array(zero["pad_1"][:3]) + zero["pad_2"][:3]) / 2, 0, atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["model"])
def test_catalogs_match_source_geometry_on_both_arms(tmp_path, case):
    for side in ("left", "right"):
        profile = yaml.safe_load(
            (
                ROOT
                / "catalog/robots"
                / RIGS[case["model"]]
                / "devices"
                / f"{side}_gripper/profile.yaml"
            ).read_text()
        )
        actual_names = [entry["name"] for entry in profile["intrinsics"]["feedback"]["positions"]]
        text = case["urdf"]
        names = case["names"]
        flange = case["flange"]
        if side == "right" and case["model"] in {"aloha", "r1pro"}:
            before, after = ("fl_", "fr_") if case["model"] == "aloha" else ("left_", "right_")
            text = text.replace(before, after)
            names = [name.replace(before, after) for name in names]
            flange = flange.replace(before, after)
        assert names == actual_names
        geometry, _ = build(tmp_path, {**case, "urdf": text, "names": names, "flange": flange})
        facts = geometry.device(profile["name"], side + "_arm")
        for key in ("intrinsics", "extrinsics"):
            assert same_facts(profile[key], facts[key])


@pytest.mark.parametrize("positions", [[], [0], [0, 0, 0], [float("nan"), 0], [0, float("inf")]])
def test_missing_or_invalid_feedback_cannot_produce_contact_frames(tmp_path, positions):
    geometry, _ = build(tmp_path, CASES[0])
    with pytest.raises(ValueError, match="two finite"):
        geometry.read(positions)


@pytest.mark.parametrize("backend", ["robotwin", "robodojo", "behavior"])
def test_native_read_paths_publish_measured_contact_frames(tmp_path, backend):
    geometry, _ = build(tmp_path, CASES[0])
    if backend == "robotwin":
        pytest.importorskip("sapien")
        from roboenv_runtime.adapters.robotwin.devices import RoboTwinDevices

        world = RoboTwinDevices.__new__(RoboTwinDevices)
        joints = [object(), object()]
        entity = SimpleNamespace(
            get_active_joints=lambda: joints,
            get_qpos=lambda: np.array([0.01, 0.02]),
            get_qvel=lambda: np.zeros(2),
        )
        world.task = SimpleNamespace(
            robot=SimpleNamespace(
                left_entity=entity, left_gripper=[(joint, 1, 0) for joint in joints]
            )
        )
        world.rig_frames = {"left": {"gripper": geometry}}
        world.device_facts = {"left_gripper": {"type": "gripper"}}
        world._sim_step_count = 250
    elif backend == "robodojo":
        from roboenv_runtime.adapters.robodojo.devices import RoboDojoDevices

        world = RoboDojoDevices.__new__(RoboDojoDevices)
        key = SimpleNamespace(
            data=SimpleNamespace(joint_pos=np.array([[0.01, 0.02]]), joint_vel=np.zeros((1, 2)))
        )
        world.actuators = {"left_gripper": (None, key, [0, 1])}
        world.rig_frames = {"left": {"gripper": geometry}}
        world.physics_steps, world.episode_start_steps, world.physics_dt = 250, 0, 0.004
    else:
        from roboenv_runtime.adapters.behavior.devices import BehaviorDevices

        world = BehaviorDevices.__new__(BehaviorDevices)
        world.robot = SimpleNamespace(
            get_joint_positions=lambda: np.array([0.01, 0.02]),
            get_joint_velocities=lambda: np.zeros(2),
        )
        world.groups = {"left_gripper": {"indices": [0, 1], "gripper": geometry}}
        world.rig_frames = {}
        world.control, world.control_interval = SimpleNamespace(frame=30), 1 / 30
    reading = world.device_read("left_gripper")
    assert set(reading["frame_poses"]) == {"pad_1", "pad_2"}
    assert reading["aperture_m"] == pytest.approx(0.03, abs=1e-8)
    assert reading["sample_time_s"] == pytest.approx(1)
    assert reading["positions"] == [0.01, 0.02]
