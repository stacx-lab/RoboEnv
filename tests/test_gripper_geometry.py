"""Contact geometry against pinned source facets, independent of simulator boot."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from roboenv.devices.model import local_frame
from roboenv_runtime.adapters.gripper_geometry import ParallelGripper
from roboenv_runtime.adapters.robotwin.rig import native_tcp
from roboenv_runtime.adapters.urdf_geometry import fixed_transform, matrix, matrix_pose, origin
from roboenv_runtime.bindings.backends.libero.devices import same_facts

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/gripper_contacts.json").read_text())
RIGS = {
    "aloha": "robotwin/robotwin-aloha-agilex",
    "x5": "robodojo/robodojo-arx-x5",
    "r1pro": "behavior/behavior-r1pro",
}


def build(tmp_path, case):
    urdf = tmp_path / "gripper.urdf"
    urdf.write_text(case["urdf"])
    return ParallelGripper(urdf, case["names"], case["flange"], case["model"]), urdf


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["model"])
def test_aperture_matches_source_surfaces_at_asymmetric_openings(tmp_path, case):
    geometry, urdf = build(tmp_path, case)
    joints = {joint.get("name"): joint for joint in ET.parse(urdf).getroot().findall("joint")}
    calibration = json.loads(
        (ROOT / "runtime/src/roboenv_runtime/adapters/gripper_contacts.json").read_text()
    )[case["model"]]
    sign = 1
    for positions in ([0, 0], [sign * 0.01, sign * 0.03], [sign * 0.035, sign * 0.015]):
        reading = geometry.read(positions)
        assert set(reading) == {"aperture_m"}
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
            point = np.asarray(calibration["contacts"][index]["point_m"])
            triangle = np.asarray(triangle)
            edges = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
            uv = np.linalg.lstsq(edges, point - triangle[0], rcond=None)[0]
            np.testing.assert_allclose(triangle[0] + edges @ uv, point, atol=1e-9)
            assert uv.min() >= -1e-6 and uv.sum() <= 1 + 1e-6
            normal = np.cross(edges[:, 0], edges[:, 1])
            np.testing.assert_allclose(
                calibration["contacts"][index]["normal"], normal / np.linalg.norm(normal), atol=1e-7
            )
            contacts.append((transform @ np.r_[point, 1])[:3])
        closing = geometry.mount[:3, :3] @ geometry.intrinsics["geometry"]["closing_axis"]
        expected = max(0, np.dot(contacts[1] - contacts[0], closing))
        assert reading["aperture_m"] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(geometry.coefficients, [sign, sign], atol=1e-9)


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
        text = case["urdf"]
        names = case["names"]
        flange = case["flange"]
        if side == "right" and case["model"] in {"aloha", "r1pro"}:
            before, after = ("fl_", "fr_") if case["model"] == "aloha" else ("left_", "right_")
            text = text.replace(before, after)
            names = [name.replace(before, after) for name in names]
            flange = flange.replace(before, after)
        geometry, _ = build(tmp_path, {**case, "urdf": text, "names": names, "flange": flange})
        facts = geometry.device(profile["name"], side + "_arm")
        tcp = facts["intrinsics"]["frames"]["tcp"]
        mount = facts["extrinsics"]["mount"]
        np.testing.assert_allclose(
            matrix(mount["translation_m"] + mount["rotation_xyzw"])
            @ matrix(tcp["translation_m"] + tcp["rotation_xyzw"]), geometry.tcp, atol=1e-9,
        )
        if case["model"] == "aloha":
            native = np.eye(4)
            native[:3, 3] = [0.12, 0, 0]
            facts["intrinsics"]["frames"]["tcp"] = local_frame(
                matrix_pose(np.linalg.inv(geometry.mount) @ native)
            )
        for key in ("intrinsics", "extrinsics"):
            assert same_facts(profile[key], facts[key])


@pytest.mark.parametrize("positions", [[], [0], [0, 0, 0], [float("nan"), 0], [0, float("inf")]])
def test_missing_or_invalid_feedback_cannot_produce_aperture(tmp_path, positions):
    geometry, _ = build(tmp_path, CASES[0])
    with pytest.raises(ValueError, match="two finite"):
        geometry.read(positions)


@pytest.mark.parametrize("backend", ["robotwin", "robodojo", "behavior"])
def test_native_read_paths_publish_aperture_without_finger_poses(tmp_path, backend):
    geometry, _ = build(tmp_path, CASES[0])
    if backend == "robotwin":
        pytest.importorskip("sapien")
        from roboenv_runtime.adapters.robotwin.devices import RoboTwinDevices

        world = RoboTwinDevices.__new__(RoboTwinDevices)
        joints = [object(), object()]
        entity = SimpleNamespace(
            get_active_joints=lambda: joints,
            get_qpos=lambda: np.array([0.01, 0.02]),
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
            data=SimpleNamespace(joint_pos=np.array([[0.01, 0.02]]))
        )
        world.actuators = {"left_gripper": (None, key, [0, 1])}
        world.rig_frames = {"left": {"gripper": geometry}}
        world.physics_steps, world.episode_start_steps, world.physics_dt = 250, 0, 0.004
    else:
        from roboenv_runtime.adapters.behavior.devices import BehaviorDevices

        world = BehaviorDevices.__new__(BehaviorDevices)
        world.robot = SimpleNamespace(
            get_joint_positions=lambda: np.array([0.01, 0.02]),
        )
        world.groups = {"left_gripper": {"indices": [0, 1], "gripper": geometry}}
        world.rig_frames = {}
        world.control, world.control_interval = SimpleNamespace(frame=30), 1 / 30
    reading = world.device_read("left_gripper")
    assert "frame_poses" not in reading
    # ALOHA's calibrated contact surfaces have a 0.000000165976 m zero-position gap.
    assert reading["aperture_m"] == pytest.approx(0.030000165976, abs=1e-10)
    assert reading["timestamp"] == pytest.approx(1)
    assert "positions" not in reading and "velocities" not in reading
    assert not {"force_n", "pressure_pa", "object_detected"}.intersection(reading)
    geometry.read = lambda positions: pytest.fail("stop must not compute aperture feedback")
    goals = {"left_gripper": [0.1], "right_gripper": [0.8]}
    stopped = []
    world.control = SimpleNamespace(
        frame=30, targets=goals,
        stop=lambda name, target: stopped.append((name, list(target))) or {"stopped": True},
    )
    world._measured_target = lambda name: pytest.fail("gripper stop must not read aperture")
    assert world.device_stop("left_gripper")["stopped"]
    assert stopped == [("left_gripper", [0.1])]
    assert goals["right_gripper"] == [0.8]


@pytest.mark.parametrize("axis,global_rotation,delta,expected", [
    ([1, 0, 0], np.diag([1, -1, -1]), np.eye(3), np.eye(3)),
    ([0, 0, 1], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
     [[0, 0, -1], [0, 1, 0], [1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
])
def test_robotwin_native_tcp_includes_joint_anchor_and_native_bias(
    tmp_path, axis, global_rotation, delta, expected
):
    urdf = tmp_path / "ee.urdf"
    urdf.write_text('<robot><joint name="ee" type="revolute"><parent link="wrist"/>'
                    '<child link="flange"/><axis xyz="' + " ".join(map(str, axis))
                    + '"/></joint></robot>')
    config = {"ee_joints": ["ee", "ee"], "gripper_bias": 0.12,
              "global_trans_matrix": global_rotation, "delta_matrix": delta}
    for index in (0, 1):
        tcp = native_tcp(urdf, config, index, "flange")
        np.testing.assert_allclose(tcp[:3, :3], expected, atol=1e-12)
        np.testing.assert_allclose(tcp[:3, 3], np.asarray(expected) @ [0.12, 0, 0], atol=1e-12)
    config["ee_frame"] = "link"
    tcp = native_tcp(urdf, config, 0, "flange")
    np.testing.assert_allclose(tcp[:3, :3], np.asarray(global_rotation) @ delta, atol=1e-12)
