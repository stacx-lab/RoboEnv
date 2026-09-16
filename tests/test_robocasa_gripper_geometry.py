"""CPU-only geometry checks against the compiled upstream Panda gripper."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

from roboenv_runtime.bindings.backends.robocasa.gripper_geometry import geometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def panda_asset(backend):
    if backend == "libero":
        configured = os.environ.get("ROBOENV_LIBERO_GRIPPER_XML")
        if not configured:
            pytest.skip("Requires the pinned LIBERO robosuite 1.4.0 gripper XML")
        return Path(configured)
    return ROOT / "third_party/robosuite/robosuite/models/assets/grippers/panda_gripper.xml"


@pytest.mark.parametrize("backend", ["robocasa", "libero"])
def test_pad_frames_and_aperture_follow_compiled_model(backend):
    asset = panda_asset(backend)
    if not asset.is_file():
        pytest.skip("Requires the pinned robosuite assets")
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    ids = SimpleNamespace(
        _model=model,
        site_name2id=lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n),
        geom_name2id=lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n),
    )
    sim = SimpleNamespace(model=ids, data=SimpleNamespace(_data=data))
    robot = SimpleNamespace(
        gripper={
            "right": SimpleNamespace(
                important_sites={"grip_site": "grip_site"},
                important_geoms={
                    "left_fingerpad": ["finger1_pad_collision"],
                    "right_fingerpad": ["finger2_pad_collision"],
                },
            )
        }
    )
    catalogs = list((ROOT / "catalog/robots" / backend).glob("*/devices/gripper/profile.yaml"))
    catalog = yaml.safe_load(catalogs[0].read_text())["intrinsics"]
    site = ids.site_name2id("grip_site")
    for rotation in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]):
        model.body_quat[1] = rotation
        model.body_pos[1] = [0.2, -0.4, 0.8]
        for opening in ([0, 0], [0.02, -0.02], [0.04, -0.01], [0.04, -0.04]):
            data.qpos[:] = opening
            mujoco.mj_forward(model, data)
            facts, aperture, frames = geometry(robot.gripper["right"], sim, np.array([0, 1]))
            assert facts == {k: catalog[k] for k in facts}
            assert aperture == catalog["feedback"]["aperture_model"]
            for profile in catalogs:
                expected = yaml.safe_load(profile.read_text())["intrinsics"]
                assert facts == {key: expected[key] for key in facts}
                assert aperture == expected["feedback"]["aperture_model"]
            gap = max(0, aperture["offset_m"] + np.dot(aperture["coefficients"], data.qpos))
            assert gap == pytest.approx(max(0, opening[0] - opening[1] - 0.001))
            actual = frames()
            for name, geom_name in (
                ("left_pad", "finger1_pad_collision"),
                ("right_pad", "finger2_pad_collision"),
            ):
                frame = actual[name]
                quat = np.array([frame[6], *frame[3:6]])
                local = np.empty(9)
                mujoco.mju_quat2Mat(local, quat)
                normal = data.site_xmat[site].reshape(3, 3) @ local.reshape(3, 3)[:, 2]
                point = data.site_xpos[site] + data.site_xmat[site].reshape(3, 3) @ frame[:3]
                geom = ids.geom_name2id(geom_name)
                in_box = data.geom_xmat[geom].reshape(3, 3).T @ (point - data.geom_xpos[geom])
                assert np.abs(in_box) == pytest.approx([0, 0.004, 0], abs=1e-10)
                assert np.linalg.norm(normal) == pytest.approx(1)
                assert frame[2] == pytest.approx(-0.0036)
            assert actual["right_pad"][0] - actual["left_pad"][0] == pytest.approx(
                opening[0] - opening[1] - 0.001
            )


def test_libero_binding_returns_contact_frames_without_advancing_physics():
    from roboenv.devices.bindings.registry import BindingContext
    from roboenv.devices.bindings.spec import BindingPlan
    from roboenv.devices.model import ExecutionContext
    from roboenv_runtime.bindings.backends.libero.devices import bind

    asset = panda_asset("libero")
    tree = ET.parse(asset).getroot()
    for mesh in tree.findall("asset/mesh"):
        mesh.set("file", str(asset.parent / mesh.get("file")))
    for element in tree.iter():
        for key in ("name", "joint", "mesh", "site"):
            if key in element.attrib:
                element.set(key, "gripper0_" + element.get(key))
    worldbody = tree.find("worldbody")
    hand = worldbody.find("body")
    worldbody.remove(hand)
    # Only the gripper is exercised; independent dummy joints supply the arm inventory.
    for index in range(7):
        body = ET.SubElement(worldbody, "body")
        ET.SubElement(body, "joint", name=f"arm{index}")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=".01", diaginertia=".001 .001 .001")
    ET.SubElement(worldbody, "body", name="flange").append(hand)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = SimpleNamespace(
        _model=model,
        body_name2id=lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name),
        site_name2id=lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name),
        joint_name2id=lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name),
        joint_id2name=lambda index: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
        geom_name2id=lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name),
    )
    robot = SimpleNamespace(
        controller=SimpleNamespace(joint_index=np.arange(7)),
        action_dim=8,
        robot_model=SimpleNamespace(eef_name="flange"),
        gripper_joints=["gripper0_finger_joint1", "gripper0_finger_joint2"],
        gripper=SimpleNamespace(
            speed=0.01,
            important_sites={"grip_site": "gripper0_grip_site"},
            important_geoms={
                "left_fingerpad": ["gripper0_finger1_pad_collision"],
                "right_fingerpad": ["gripper0_finger2_pad_collision"],
            },
        ),
    )
    world = SimpleNamespace(
        _robot=robot, sim=SimpleNamespace(model=ids, data=SimpleNamespace(_data=data))
    )
    root = ROOT / "catalog/robots/libero/libero-panda-tabletop"
    spec = next(
        spec
        for spec in BindingPlan.load(root, backend="libero").devices
        if spec.entity == "gripper"
    )
    bound = bind(spec, root, BindingContext("libero", "panda", SimpleNamespace(_env=world)))
    for opening in ([0.04, -0.04], [0.02, -0.01]):
        data.qpos[7:] = opening
        mujoco.mj_forward(model, data)
        reading = bound.execute("read", {}, ExecutionContext(read_only=True))
        assert reading["positions"] == opening
        assert reading["aperture_m"] == pytest.approx(opening[0] - opening[1] - 0.001)
        assert set(reading["frame_poses"]) == {"left_pad", "right_pad"}
        assert data.time == 0
