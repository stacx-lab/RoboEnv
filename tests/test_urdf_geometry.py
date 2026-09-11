"""Numerical geometry checks independent of simulator assets and robot generators."""

import math

import numpy as np
import pytest
from roboenv_runtime.adapters.urdf_geometry import (
    fixed_transform,
    matrix,
    matrix_pose,
    serial_chain,
)


def robot(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test">
      <link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="shoulder" type="revolute">
        <parent link="base"/><child link="arm"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
        <axis xyz="0 0 2"/><limit lower="-2" upper="2" effort="5" velocity="1"/>
      </joint>
      <joint name="camera" type="fixed">
        <parent link="arm"/><child link="tip"/><origin xyz="0 0 0.2"/>
      </joint>
    </robot>""")
    return path


def test_chain_preserves_fixed_tip_and_urdf_joint_frame(tmp_path):
    model, base, tip = serial_chain(robot(tmp_path), ["shoulder"], flange="tip")
    assert (base, tip) == ("base", "tip")
    assert model["chain"][0]["axis"] == [0.0, 0.0, 1.0]
    assert model["joints"][0]["limits"] == {"min": -2.0, "max": 2.0}
    transform = matrix(model["chain"][0]["origin"]) @ matrix(model["chain"][1]["origin"])
    np.testing.assert_allclose(transform[:3, 3], [1, 0, 0.2], atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_fixed_mount_cannot_hide_an_actuated_joint(tmp_path):
    path = robot(tmp_path)
    np.testing.assert_allclose(fixed_transform(path, "arm", "tip")[:3, 3], [0, 0, 0.2])
    with pytest.raises(ValueError, match="moving joint"):
        fixed_transform(path, "base", "tip")
    with pytest.raises(ValueError, match="descended"):
        serial_chain(path, ["shoulder"], base="missing")


@pytest.mark.parametrize("angle", [0, 0.2, math.pi - 1e-10, math.pi, -2.4])
def test_pose_roundtrip_at_singular_quaternion_angles(angle):
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    pose = [1.0, -2.0, 3.0, *(axis * np.sin(angle / 2)), np.cos(angle / 2)]
    transform = matrix(pose)
    np.testing.assert_allclose(matrix(matrix_pose(transform)), transform, atol=1e-12)


@pytest.mark.parametrize("pose", [[0] * 7, [0] * 6, [float("nan"), 0, 0, 0, 0, 0, 1]])
def test_invalid_poses_are_rejected(pose):
    with pytest.raises(ValueError, match="Pose requires"):
        matrix(pose)


@pytest.mark.parametrize("attribute", ['xyz="nan 0 0"', 'xyz="0 0"', 'rpy="0 inf 0"'])
def test_invalid_urdf_origins_are_rejected(tmp_path, attribute):
    path = robot(tmp_path)
    path.write_text(path.read_text().replace('xyz="1 0 0" rpy="0 0 1.5707963267948966"', attribute))
    with pytest.raises(ValueError, match="finite xyz"):
        serial_chain(path, ["shoulder"])


def test_reflections_are_not_silently_converted_to_rotations():
    reflection = np.diag([-1, 1, 1, 1])
    with pytest.raises(ValueError, match="proper rotation"):
        matrix_pose(reflection)
