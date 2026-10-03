"""Go2 controller and portable device contracts without an Isaac installation."""

import copy
import itertools
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import (
    COMMAND_FIELDS,
    DEFAULT,
    RobotModel,
    Trot,
    leg_ik,
    matrix_quat,
    quat_matrix,
    rotation,
    validate_command,
)
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "catalog/robots/isaaclab/isaaclab-go2-d1"


def command(**changes):
    values = dict(zip(COMMAND_FIELDS, DEFAULT, strict=True))
    values.update(changes)
    return {
        "mode": "locomotion",
        "velocity": {k: float(values[k]) for k in ("vx", "vy", "wz")},
        "posture": {k: float(values[k]) for k in ("height", "roll", "pitch")},
    }


@pytest.mark.parametrize("existing", [None, "YES", "NO", ""])
def test_native_defaults_eula_before_isaac_import(monkeypatch, tmp_path, existing):
    from roboenv_runtime.adapters.isaaclab.go2_d1 import native

    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    if existing is not None:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", existing)
    expected = "YES" if existing is None else existing
    (tmp_path / "home.json").write_text("{}")
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema": "roboenv/go2-d1-assets/1", "files": {}}))
    monkeypatch.setattr(native, "RobotModel", lambda path: object())
    monkeypatch.setattr(native, "validate_robot_profiles", lambda *args: None)
    monkeypatch.setattr(native.Native, "_create", lambda *args: None)

    class AppModule(ModuleType):
        def __getattr__(self, name):
            if name == "AppLauncher":
                assert os.environ.get("OMNI_KIT_ACCEPT_EULA") == expected
                return lambda **kwargs: SimpleNamespace(app=object())
            raise AttributeError(name)

    package = ModuleType("isaaclab")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", AppModule("isaaclab.app"))

    world = native.Native(manifest)
    assert world.app is not None


@pytest.mark.parametrize("key", COMMAND_FIELDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0", None])
def test_command_rejects_bad_values(key, bad):
    args = command()
    args["velocity" if key in ("vx", "vy", "wz") else "posture"][key] = bad
    with pytest.raises(ValueError):
        validate_command(args)


@pytest.mark.parametrize("key", COMMAND_FIELDS)
@pytest.mark.parametrize("value", [-99.0, 99.0])
def test_base_accepts_finite_commands_without_range_limits(key, value):
    result = validate_command(command(**{key: value}))
    assert result[COMMAND_FIELDS.index(key)] == value


def test_command_requires_all_fields():
    args = command()
    del args["posture"]["pitch"]
    with pytest.raises(ValueError):
        validate_command(args)


@pytest.mark.parametrize("side", [-1, 1])
def test_analytic_leg_inverse_matches_three_joint_geometry(side):
    for x, y, z in itertools.product([-0.05, 0.04], [side * 0.0955, side * 0.12], [-0.26, -0.33]):
        hip, thigh, calf = leg_ik([x, y, z], side)
        actual = rotation([1, 0, 0], hip) @ (
            np.array([0, side * 0.0955, 0])
            + rotation([0, 1, 0], thigh)
            @ (np.array([0, 0, -0.213]) + rotation([0, 1, 0], calf) @ [0, 0, -0.213])
        )
        assert np.allclose(actual, [x, y, z], atol=1e-8)
    with pytest.raises(ValueError):
        leg_ik([1, 0, -0.3], side)


def test_gait_feedback_and_output_scope():
    gait = Trot()
    cmd = DEFAULT.copy()
    cmd[0] = 0.05
    for _ in range(200):
        target = gait.targets(cmd, 0.005, DEFAULT[3:], [0, 0, 0])
    assert target.shape == (12,)
    assert np.isfinite(target).all()
    assert gait.velocity_integral[0] > 0
    assert np.allclose(gait.command, cmd)


@pytest.mark.parametrize("task", ["control", "reach", "pick-place"])
def test_complete_catalog_preserves_profiles(task):
    resolved = resolve_task(f"isaaclab/go2-d1-{task}", project=ROOT)
    assert resolved.task.backend_kwargs["task_name"] == f"go2-d1-{task}"
    assert resolved.task.execution_mode == "realtime"
    assert "realtime" not in resolved.task.backend_kwargs
    assert "command_timeout_s" not in resolved.task.backend_kwargs
    assert resolved.task.backend_kwargs["max_steps"] == 6000
    specs = {s.name: s for s in resolved.bindings.devices}
    assert set(specs) == {"base", "arm", "gripper", "camera_head", "camera_wrist", "world"}
    for name in ("base", "arm"):
        profile = specs[name].resolved_profile(resolved.robot)
        # Resolving a task preserves the complete catalog resources.
        source = yaml.safe_load((RIG / f"devices/{name}/profile.yaml").read_text())
        assert profile == source
    wrist = specs["camera_wrist"].resolved_profile(resolved.robot)
    assert wrist["extrinsics"]["mount"]["parent"] == "arm"
    assert wrist["extrinsics"]["mount"]["parent_frame"] == "flange"


def test_rotation_pi_and_arm_ik(tmp_path):
    # A small synthetic chain exercises the solver without licensed asset files.
    root = ET.Element("robot", name="fixture")
    parent = "base"
    joint = ET.SubElement(root, "joint", name="mount", type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link="d1_base_link")
    parent = "d1_base_link"
    for i, axis in enumerate(["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"], 1):
        joint = ET.SubElement(root, "joint", name=f"d1_Joint{i}", type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        parent = f"d1_Link{i}"
        ET.SubElement(joint, "child", link=parent)
        ET.SubElement(joint, "origin", xyz=".08 0 .04", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(joint, "limit", lower="-2", upper="2")
    path = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(path)
    model = RobotModel(path)
    seed = np.array([0.1, -0.5, 0.8, 0.2, -0.2, 0.1])
    desired = model.arm_fk(seed + 0.04)
    target = [*desired[:3, 3], *matrix_quat(desired[:3, :3])]
    q = model.arm_ik(target, seed)
    assert np.allclose(model.arm_fk(q), desired, atol=0.005)
    with pytest.raises(ValueError):
        model.arm_ik([10, 10, 10, 0, 0, 0, 1], seed, iterations=3)
    for axis in np.eye(3):
        r = rotation(axis, np.pi)
        assert np.allclose(quat_matrix(matrix_quat(r)), r)


def test_invalid_write_and_stop_do_not_step_or_change_other_goals():
    env = Go2Env.__new__(Go2Env)
    profile = yaml.safe_load((RIG / "devices/arm/profile.yaml").read_text())
    joints = profile["intrinsics"]["model"]["joints"]
    env.native = SimpleNamespace(
        steps=0,
        done=False,
        model=SimpleNamespace(
            arm_names=[j["name"] for j in joints],
            joints={
                j["name"]: {"limits": [j["limits"]["min"], j["limits"]["max"]]} for j in joints
            },
        ),
    )
    env.max_steps = 20
    env.targets = {"base": DEFAULT.copy(), "arm": np.zeros(6), "gripper": np.zeros(2)}
    env.events = []
    before = copy.deepcopy(env.targets)
    with pytest.raises(ValueError):
        env.set_target("arm", {"target": [20.0] * 6})
    assert not env.events and env.steps == 0
    assert all(np.array_equal(env.targets[n], q) for n, q in before.items())
    env.set_target("base", command(vx=0.05))
    assert env.targets["base"][0] == 0.05
    env.stop("base")
    assert env.steps == 0 and env.targets["base"][0] == 0
    assert np.array_equal(env.targets["arm"], before["arm"])
    assert np.array_equal(env.targets["gripper"], before["gripper"])


def test_stop_holds_measured_overshoot_inside_limits():
    env = Go2Env.__new__(Go2Env)
    profile = yaml.safe_load((RIG / "devices/arm/profile.yaml").read_text())
    limits = {
        j["name"]: [j["limits"]["min"], j["limits"]["max"]]
        for j in profile["intrinsics"]["model"]["joints"]
    }
    names = list(limits)
    env.native = SimpleNamespace(
        steps=0,
        model=SimpleNamespace(
            arm_names=names, joints={n: {"limits": b} for n, b in limits.items()}
        ),
    )
    env.targets = {"arm": np.zeros(6)}
    env.events = []
    measured = [0.0, limits[names[1]][1] + 1.6e-5, 0.1, 0.0, -0.5, 0.0]
    env.read_device = lambda name: {"positions": measured}
    env.stop("arm")
    assert env.targets["arm"][1] == limits[names[1]][1]
    assert np.array_equal(np.delete(env.targets["arm"], 1), np.delete(measured, 1))


def test_gripper_read_needs_no_joint_feedback():
    env = Go2Env.__new__(Go2Env)
    env.native = SimpleNamespace(steps=5, task=SimpleNamespace(failure=None))
    assert env.read_device("gripper") == {"aperture_m": None, "faults": [], "timestamp": 0.1}
    with pytest.raises(ValueError, match="Unknown actuator"):
        env.read_device("unknown")


def test_camera_observation_helper_is_importable():
    from roboenv_runtime.adapters.isaaclab.observations import camera_state

    assert callable(camera_state)


def test_center_of_mass_includes_fixed_payload_and_origin(tmp_path):
    path = tmp_path / "mass.urdf"
    path.write_text("""<robot name="mass">
      <link name="base"><inertial><mass value="1"/>
        <origin xyz="0 0 0"/></inertial></link>
      <link name="payload"><inertial><mass value="3"/>
        <origin xyz="0.1 0 0"/></inertial></link>
      <joint name="mount" type="fixed"><parent link="base"/>
        <child link="payload"/><origin xyz="0.3 0 0" rpy="0 0 0"/></joint>
    </robot>""")
    assert np.allclose(RobotModel(path).center_of_mass({}), [0.3, 0, 0])
    path.write_text('<robot name="empty"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="mass"):
        RobotModel(path).center_of_mass({})


def test_high_speed_gait_retains_independent_posture_and_leg_scope():
    gait = Trot(support_center=[0.01, -0.005])
    target = np.array([1.0, 0.5, np.pi / 6, 0.32, 0.06, -0.06])
    for _ in range(600):
        joints = gait.targets(target, 0.005, target[3:], target[:3], [0.0, 0.0, 0.0])
        assert joints.shape == (12,)
        assert np.isfinite(joints).all()
    assert np.allclose(gait.command, target)
    assert np.allclose(gait.support_center, [0.01, -0.005])
