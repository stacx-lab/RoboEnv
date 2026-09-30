"""D1 manufacturer torque mapping and uncapped arm velocity across owner surfaces."""

import json
import math
import sys
import xml.etree.ElementTree as ET
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from helpers import load_tool
from roboenv_runtime.adapters.isaaclab.go2_d1 import native

TORQUES = [3.3, 3.3, 1.7, 1.7, 1.7, 1.7]
BOUNDS = [2.35, 1.57, 1.57, 2.35, 1.57, 2.35]


def test_preparation_and_catalog_preserve_positions_and_use_motor_torques(tmp_path, monkeypatch):
    # No meshes are needed for this source fixture; preparation must not need a GPU.
    monkeypatch.setitem(sys.modules, "trimesh", ModuleType("trimesh"))
    source = tmp_path / "source"
    source.mkdir()
    root = ET.Element("robot", name="go2_d1")
    parent = "d1_base_link"
    for index, bound in enumerate(BOUNDS, 1):
        joint = ET.SubElement(root, "joint", name=f"d1_Joint{index}", type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        parent = f"d1_Link{index}"
        ET.SubElement(joint, "child", link=parent)
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower=str(-bound), upper=str(bound), effort="0", velocity="0")
    for name in ("d1_base_link", "wrist_d405_color_optical_frame", "head_d435_color_optical_frame"):
        joint = ET.SubElement(root, "joint", name=name + "_mount", type="fixed")
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=name)
    for name in ("wrist_d405_link", "head_d435_link"):
        link = ET.SubElement(root, "link", name=name)
        inertia = ET.SubElement(link, "inertial")
        ET.SubElement(inertia, "mass", value="0.1")
        ET.SubElement(inertia, "inertia", ixx="0", iyy="0", izz="0")
    ET.ElementTree(root).write(source / "go2_d1.urdf")
    (source / "preview_pose.json").write_text(json.dumps({f"d1_Joint{i}": 0 for i in range(1, 7)}))
    original = (source / "go2_d1.urdf").read_bytes()
    prepared = tmp_path / "prepared"
    prepare = load_tool("prepare_go2_d1").prepare
    prepare(source, prepared)
    assert (source / "go2_d1.urdf").read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare(source, prepared)
    robot = ET.parse(prepared / "robot.urdf").getroot()
    manifest = json.loads((prepared / "assets.json").read_text())
    for index, (bound, torque) in enumerate(zip(BOUNDS, TORQUES, strict=True), 1):
        limit = robot.find(f"joint[@name='d1_Joint{index}']/limit")
        actual = [float(limit.get(k)) for k in ("lower", "upper", "effort")]
        assert actual == [-bound, bound, torque]
        assert math.isinf(float(limit.get("velocity")))
        assert manifest["practice_assumptions"][f"d1_Joint{index}"]["velocity"] is None
    from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import RobotModel
    from roboenv_runtime.adapters.isaaclab.go2_d1.environment import load_profiles

    values = load_profiles()
    joints = values["arm"]["intrinsics"]["model"]["joints"]
    assert [joint["max_effort"] for joint in joints] == TORQUES
    assert all(joint["max_velocity"] is None for joint in joints)
    assert [[j["limits"]["min"], j["limits"]["max"]] for j in joints] == [
        [-bound, bound] for bound in BOUNDS
    ]
    # Match this test's straight, zero-offset synthetic robot; production YAML
    # retains its real geometry. Exercise native validation without a generator.
    arm = values["arm"]["intrinsics"]
    for link in arm["model"]["chain"]:
        link["origin"] = [0, 0, 0, 0, 0, 0, 1]
        link["axis"] = [0, 0, 1]
    for name in ("arm", "camera_head", "camera_wrist"):
        mount = values[name]["extrinsics"]["mount"]
        mount["translation_m"] = [0, 0, 0]
        mount["rotation_xyzw"] = [0, 0, 0, 1]
    model = RobotModel(prepared / "robot.urdf")
    home = json.loads((prepared / "home.json").read_text())
    native.validate_robot_profiles(values, model, home)
    values["arm"]["intrinsics"]["model"]["joints"][0]["limits"]["max"] += 0.1
    with pytest.raises(ValueError, match="d1_Joint1"):
        native.validate_robot_profiles(values, model, home)
    values["arm"]["intrinsics"]["model"]["joints"][0]["limits"]["max"] -= 0.1
    values["camera_wrist"]["extrinsics"]["mount"]["translation_m"][0] += 0.1
    with pytest.raises(ValueError, match="camera_wrist"):
        native.validate_robot_profiles(values, model, home)



def test_native_arm_overrides_imported_velocity_and_effort_limits(tmp_path, monkeypatch):
    # Execute the real native configuration path; stop before constructing physics.
    class Config(SimpleNamespace):
        func = staticmethod(lambda *args: None)

    Config.InitialStateCfg = Config
    Config.PDGainsCfg = Config
    sim = ModuleType("isaaclab.sim")
    for name in (
        "SimulationContext", "SimulationCfg", "GroundPlaneCfg", "RigidBodyMaterialCfg",
        "DomeLightCfg", "UrdfFileCfg", "ArticulationRootPropertiesCfg",
    ):
        setattr(sim, name, Config)
    sim.SimulationContext = lambda cfg: Config(cfg=cfg)
    sim.UrdfConverterCfg = SimpleNamespace(JointDriveCfg=Config)
    actuators = ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = Config
    assets = ModuleType("isaaclab.assets")
    assets.ArticulationCfg = Config
    captured = []

    class Configured(Exception):
        pass

    def capture(config):
        captured.append(config)
        raise Configured

    assets.Articulation = capture
    package = ModuleType("isaaclab")
    package.__path__ = []
    package.sim = sim
    for name, module in (
        ("isaaclab", package), ("isaaclab.sim", sim), ("isaaclab.actuators", actuators),
        ("isaaclab.assets", assets), ("torch", ModuleType("torch")),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    world = native.Native.__new__(native.Native)
    world.task_name = "go2-d1-control"
    world.home = {}
    world.model = SimpleNamespace(
        arm_names=[f"d1_Joint{i}" for i in range(1, 7)],
        center_of_mass=lambda home: np.zeros(3),
    )
    with pytest.raises(Configured):
        world._create(tmp_path, "cpu", False)
    arm = captured[0].actuators["arm"]
    assert arm.effort_limit_sim == dict(zip(world.model.arm_names, TORQUES, strict=True))
    assert math.isinf(arm.velocity_limit_sim) and arm.velocity_limit_sim > 0
    assert captured[0].actuators["fingers"].effort_limit_sim == 40
    assert captured[0].actuators["fingers"].velocity_limit_sim == 0.1
