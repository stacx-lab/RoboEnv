"""D1 manufacturer torque mapping and uncapped arm velocity across owner surfaces."""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import load_tool
from roboenv_runtime.adapters.isaaclab.go2_d1 import native

ROOT = Path(__file__).resolve().parents[1]
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
    catalog = tmp_path / "catalog"
    load_tool("generate_go2_d1_catalog").generate(prepared, catalog)
    relative = "robots/isaaclab/isaaclab-go2-d1/devices/arm/profile.yaml"
    generated = yaml.safe_load((catalog / relative).read_text())["intrinsics"]["model"]["joints"]
    checked_in = yaml.safe_load((ROOT / "catalog" / relative).read_text())
    checked_in = checked_in["intrinsics"]["model"]["joints"]
    assert generated == checked_in
    assert [joint["max_effort"] for joint in generated] == TORQUES
    assert all(joint["max_velocity"] is None for joint in generated)


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
