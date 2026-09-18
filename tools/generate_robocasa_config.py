"""Pre-generate task-independent Panda–Omron profiles; no kitchen, render, or step.

Run with the prepared RoboCasa interpreter. This is a development command, never
called by launch. Facts come from the pinned robot/controller and camera sources.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml
from robocasa.utils.camera_utils import get_robot_cam_configs
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.export import export_devices, write_driver
from roboenv_runtime.adapters.robocasa.backend import DEFAULT_CAMERAS, joint_controller_config
from roboenv_runtime.bindings import default_registry
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.utils.binding_utils import MjSim

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "catalog/robots/robocasa/robocasa-panda-omron"


def descriptors():
    robot = ROBOT_CLASS_MAPPING["PandaOmron"](
        "PandaOmron",
        composite_controller_config=joint_controller_config("PandaOmron"),
        initialization_noise=None,
        control_freq=20,
    )
    robot.load_model()
    model = ManipulationTask(EmptyArena(), [robot.robot_model])
    xml = ET.fromstring(model.get_xml())
    # Apply the same robot-mounted camera definitions as Kitchen.edit_model_xml.
    for name, config in get_robot_cam_configs("PandaOmron").items():
        parent = xml.find(f".//body[@name='{config['parent_body']}']")
        if parent is None:
            raise ValueError(f"Missing camera parent: {config['parent_body']}")
        camera = parent.find(f".//camera[@name='{name}']")
        if camera is None:
            camera = ET.SubElement(parent, "camera", name=name, mode="fixed")
        camera.set("pos", " ".join(map(str, config["pos"])))
        camera.set("quat", " ".join(map(str, config["quat"])))
        camera.attrib.update(config.get("camera_attribs", {}))
    sim = MjSim.from_xml_string(ET.tostring(xml, encoding="unicode"))
    try:
        robot.reset_sim(sim)
        robot.setup_references()
        robot.reset(deterministic=True)
        sim.forward()
        native = SimpleNamespace(
            realtime=True,
            _robot=robot,
            sim=sim,
            env=SimpleNamespace(control_freq=20),
            _render=True,
            _camera_names=DEFAULT_CAMERAS,
            _camera_size=(256, 256),
        )
        adapter = SimpleNamespace(_env=native)
        plan = BindingPlan.load(ROBOT, backend="robocasa")
        service = default_registry().assemble(
            plan, ROBOT, BindingContext("robocasa", "panda_omron", adapter)
        )
        values = service.inventory()["devices"]
        # The facts must not change when the robot changes pose in a task.
        sim.data.qpos[robot._ref_joint_pos_indexes] += 0.01
        sim.data.qpos[robot._ref_torso_joint_pos_indexes] = 0.1
        sim.forward()
        other = default_registry().assemble(
            plan, ROBOT, BindingContext("robocasa", "panda_omron", adapter)
        )
        for before, after in zip(values, other.inventory()["devices"], strict=True):
            assert before["intrinsics"] == after["intrinsics"]
            mount = before["extrinsics"].get("mount")
            other_mount = after["extrinsics"].get("mount")
            if mount is None:
                assert before["extrinsics"] == after["extrinsics"]
                continue
            assert mount["parent"] == other_mount["parent"]
            assert mount["parent_frame"] == other_mount["parent_frame"]
            for key in ("translation_m", "rotation_xyzw"):
                np.testing.assert_allclose(mount[key], other_mount[key], atol=1e-10)
        return values
    finally:
        sim.free()


def canonical(value):
    if isinstance(value, float):
        return round(value, 12) or 0.0
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def generate(out, provenance=None):
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        export_devices(descriptors(), temporary, {"endpoint": "http://127.0.0.1:8123"})
        for path in sorted(Path(temporary).glob("*/profile.yaml")):
            profile = canonical(yaml.safe_load(path.read_text()))
            if path.parent.name == "world":
                # This catalog imposes no additional global workspace AABB.
                # It does not claim the kitchen is empty or collision-free.
                profile["intrinsics"].update(
                    source="default",
                    bounds=None,
                    method="unknown",
                    calibrated_at=None,
                )
            profile["meta"] = {
                "created_at": "2026-09-06T00:00:00Z",
                "updated_at": "2026-09-08T00:00:00Z",
            }
            destination = out / "devices" / path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
            if profile["type"] != "world":
                write_driver(destination, profile["type"])
            # There is deliberately no task or session connection file.
    (out / "workspace.yaml").write_bytes((ROBOT / "workspace.yaml").read_bytes())
    if provenance is None:
        return
    sources = [
        "third_party/robocasa/robocasa/utils/camera_utils.py",
        "third_party/robocasa/robocasa/environments/kitchen/kitchen.py",
        "third_party/robosuite/robosuite/models/robots/compositional.py",
        "third_party/robosuite/robosuite/models/assets/robots/panda/robot.xml",
        "third_party/robosuite/robosuite/models/assets/bases/omron_mobile_base.xml",
        "third_party/robosuite/robosuite/models/assets/grippers/panda_gripper.xml",
        "third_party/robosuite/robosuite/controllers/config/robots/default_pandaomron.json",
        "third_party/robosuite/robosuite/controllers/config/default/parts/joint_position.json",
        "third_party/robosuite/robosuite/controllers/parts/generic/joint_pos.py",
        "third_party/robosuite/robosuite/controllers/parts/mobile_base/joint_vel.py",
        "third_party/robosuite/robosuite/models/grippers/panda_gripper.py",
        "third_party/robosuite/robosuite/utils/camera_utils.py",
        "runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py",
        "runtime/src/roboenv_runtime/adapters/robocasa/geometry.py",
        "runtime/src/roboenv_runtime/adapters/robocasa/base.py",
        "src/roboenv/assets/_roboenv.py",
        "src/roboenv/assets/joint-client.py",
        "runtime/src/roboenv_runtime/bindings/backends/robocasa/joint.py",
        "src/roboenv/assets/camera-client.py",
        "src/roboenv/assets/gripper-client.py",
        "src/roboenv/assets/velocity-client.py",
        "runtime/src/roboenv_runtime/bindings/backends/robocasa/velocity.py",
        "runtime/src/roboenv_runtime/bindings/backends/robocasa/gripper.py",
        "runtime/src/roboenv_runtime/bindings/backends/robocasa/camera.py",
        "src/roboenv/devices/export.py",
        "runtime/src/roboenv_runtime/adapters/robocasa/depth.py",
        "runtime/src/roboenv_runtime/adapters/robocasa/backend.py",
    ]
    provenance.write_text(
        json.dumps(
            {
                "generator": "tools/generate_robocasa_config.py",
                "method": "CPU model assembly; zero physics steps; no task",
                "robot": "PandaOmron",
                "cameras": list(DEFAULT_CAMERAS),
                "camera_size": [256, 256],
                "control_frequency_hz": 20,
                "randomize_cameras": False,
                "sources": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROBOT)
    parser.add_argument(
        "--provenance", type=Path, help="Optional source receipt outside the robot directory"
    )
    args = parser.parse_args()
    generate(args.out, args.provenance)
