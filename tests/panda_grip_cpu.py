"""Isolated native Panda pads/actuators; no kitchen, renderer or task-success claim."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv


def check(root):
    asset = root / "third_party/robosuite/robosuite/models/assets/grippers/panda_gripper.xml"
    tree = ET.parse(asset).getroot()
    for mesh in tree.findall("asset/mesh"):
        mesh.set("file", str(asset.parent / mesh.get("file")))
    # Isolate finger-pad contact from unassembled hand-mesh self collisions.
    # Pad friction, actuator gains and force limits remain the native XML values.
    for geom in tree.findall(".//geom"):
        if "pad_collision" not in geom.get("name", ""):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    hand = tree.find("worldbody/body")
    hand.set("pos", "0 0 .5")
    hand.set("quat", "0 1 0 0")
    obj = ET.SubElement(tree.find("worldbody"), "body", name="test_object")
    ET.SubElement(obj, "freejoint")
    ET.SubElement(obj, "geom", name="cube", type="box", size=".015 .015 .008",
                  mass=".2", friction="1 .005 .0001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("finger1_pad_collision", "finger2_pad_collision")]
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand.get("name"))
    results = {}
    for case in ("wait", "stop", "stop_translate", "measured_stop"):
        model.body_pos[hand_id, 0] = 0
        data = mujoco.MjData(model)
        data.qpos[:2] = [.025, -.025]
        mujoco.mj_forward(model, data)
        data.qpos[2:5] = np.mean(data.geom_xpos[pads], axis=0)
        data.qpos[5:9] = [1, 0, 0, 0]
        data.ctrl[:] = 0
        # Establish contact before applying gravity; this is a controlled fixture.
        model.opt.gravity[:] = 0
        for _ in range(500):
            mujoco.mj_step(model, data)
        model.opt.gravity[:] = [0, 0, -9.81]
        for _ in range(1000):
            mujoco.mj_step(model, data)
        start = data.qpos[4]
        assert abs(start - .4066) < .02, "fixture must first establish a real grasp"
        gripper = SimpleNamespace(current_action=np.array([-1., 1.]))

        def set_goal(goal, data=data):
            low, high = model.actuator_ctrlrange.T
            data.ctrl[:] = low + (goal + 1) * (high - low) / 2

        native = RoboCasaEnv.__new__(RoboCasaEnv)
        native.env = SimpleNamespace(robots=[SimpleNamespace(
            gripper={"right": gripper},
            part_controllers={"right_gripper": SimpleNamespace(set_goal=set_goal)},
        )])
        native._stream_commands = {"right_gripper": np.array([1.])}
        if case.startswith("stop"):
            native.stop_part("right_gripper")
            native.stop_part("right_gripper")
            np.testing.assert_array_equal(data.ctrl, [0, 0])
        elif case == "measured_stop":
            data.ctrl[:] = data.qpos[:2]  # negative control: the former implementation
        for tick in range(1500):
            if case == "stop_translate":
                model.body_pos[hand_id, 0] = .005 * np.sin(tick / 1500 * 2 * np.pi)
            mujoco.mj_step(model, data)
        drop = float(start - data.qpos[4])
        results[case] = drop
        if case == "measured_stop":
            assert drop > .05, results
        else:
            assert abs(drop) < .015, results
            # An explicit open goal still releases the payload.
            set_goal(np.array([1., -1.]))
            for _ in range(500):
                mujoco.mj_step(model, data)
            assert start - data.qpos[4] > .05, results
    return results


if __name__ == "__main__":
    from pathlib import Path

    print(check(Path(__file__).resolve().parents[1]))
