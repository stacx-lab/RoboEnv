"""Explicit SIMPLE gravity, tracking, stop and replay diagnostics; no task solution."""

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from roboenv_runtime.adapters.simple.backend import SimpleEnv
from roboenv_runtime.adapters.simple.profiles import profiles
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("left_arm", "right_arm")


def gravity_class():
    path = ROOT / "third_party/SIMPLE/src/simple/robots/arm_gravity.py"
    spec = importlib.util.spec_from_file_location("arm_gravity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArmGravityCompensation


def physics(facts, models, pose_error):
    import mujoco

    asset = ROOT / ".cache/simple/data/robots/g1_sonic/g1_29dof_with_hand.xml"
    model = mujoco.MjModel.from_xml_path(str(asset))
    live = mujoco.MjData(model)
    names = [j["name"] for n in ARMS for j in facts["parts"][n]["joints"]]
    body_ids = [
        i
        for i in range(model.njnt)
        if model.joint(i).name
        in {j["name"] for n in (*ARMS, "waist") for j in facts["parts"][n]["joints"]}
        or any(x in model.joint(i).name for x in ("hip", "knee", "ankle"))
    ]
    compensate = gravity_class()(model, names, body_ids)
    rng = np.random.default_rng(81)
    errors = []
    for _ in range(20):
        live.qpos[:] = model.qpos0
        live.qpos[3:7] = rng.normal(size=4)
        live.qpos[3:7] /= np.linalg.norm(live.qpos[3:7])
        for i in body_ids:
            live.qpos[model.jnt_qposadr[i]] = rng.uniform(*model.jnt_range[i])
        live.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, live)
        before = [
            x.copy()
            for x in (
                live.qpos,
                live.qvel,
                live.qacc,
                live.qfrc_bias,
                live.qacc_warmstart,
                live.ctrl,
                live.xpos,
            )
        ]
        torque = compensate(live).copy()
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = live.qpos
        mujoco.mj_forward(model, scratch)
        expected = np.zeros(len(body_ids))
        expected[compensate.slots] = scratch.qfrc_bias[compensate.dofs]
        errors.append(float(np.max(np.abs(torque - expected))))
        np.testing.assert_allclose(torque, expected, atol=1e-12)
        for old, new in zip(
            before,
            (
                live.qpos,
                live.qvel,
                live.qacc,
                live.qfrc_bias,
                live.qacc_warmstart,
                live.ctrl,
                live.xpos,
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(old, new)
    # Execute the actual native actuation method with a minimal robot fixture.
    tree = ast.parse((ROOT / "third_party/SIMPLE/src/simple/robots/g1_sonic.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1Sonic")
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply_action"
    )
    scope = {
        "np": np,
        "mujoco": mujoco,
        "ActionCmd": object,
        "WholeBodyEEFController": SimpleNamespace,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "native-actuation", "exec"), scope)
    robot = SimpleNamespace(
        sonic_config={"FREE_BASE": False},
        controller=SimpleNamespace(),
        num_body_dof=len(body_ids),
        num_hand_dof=0,
        body_joint_index=np.array(body_ids),
        qpos_offset=7,
        qvel_offset=6,
        mjData=live,
        torques=np.zeros(model.nu),
        torque_limit=np.full(model.nu, 2.0),
    )
    target = live.qpos[model.jnt_qposadr[body_ids]] + 0.1

    class Action(dict):
        type = "decoupled_wbc"

    pd = 100 * (target - live.qpos[model.jnt_qposadr[body_ids]])
    pd -= 5 * live.qvel[model.jnt_dofadr[body_ids]]
    for enabled in (False, True):
        robot.arm_gravity = compensate if enabled else None
        expected = np.zeros(model.nu)
        expected[np.array(body_ids) - 1] = pd + (compensate(live) if enabled else 0)
        scope["apply_action"](robot, Action(target_q=target))
        np.testing.assert_array_equal(live.ctrl, np.clip(expected, -2, 2))

    # Fixed torso rig: retain native arm/hand dynamics, remove lower-body motion.
    xml = ET.parse(asset).getroot()
    xml.find("compiler").set("meshdir", str(asset.parent / "meshes"))
    retained = {
        j["name"] for n in (*ARMS, "left_hand", "right_hand") for j in facts["parts"][n]["joints"]
    }
    for body in xml.iter("body"):
        for node in list(body):
            if node.tag in {"joint", "freejoint"} and node.get("name") not in retained:
                body.remove(node)
    for section in ("sensor", "actuator"):
        for node in list(xml.find(section)):
            if node.get("joint") and node.get("joint") not in retained:
                xml.find(section).remove(node)
    ET.SubElement(xml, "option", timestep="0.005")
    fixed = {}
    for gravity, enabled in ((9.81, False), (9.81, True), (0.0, True)):
        m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        m.opt.gravity[:] = [0, 0, -gravity]
        d = mujoco.MjData(m)
        ids = [m.joint(n).id for n in names]
        ff = gravity_class()(m, names, ids)
        qa, da = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        act = [int(np.where(m.actuator_trnid[:, 0] == i)[0][0]) for i in ids]
        goal = np.array([-0.25, 0.2, 0, 0.2, 0, 0, 0, -0.25, -0.2, 0, 0.2, 0, 0, 0])
        d.qpos[qa] = goal
        kp, kd = np.tile([100, 100, 40, 40, 20, 20, 20], 2), np.tile([5, 5, 2, 2, 2, 2, 2], 2)
        hands = [i for i in range(m.njnt) if i not in ids]
        ha = [int(np.where(m.actuator_trnid[:, 0] == i)[0][0]) for i in hands]
        hkp = np.array([5 if "thumb" in m.joint(i).name else 2.5 for i in hands])
        mujoco.mj_forward(m, d)
        torso = d.xpos[m.body("torso_link").id].copy()
        tail = []
        for i in range(1200):
            torque = kp * (goal - d.qpos[qa]) - kd * d.qvel[da]
            if enabled:
                torque += ff(d)
            d.ctrl[act] = np.clip(torque, *m.actuator_ctrlrange[act].T)
            d.ctrl[ha] = -hkp * d.qpos[m.jnt_qposadr[hands]] - d.qvel[m.jnt_dofadr[hands]]
            mujoco.mj_step(m, d)
            mujoco.mj_kinematics(m, d)
            np.testing.assert_array_equal(torso, d.xpos[m.body("torso_link").id])
            if i >= 800:
                tail.append(
                    [
                        pose_error(
                            models[n].fk(goal[j * 7 : (j + 1) * 7].tolist())["pose"],
                            models[n].solver.forward(d.qpos[qa[j * 7 : (j + 1) * 7]])[0],
                        )
                        for j, n in enumerate(ARMS)
                    ]
                )
        fixed[f"gravity={gravity},compensation={enabled}"] = {
            n: {
                key: float(np.mean([s[j][key] for s in tail]))
                for key in ("position_m", "rotation_rad")
            }
            for j, n in enumerate(ARMS)
        }
    for n in ARMS:
        off = fixed["gravity=9.81,compensation=False"][n]
        on = fixed["gravity=9.81,compensation=True"][n]
        assert on["position_m"] < off["position_m"] * 0.2
        assert on["rotation_rad"] < off["rotation_rad"] * 0.2
        assert fixed["gravity=0.0,compensation=True"][n]["position_m"] < 1e-10
    return {"max_force_error_nm": max(errors), "configurations": 20, "fixed_torso": fixed}


def native(enabled, out, models, pose_error):
    world = SimpleEnv(sim_mode="mujoco", arm_gravity_compensation=enabled)
    try:
        world.reset(0)
        assert (world.robot.arm_gravity is not None) is enabled
        assert world.robot.sonic_config["ARM_GRAVITY_COMPENSATION"] is enabled
        goals = {}
        for n in ARMS:
            target = list(world.control.targets[n])
            target[0] -= 0.25
            target[3] += 0.2
            goals[n] = models[n].fk(target)["pose"]
            world.control.write(n, {"target": target})
        errors = []
        for _ in range(300):
            world.advance(world.control.targets)
            errors.append(
                {
                    n: pose_error(goals[n], world.read_device(n)["frame_poses"]["flange"])
                    for n in ARMS
                }
            )
        before_stop = {n: world.read_device(n)["frame_poses"]["flange"] for n in ARMS}
        for n in ARMS:
            world.control.stop(n)
        drift = {n: [] for n in ARMS}
        for _ in range(100):
            world.advance(world.control.targets)
            for n in ARMS:
                drift[n].append(
                    pose_error(before_stop[n], world.read_device(n)["frame_poses"]["flange"])[
                        "position_m"
                    ]
                )
        result = {
            n: {
                **{
                    key: float(np.mean([s[n][key] for s in errors[-100:]]))
                    for key in ("position_m", "rotation_rad")
                },
                "max_post_stop_drift_m": max(drift[n]),
            }
            for n in ARMS
        }
        # Exercise changing dynamics without conflating world and local frame error.
        dynamic = []
        for velocity, posture in (
            ([0.1, 0, 0.1], [0.74, 0, 0, 0]),
            ([0, 0, 0], [0.7, 0.03, 0.05, 0.08]),
        ):
            world.control.write(
                "base",
                {
                    "mode": "locomotion",
                    "velocity": dict(zip(("vx", "vy", "wz"), velocity, strict=True)),
                    "posture": dict(zip(("height", "roll", "pitch", "yaw"), posture, strict=True)),
                },
            )
            initial = matrix(world._pose("torso_link"))
            for _ in range(50):
                world.advance(world.control.targets)
            torso = matrix(world._pose("torso_link"))
            dynamic.append(
                {
                    n: {
                        "local": pose_error(
                            goals[n], world.read_device(n)["frame_poses"]["flange"]
                        ),
                        "frame_drift": pose_error(
                            matrix_pose(initial @ matrix(goals[n])),
                            matrix_pose(torso @ matrix(goals[n])),
                        ),
                    }
                    for n in ARMS
                }
            )
        (out / f"record-{enabled}.json").write_text(json.dumps(world.record()) + "\n")
        return {"arms": result, "dynamic": dynamic}
    finally:
        world.close()


def main(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kinex / "python"))
    from kinex.joints import JointDevice
    from kinex.motion import pose_error

    facts = json.loads((ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json").read_text())
    models = {n: JointDevice(profiles(facts)[n]) for n in ARMS}
    if args.native_setting is not None:
        result = native(args.native_setting == "on", args.out, models, pose_error)
        (args.out / "native.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    result = {"physics": physics(facts, models, pose_error)}
    for enabled in (False, True):
        # Upstream registries cache robot instances/configuration in a process.
        # Each condition uses a fresh worker, matching the service lifecycle.
        directory = args.out / ("on" if enabled else "off")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--out",
                str(directory),
                "--kinex",
                str(args.kinex),
                "--native-setting",
                "on" if enabled else "off",
            ],
            check=True,
        )
        result[str(enabled)] = json.loads((directory / "native.json").read_text())
    (args.out / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    for n in ARMS:
        on, off = result["True"]["arms"][n], result["False"]["arms"][n]
        assert on["position_m"] < off["position_m"] * 0.2, result
        assert on["rotation_rad"] < off["rotation_rad"] * 0.2, result
        assert on["max_post_stop_drift_m"] < 0.005, result
    (args.out / "report.json").write_text(json.dumps({"passed": True, **result}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinex", type=Path, default=ROOT.parent / "kinex")
    parser.add_argument("--native-setting", choices=("on", "off"), help=argparse.SUPPRESS)
    main(parser.parse_args())
