"""Focused ABC native checks in its isolated interpreter; no model or task-solver calls."""

import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.abc.backend import ABCEnv
from roboenv_runtime.adapters.abc.scoring import BlockTower, MarkersInDrawer, geom_points


def scoring_checks():
    import abc_sim
    import mujoco

    env = abc_sim.make_env(task="put_markers_in_top_drawer", render_cameras=False)
    try:
        env.reset(seed=2)
        evaluator = MarkersInDrawer(env.model, "top")
        assert not evaluator.evaluate(env.data)["success"]
        rotation = env.data.xmat[evaluator.drawer].reshape(3, 3)
        origin = env.data.xpos[evaluator.drawer].copy()
        for i, (body, geoms) in enumerate(
            zip(evaluator.markers, evaluator.marker_geoms, strict=True)
        ):
            points = np.concatenate([geom_points(env.model, env.data, g) for g in geoms])
            local = (points - env.data.xpos[body]) @ env.data.xmat[body].reshape(3, 3)
            _, axes = np.linalg.eigh(np.cov(local.T))
            axes = axes[:, ::-1].T
            if np.linalg.det(axes) < 0:
                axes[-1] *= -1
            aligned = local @ axes.T
            center = (aligned.min(0) + aligned.max(0)) / 2
            half = (aligned.max(0) - aligned.min(0)) / 2
            desired = np.array([0.0, -0.065 + i * 0.028, evaluator.lower[2] + half[2] + 0.001])
            position = origin + rotation @ (desired - center)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rotation @ axes).ravel())
            address = env.model.jnt_qposadr[env.model.body_jntadr[body]]
            env.data.qpos[address : address + 7] = [*position, *quat]
        mujoco.mj_forward(env.model, env.data)
        assert evaluator.condition(env.data)[0], evaluator.condition(env.data)
        evaluator.evaluate(env.data)
        env.data.time += 1.01
        assert evaluator.evaluate(env.data)["success"]
        env.data.qpos[evaluator.joint.qposadr[0]] = 0.02
        mujoco.mj_forward(env.model, env.data)
        assert not evaluator.evaluate(env.data)["success"]
        env.data.qpos[evaluator.joint.qposadr[0]] = 0
        env.data.qpos[env.model.jnt_qposadr[env.model.body_jntadr[evaluator.markers[0]]]] += 0.4
        mujoco.mj_forward(env.model, env.data)
        assert not evaluator.condition(env.data)[0]
        assert not MarkersInDrawer(env.model, "middle").condition(env.data)[0]
    finally:
        env.close()
    # A native contact fixture distinguishes vertical support from nearby objects.
    for count in (4, 5):
        blocks = "".join(
            f'<body name="b{i}" pos="{0 if i < count else 0.2 + (i - count) * 0.05} 0 '
            f'{0.02 + i * 0.04 if i < count else 0.02}"><freejoint/><geom type="box" '
            'size=".02 .02 .02" mass=".05"/></body>'
            for i in range(9)
        )
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002"/>'
            '<worldbody><body name="left_arm"/><body name="right_arm"/>'
            '<geom name="table_plane" type="plane" size="1 1 .1"/>'
            + blocks
            + "</worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        evaluator = BlockTower(model)
        result = None
        for _ in range(1200):
            mujoco.mj_step(model, data)
            result = evaluator.evaluate(data)
        assert result["success"] == (count == 5), result
        if count == 5:
            data.qpos[model.jnt_qposadr[model.body_jntadr[model.body("b4").id]]] += 0.2
            mujoco.mj_forward(model, data)
            assert not evaluator.evaluate(data)["success"]
    return {"markers_positive_negative": True, "tower_positive_negative": True}


def contact_check():
    import mujoco
    from abc_sim.config import _base_config
    from roboenv_runtime.adapters.abc.backend import SOURCE

    spec = mujoco.MjSpec.from_file(str(SOURCE / "abc_sim/models/yam_drawer_scene.xml"))
    body = spec.worldbody.add_body(name="contact_fixture", pos=[0, 0, 2])
    body.add_freejoint(name="fixture_joint")
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.014, 0.014, 0.016],
        mass=0.04,
        friction=[2, 0.1, 0.01],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    for side, config in _base_config().robots.items():
        for i, q in enumerate(config.init_q[:6], 1):
            data.qpos[model.joint(f"{side}_joint{i}").qposadr[0]] = q
            data.ctrl[model.actuator(f"{side}_joint{i}").id] = q
        data.qpos[model.joint(side + "_left_finger").qposadr[0]] = 0.015
        data.qpos[model.joint(side + "_right_finger").qposadr[0]] = -0.015
    mujoco.mj_forward(model, data)
    address = model.joint("fixture_joint").qposadr[0]
    data.qpos[address : address + 3] = data.site_xpos[model.site("left_grasp_site").id]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.xmat[model.body("left_link_6").id])
    data.qpos[address + 3 : address + 7] = quat
    mujoco.mj_forward(model, data)
    start = data.qpos[address : address + 3].copy()
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.qpos[address : address + 3] - start) < 0.01
    data.ctrl[model.actuator("left_gripper").id] = 0.0475
    for _ in range(250):
        mujoco.mj_step(model, data)
    assert data.qpos[address + 2] < start[2] - 0.05
    return True


def check():
    from roboenv.provisioning.gpu import configure_gpu

    configure_gpu("abc")
    result = scoring_checks()
    result["loaded_contact_and_release"] = contact_check()
    world = ABCEnv("put_plastic_bottles_in_bin")
    try:
        world.reset(seed=0)
        initial = world.owner_state()
        assert world.frame_limit == 8823
        before = world.owner_state()
        for value in ([float("nan")] * 6, [0.0] * 5):
            try:
                world.device_write("left_arm", value)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid motion accepted")
        assert world.owner_state() == before
        goal = world.device_read("left_arm")["positions"]
        goal[0] += 0.05
        peer = world.control.targets["right_arm"].copy()
        world.device_write("left_arm", goal)
        for _ in range(20):
            world.control.tick()
        assert abs(world.device_read("left_arm")["positions"][0] - goal[0]) < 0.01
        assert world.control.targets["right_arm"] == peer
        world.device_write("left_gripper", [0.3])
        world.device_stop("left_gripper")
        assert world.control.targets["left_gripper"] == [0.3]
        result.update(motion=True, task_success=world.task_completed())
        world.reset(seed=0)
        assert world.control.frame == 0
        assert world.owner_state() == initial
        result["reset"] = True
    finally:
        world.close()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
