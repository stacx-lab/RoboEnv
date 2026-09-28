"""Real CPU control/frame probe, also usable as an evidence-producing command."""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.tasks.libero import ROBOTS
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("libero_spatial", 0),
    ("libero_object", 0),
    ("libero_10", 0),
    ("libero_10", 2),
    ("libero_10", 5),
    ("libero_90", 8),
    ("libero_10", 9),
    ("libero_10", 3),
]


def probe(suite, task, init, directory, render=False):
    sys.path.insert(0, str(Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")) / "python"))
    from kinex.joints import JointDevice

    adapter = LiberoAdapter(suite, task, init_state_index=init, enable_render=render)
    adapter.run_dir = directory
    try:
        adapter.reset(0)
        world = adapter._env
        root = ROOT / "catalog/robots" / ROBOTS[world.wrapper.problem_name.lower()]
        plan = BindingPlan.load(root, backend="libero")
        if not render:
            plan = replace(plan, devices=tuple(s for s in plan.devices if s.entity != "camera"))
        from roboenv_runtime.runtime.world import WorldSession

        session = WorldSession(adapter, mode="stepped")
        service = default_registry().assemble(
            plan,
            root,
            BindingContext("libero", "panda", adapter, world=session),
            motion_scheduler=session.schedule,
            resolve=session.resolve,
        )

        def call(name, action, **args):
            return service.call({"device": name, "action": action, "args": args})["data"]

        initial = call("arm", "read")
        assert adapter.budget().used == 5
        model = JointDevice(
            {"name": "arm", "intrinsics": service.describe("arm")["intrinsics"], "type": "joint"}
        )
        state = world.sim.get_state()
        fk_errors = []
        for shift in (0, 0.02, -0.015):
            q = np.array(initial["positions"])
            q[[0, 2, 4]] += shift
            world.sim.data.qpos[world._robot._ref_joint_pos_indexes] = q
            world.sim.forward()
            actual = call("arm", "read")["frame_poses"]["flange"]
            expected = model.fk(q.tolist())["pose"]
            fk_errors.append(float(np.max(np.abs(np.array(actual[:3]) - expected[:3]))))
            assert fk_errors[-1] < 1e-09
            assert abs(np.dot(actual[3:], expected[3:])) > 1 - 1e-09
        world.sim.set_state(state)
        world.sim.forward()
        q = initial["positions"]
        answer = model.ik(model.fk(q)["pose"], seed=q)["joints"]
        assert np.allclose(answer, q, atol=0.0001)
        before = world.sim.get_state().flatten().copy()
        before_goal = world._robot.controller.goal_qpos.copy()
        for target in ([0] * 6, [True] * 7, [float("nan")] * 7, [float("inf")] * 7, [100] * 7):
            try:
                call("arm", "write", actions=[target], duration_s=len([target]) / 20)
            except Exception:
                pass
            else:
                raise AssertionError("Invalid target accepted")
            assert np.array_equal(world.sim.get_state().flatten(), before)
            assert np.array_equal(world._robot.controller.goal_qpos, before_goal)
        goal = np.array(q)
        goal[[0, 2, 4]] += [0.1, -0.02, 0.02]
        assert call("arm", "write", actions=[goal.tolist()], duration_s=0.05)["execution_id"]
        same_goal = world._robot.controller.goal_qpos
        call("arm", "write", actions=[goal.tolist()], duration_s=len([goal.tolist()]) / 20)
        assert world._robot.controller.goal_qpos is same_goal
        call("world", "wait", duration_s=5)
        measured = call("arm", "read")
        error = float(np.max(np.abs(np.array(measured["positions"]) - goal)))
        assert error < 0.01, (suite, task, error)
        call("gripper", "write", actions=[1], duration_s=1)
        call("world", "wait", duration_s=1)
        closed = call("gripper", "read")["positions"]
        assert np.array_equal(world._robot.controller.goal_qpos, goal)
        call("gripper", "write", actions=[-1], duration_s=1)
        call("world", "wait", duration_s=1)
        opened = call("gripper", "read")["positions"]
        assert abs(opened[0] - opened[1]) > abs(closed[0] - closed[1]) + 0.01
        call("gripper", "stop")
        call("arm", "write", actions=[q], duration_s=len([q]) / 20)
        call("world", "wait", duration_s=0.15)
        stop_q = call("arm", "read")["positions"]
        budget = adapter.budget().used
        call("arm", "stop")
        assert adapter.budget().used == budget
        assert np.allclose(world._robot.controller.goal_qpos, stop_q, atol=1e-12)
        call("world", "wait", duration_s=5)
        stop_error = float(np.max(np.abs(np.array(call("arm", "read")["positions"]) - stop_q)))
        assert stop_error < 0.01
        for name in ("arm", "gripper"):
            for action in ("read", "status", "load", "unload"):
                budget = adapter.budget().used
                call(name, action)
                assert adapter.budget().used == budget
        frames = {}
        if render:
            for camera in ("camera", "camera_wrist"):
                budget = adapter.budget().used
                result = call(camera, "read")
                assert (result["width"], result["height"]) == (256, 256)
                assert Path(result["file"]).read_bytes().startswith(b"\x89PNG")
                assert adapter.budget().used == budget
                frames[camera] = result["file"]
        used = adapter.budget().used
        call("arm", "write", actions=[q], duration_s=len([q]) / 20)
        call("world", "wait", duration_s=0.05)
        assert adapter.budget().used == used + 1
        previous = world.env.step
        world.env.step = lambda action: (_ for _ in ()).throw(RuntimeError("probe"))
        try:
            world.tick()
        except RuntimeError:
            pass
        finally:
            world.env.step = previous
        assert "set_goal" not in world._robot.controller.__dict__
        assert np.isfinite(world.sim.get_state().flatten()).all()
        return {
            "suite": suite,
            "task": task,
            "init_state_index": init,
            "seed": 0,
            "rig": ROBOTS[world.wrapper.problem_name.lower()],
            "problem": world.wrapper.problem_name.lower(),
            "initialization": world.initialization,
            "joint_error_rad": error,
            "stop_error_rad": stop_error,
            "fk_error_m": max(fk_errors),
            "frames": frames,
            "budget": adapter.budget().as_dict(),
            "native_success": adapter.score().completed,
            "passed": True,
            "full_task_attempted": False,
        }
    finally:
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    chosen = CASES if args.case is None else [CASES[args.case]]
    results = [probe(suite, task, args.init, args.out, args.render) for suite, task in chosen]
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(results), "render": args.render}))
