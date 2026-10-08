"""Owner-side task feasibility probe; privileged fixture, not an agent solution."""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.controllers import matrix_quat, quat_matrix
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.environment import Go2Env
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.task import PICK_PLACE, REACH


def run(out, task, cameras=False, drop=False):
    out.mkdir(parents=True, exist_ok=False)
    w = Go2Env(
        task_name=task, cameras=cameras, max_steps=6000
    )
    trace = []
    try:
        n = w.native
        tcp_offset = np.asarray(n.task_spec["tcp_in_flange_m"])
        posture = [0.30, 0.0, 0.0]
        velocity = [0.0, 0.0, 0.0]
        arm = np.array([n.home[k] for k in n.model.arm_names])
        opening = 1.0

        def step(count, label):
            for _ in range(count):
                if w.done:
                    return
                w.set_target(
                    "base",
                    {
                        "mode": "locomotion",
                        "velocity": dict(zip(("vx", "vy", "wz"), velocity, strict=True)),
                        "posture": dict(zip(("height", "roll", "pitch"), posture, strict=True)),
                    },
                )
                w.set_target("arm", {"target": arm.tolist()})
                w.set_target("gripper", {"target": opening})
                w.advance()
                if w.steps % 10 == 0:
                    trace.append({"frame": w.steps, "phase": label, **n.snapshot()})
                if w.steps % 100 == 0:
                    print(label, w.steps, n.root_pose()[:3], n.task.snapshot(), flush=True)

        def drive(x):
            for _ in range(1100):
                root = n.root_pose()
                r = quat_matrix(root[3:])
                yaw = np.arctan2(r[1, 0], r[0, 0])
                error = np.array([x, 0]) - root[:2]
                local = r[:2, :2].T @ error
                velocity[:] = [
                    float(np.clip(local[0] * 1.5, -0.25, 0.25)),
                    float(np.clip(local[1], -0.10, 0.10)),
                    float(np.clip(-yaw, -0.2, 0.2)),
                ]
                step(1, "walk")
                if w.done:
                    return
                if np.linalg.norm(error) < 0.008:
                    break
            velocity[:] = [0.0, 0.0, 0.0]
            step(100, "stand")

        def tcp():
            root = n.root_pose()
            t = n.model.forward(n.joints())["d1_Link6"]
            return root[:3] + quat_matrix(root[3:]) @ (t[:3, 3] + t[:3, :3] @ tcp_offset)

        def move(target, rotation, seconds=3.0):
            nonlocal arm
            initial = tcp()
            count = int(seconds / 0.02)
            for i in range(count):
                if w.done:
                    return
                if i % 5 == 0:
                    root = n.root_pose()
                    rb = quat_matrix(root[3:])
                    goal = initial + (np.array(target) - initial) * min((i + 5) / count, 1.0)
                    flange_world = goal - rotation @ tcp_offset
                    mount = n.model.forward({})["d1_base_link"]
                    p = mount[:3, :3].T @ (rb.T @ (flange_world - root[:3]) - mount[:3, 3])
                    rot = mount[:3, :3].T @ rb.T @ rotation
                    arm = np.array(n.model.arm_ik([*p, *matrix_quat(rot)], arm, iterations=220))
                step(1, "arm")

        step(100, "settle")
        if task == REACH:
            drive(1.10)
            rotation = quat_matrix([0, np.sqrt(0.5), 0, np.sqrt(0.5)])
            move([1.4, 0, 0.65], rotation, 3.0)
            step(100, "reach_hold")
        else:
            drive(0.88)
            posture[:] = [0.26, 0.0, 0.12]
            step(150, "crouch")
            down = np.diag([-1.0, 1.0, -1.0])
            # Rotate above the ground before descending into the physical grasp.
            root = n.root_pose()
            rb = quat_matrix(root[3:])
            seed = [0.0, 1.45, 0.30, 0.0, -0.3, 0.0]
            pos = (
                rb.T @ (np.array([1.25, 0, 0.25]) - down @ tcp_offset - root[:3])
                - [0.085, 0, 0.094]
            )
            arm = np.array(n.model.arm_ik([*pos, *matrix_quat(rb.T @ down)], seed, iterations=250))
            step(200, "pregrasp")
            move([1.25, 0, 0.045], down, 4.0)
            step(50, "align")
            opening = 0.0
            step(100, "grasp")
            move([1.25, 0, 0.25], down, 4.0)
            step(75, "lift")
            if drop:
                opening = 1.0
                step(100, "intentional_drop")
            drive(2.13)
            move([2.5, 0, 0.045], down, 4.0)
            step(50, "lower")
            opening = 1.0
            step(150, "release")
        if cameras:
            for name in n.cameras:
                w.read_camera(name, out)
        record = w.record()
        (out / "record.json").write_text(json.dumps(record, indent=2))
        (out / "trace.json").write_text(json.dumps(trace))
        result = {
            "task": task,
            "completed": w.completed,
            "steps": w.steps,
            "state": n.task.snapshot(),
        }
        (out / "report.json").write_text(json.dumps(result, indent=2))
        print("RESULT", result, flush=True)
        if drop:
            if w.completed or n.task.failure != "object_dropped":
                raise ValueError("Intentional drop was not rejected")
        elif not w.completed:
            raise ValueError("Task feasibility probe did not succeed")
    finally:
        if not (out / "record.json").exists():
            (out / "record.json").write_text(json.dumps(w.record(), indent=2))
            (out / "trace.json").write_text(json.dumps(trace))
        w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=[REACH, PICK_PLACE], required=True)
    p.add_argument("--cameras", action="store_true")
    p.add_argument("--drop", action="store_true", help="Negative probe: release during transport")
    args = p.parse_args()
    if args.drop and args.task != PICK_PLACE:
        p.error("--drop requires pick-and-place")
    run(args.out, args.task, args.cameras, args.drop)
