"""Bounded YAM dynamics and buffer/hold checks in the isolated Isaac interpreter."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.robots.yam.native import array, body_dynamics
from roboenv_runtime.runtime.worker import SimWorker
from roboenv_runtime.spectator import create_spectator_renderer


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    resolved = resolve_task("isaaclab/yam-calibrate-wrist-cameras")
    robot = out / "robot"
    shutil.copytree(resolved.robot, robot)
    (robot / "bindings.yaml").write_text(yaml.safe_dump(resolved.bindings.as_dict()))
    report = {"passed": False, "scope": "bounded dynamics/hold; no calibration accuracy claim"}
    trace = []
    worker = renderer = None
    try:
        adapter = IsaacLabAdapter(task_name="yam-calibrate-wrist-cameras")
        worker = SimWorker(adapter, task_id=resolved.task.id, seed=0, run_dir=out / "world",
                           device_root=robot, execution_mode="stepped")
        env = adapter.environment

        def call(device, action, **args):
            return worker.dispatch("devices.call", {
                "device": device, "action": action, "args": args,
            })["data"]

        def state():
            return {side: {"q": r.root_view.get_dof_positions().numpy().copy(),
                           "v": r.root_view.get_dof_velocities().numpy().copy()}
                    for side, r in env.native.robots.items()}

        def render_only(fn):
            before, steps = state(), env.native.steps
            fn()
            after = state()
            assert env.native.steps == steps
            for side in before:
                for field in before[side]:
                    np.testing.assert_array_equal(after[side][field], before[side][field])

        def record():
            trace.append({"time_s": worker.world.time_s, "targets": copy.deepcopy(env.targets),
                          "arms": {side: {"q": array(r.data.joint_pos).tolist(),
                                          "v": array(r.data.joint_vel).tolist()}
                                   for side, r in env.native.robots.items()}})
            if renderer is not None and len(trace) % 3 == 0:
                render_only(renderer.capture)

        worker.world.after_advance = record
        report["mass_error_kg"] = {}
        for side, articulation in env.native.robots.items():
            expected = body_dynamics(Path(".cache/isaaclab/yam") / (side + ".xml"))
            actual = articulation.root_view.get_masses().numpy()[0]
            wanted = np.array([expected[name]["mass"] for name in articulation.body_names])
            np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-8)
            report["mass_error_kg"][side] = float(np.max(np.abs(actual - wanted)))

        def settled(label):
            metrics = {}
            for side in ("left", "right"):
                name = side + "_arm"
                reading = call(name, "read")
                error = float(np.max(np.abs(np.array(reading["positions"]) - env.targets[name])))
                velocity = float(np.max(np.abs(reading["velocities"])))
                metrics[side] = {"max_error_rad": error, "max_speed_rad_s": velocity}
                assert error < 0.002, (label, side, metrics)
                assert velocity < 0.002, (label, side, metrics)
            report[label] = metrics

        call("world", "wait", duration_s=3.06)
        settled("home_hold")
        before = state()
        renderer = create_spectator_renderer(adapter)
        for side in before:
            for field in before[side]:
                np.testing.assert_array_equal(state()[side][field], before[side][field])
        for side, joint in (("left", 3), ("right", 4)):
            name = side + "_arm"
            start = np.array(call(name, "read")["positions"])
            goal = start.copy()
            goal[joint] += 0.35
            peer = "right_arm" if side == "left" else "left_arm"
            peer_target = env.targets[peer].copy()
            u = np.linspace(0, 1, 440)
            blend = 10*u**3 - 15*u**4 + 6*u**5
            actions = (start + blend[:, None] * (goal - start)).tolist()
            call(name, "write", actions=actions, duration_s=4.4)
            call("world", "wait", duration_s=4.42)
            assert env.targets[peer] == peer_target
            np.testing.assert_array_equal(env.targets[name], goal)
            assert not any(event["op"] == "stop" for event in env.events)
            call("world", "wait", duration_s=1.02)
            settled(side + "_completed_hold")
            for camera in ("camera_left", "camera_right", "camera_top"):
                env.images.clear()
                render_only(lambda camera=camera: call(camera, "read"))
        # Explicit stop may latch measured position, but cannot reset its peer.
        peer = env.targets["right_arm"].copy()
        call("left_arm", "stop")
        assert env.targets["right_arm"] == peer
        call("world", "wait", duration_s=1.02)
        settled("explicit_stop_hold")
        report.update(passed=True, simulated_seconds=worker.world.time_s,
                      render_preserved_raw_state=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2))
        (out / "trace.json").write_text(json.dumps(trace))
        if renderer is not None:
            renderer.close()
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
