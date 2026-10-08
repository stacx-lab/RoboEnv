"""Bounded, no-model spectator feasibility probe in a fresh native interpreter.

This exercises the real worker and its clock, but not HTTP or a recording service.
It writes private evidence before native cleanup, which can terminate Isaac Python.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from roboenv.provisioning.interpreter import resolve_interpreter
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.core.adapter import make_adapter
from roboenv_runtime.runtime.atomic import atomic_write_json
from roboenv_runtime.runtime.worker import SimWorker, jsonable, write_png
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer

ROOT = Path(__file__).resolve().parents[1]


def snapshot(worker):
    """Expensive comparisons belong in the probe, never the rendering loop."""
    adapter, native = worker.adapter, worker.adapter._env
    if adapter.name in {"libero", "robocasa", "simple"}:
        import mujoco

        model, data = ((native.mjmodel, native.mjdata) if adapter.name == "simple" else
                       (native.sim.model._model, native.sim.data._data))
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        state = {"integration": state, "camera_position": model.cam_pos,
                 "camera_rotation": model.cam_quat, "camera_fov": model.cam_fovy,
                 "offscreen": [model.vis.global_.offwidth, model.vis.global_.offheight]}
        if adapter.name == "simple":
            state["record"] = native.record()
            state["native_steps"] = native.env.step_count
            if native.env.isaac is not None:
                isaac = native.env.isaac
                state["render_steps"] = isaac.step_id
                state["agent_cameras"] = {
                    name: {"resolution": camera.get_resolution(),
                           "intrinsics": camera.get_intrinsics_matrix(),
                           "pose": camera.get_world_pose(camera_axes="usd")}
                    for name, camera in isaac.cameras.items()
                }
    elif adapter.name == "isaaclab":
        from roboenv_runtime.adapters.isaaclab.robots.yam.native import array

        engine = native.native
        robots = engine.robots if adapter.embodiment == "yam" else {"robot": engine.robot}
        state = {"robots": {
            name: {field: array(getattr(robot.data, field)) for field in (
                "root_link_pose_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")}
            for name, robot in robots.items()}, "steps": engine.steps,
            "targets": native.targets, "events": native.events,
            "task": engine.snapshot() if adapter.embodiment == "go2_d1" else {
                "contacts": {name: array(sensor.data.force_matrix_w)
                             for name, sensor in engine.contacts.items()}},
            "cameras": {name: {"intrinsics": array(camera.data.intrinsic_matrices),
                                "resolution": [camera.cfg.width, camera.cfg.height],
                                "position": array(camera.data.pos_w),
                                "orientation": array(camera.data.quat_w_world)}
                        for name, camera in engine.cameras.items()}}
        # Sensor buffers can remain cached between physics ticks. Check PhysX
        # directly too, so a render-only update cannot conceal native motion.
        state["raw_physics"] = {
            name: {
                field: getattr(robot.root_view, method)().numpy().copy()
                for field, method in (
                    ("joint_pos", "get_dof_positions"),
                    ("joint_vel", "get_dof_velocities"),
                    ("root_pose", "get_root_transforms"),
                    ("root_vel", "get_root_velocities"),
                )
            }
            for name, robot in robots.items()
        }
    else:
        state = native.owner_state()
    rng = {"python": random.getstate(), "numpy": np.random.get_state()}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        rng["torch"] = torch.get_rng_state().tolist()
        if torch.cuda.is_initialized():
            rng["cuda"] = [value.tolist() for value in torch.cuda.get_rng_state_all()]
    return {"state": digest(jsonable(state)), "rng": digest(jsonable(rng)),
            "budget": adapter.budget().as_dict(), "score": adapter.score().as_dict(),
            "devices": worker.devices.inventory()}


def resolve_probe(args):
    resolved = resolve_task(args.task, project=ROOT)
    if args.sim_mode is not None:
        if resolved.task.backend != "simple":
            raise ValueError("--sim-mode is supported only for SIMPLE")
        resolved = replace(resolved, task=replace(resolved.task, backend_kwargs={
            **resolved.task.backend_kwargs, "sim_mode": args.sim_mode,
        }))
    return resolved


def probe(args):
    resolved = resolve_probe(args)
    task = resolved.task
    robot = args.out / "robot"
    shutil.copytree(resolved.robot, robot)
    (robot / "bindings.yaml").write_text(yaml.safe_dump(resolved.bindings.as_dict()))
    report = {"passed": False, "task": task.id, "clock": args.clock,
              "sim_mode": task.backend_kwargs.get("sim_mode"),
              "scope": "native spectator isolation; no HTTP, task-success or replay certification"}
    worker = renderer = None
    try:
        adapter = make_adapter(task.backend, **task.backend_kwargs)
        worker = SimWorker(adapter, task_id=task.id, seed=task.seed, run_dir=args.out / "world",
                           device_root=robot, execution_mode=args.clock)
        inventory = worker.devices.inventory()["devices"]
        cameras = [d["name"] for d in inventory if d["type"] in {"rgb", "rgbd"}]

        def call(device, action, **kwargs):
            return worker.dispatch("devices.call", {
                "device": device, "action": action, "args": kwargs,
            })["data"]

        def images():
            import cv2

            if adapter.name in {"simple", "isaaclab"}:
                # Force real sensor acquisition: cached images would hide shared
                # render-product corruption in the before/after comparison.
                adapter._env.images.clear()
            result = {}
            for name in cameras:
                reading = call(name, "read")
                for field in ("file", "depth_file"):
                    if field in reading:
                        # Kit can mix its Pillow plugins with the interpreter's
                        # Pillow. Decode independently of those global plugins.
                        image = cv2.imread(reading[field], cv2.IMREAD_UNCHANGED)
                        if image is None:
                            raise ValueError(f"Could not decode {name}/{field}")
                        result[name + "/" + field] = image
            return result

        # Establish ordinary renderer variability and initialize existing sensors.
        if adapter.name == "isaaclab":
            # Settle RTX temporal history after positioning all device cameras.
            # Keep the same MAE gate; compare stable images, not startup frames.
            for _ in range(32):
                images()
        first = images()
        if adapter.name == "isaaclab":
            from roboenv_runtime.spectator.isaaclab import Engine

            # Match the setup/capture render gap. Adjacent RTX frames share a
            # temporal history and underestimate ordinary long-gap variability.
            control_renderer = Engine(adapter)
            for _ in range(control_renderer.warmup_frames + 1):
                control_renderer.render()
            report["baseline_render_updates"] = control_renderer.warmup_frames + 1
        baseline = images()
        before = snapshot(worker)
        specs = ([ViewSpec(**value) for value in json.loads(args.views.read_text())]
                 if args.views else None)
        renderer = create_spectator_renderer(adapter, specs)
        if snapshot(worker) != before:
            raise ValueError("Spectator setup changed native state, RNG, cameras or verdict")
        report["views"] = [asdict(view) for view in renderer.views]
        captures, latency = 0, []
        next_capture = 0.0

        def capture(force=False):
            nonlocal captures, next_capture
            now = worker.world.time_s
            if not force and now + 1e-9 < next_capture:
                return
            state = snapshot(worker)
            started = time.monotonic()
            frames = renderer.capture()
            latency.append(time.monotonic() - started)
            if snapshot(worker) != state:
                raise ValueError("Capture changed native state, RNG, cameras or verdict")
            if captures == 0 or force:
                for name, frame in frames.items():
                    write_png(args.out / f"{captures}_{name}.png", frame)
            captures += 1
            next_capture = (int((now + 1e-9) * 10) + 1) / 10

        capture()
        after = images()
        if adapter.name == "isaaclab":
            for name in baseline:
                if name.endswith("/file"):
                    stem = name.replace("/", "_")
                    write_png(args.out / (stem + "_before.png"), baseline[name][..., ::-1])
                    write_png(args.out / (stem + "_after.png"), after[name][..., ::-1])
        report["agent_image_mae"] = {}
        image_failures = []
        for name in baseline:
            natural = np.abs(first[name].astype(float) - baseline[name]).mean()
            change = np.abs(after[name].astype(float) - baseline[name]).mean()
            report["agent_image_mae"][name] = {"baseline": natural, "spectator": change}
            if change > natural + 1:
                image_failures.append(name)
        terminal = worker.world.after_advance

        def after_advance():
            capture()
            terminal()

        worker.world.after_advance = after_advance
        arm = next(d["name"] for d in inventory
                   if d["type"] == "joint" and "arm" in d["name"])
        initial = call(arm, "read")["positions"]
        goal = list(initial)
        goal[0] += 0.02
        call(arm, "write", actions=[goal], duration_s=args.seconds)
        wait = worker.dispatch_async("devices.call", {
            "device": "world", "action": "wait", "args": {"duration_s": args.seconds},
        })
        while not wait.done():
            worker.world.advance()
            # Exercise normal agent camera reads while the write/wait is pending.
            if captures % 5 == 0:
                call(cameras[0], "read")
            time.sleep(0.001)
        wait.result()
        call(arm, "stop")
        final = call(arm, "read")["positions"]
        report["motion_rad"] = max(abs(a - b) for a, b in zip(initial, final, strict=True))
        if report["motion_rad"] < 0.001:
            raise ValueError("Probe did not produce measured arm motion")
        capture(force=True)
        state = snapshot(worker)
        renderer.close()
        renderer = None
        if snapshot(worker) != state:
            raise ValueError("Spectator cleanup changed the world")
        images()  # Agent sensors must remain usable after spectator cleanup.
        report.update(passed=not image_failures, state_isolation_passed=True,
                      image_isolation_passed=not image_failures,
                      captures=captures, max_capture_s=max(latency),
                      mean_capture_s=sum(latency) / len(latency),
                      native_success=adapter.score().completed)
        if image_failures:
            report["error"] = (
                "Agent images changed beyond baseline variability: " + ", ".join(image_failures)
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Native shutdown must not erase the gate's measured result.
        atomic_write_json(args.out / "result.json", report)
        try:
            if renderer is not None:
                renderer.close()
        finally:
            if worker is not None:
                worker.close()
    return 0 if report["passed"] else 1


def finish_report(directory, code):
    """Keep a post-check native shutdown crash distinct from passing isolation."""
    path = directory / "result.json"
    report = json.loads(path.read_text())
    report["isolation_passed"] = report.get("passed", False)
    report["native_exit_code"] = code
    report["passed"] = code == 0 and report["isolation_passed"]
    if code != 0 and report["isolation_passed"]:
        report["error"] = f"Native process exited with status {code} after isolation checks"
    atomic_write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path)
    parser.add_argument("--clock", choices=("stepped", "realtime"), default="stepped")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--views", type=Path, help="JSON list of ViewSpec fields")
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"),
                        help="Override SIMPLE rendering mode")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300 or not 0 < args.timeout < float("inf"):
        parser.error("Use 0 < seconds <= 300 and a finite positive timeout")
    args.out = args.out.resolve()
    if args.views:
        args.views = args.views.resolve()
    if args.native:
        return probe(args)
    args.out.mkdir(parents=True, exist_ok=False)
    task = resolve_probe(args).task
    interpreter = resolve_interpreter(ROOT, task.backend, args.interpreter)
    command = [str(interpreter), str(Path(__file__).resolve()), args.task, "--native",
               "--out", str(args.out), "--clock", args.clock,
               "--seconds", str(args.seconds)]
    if args.views:
        command.extend(["--views", str(args.views)])
    if args.sim_mode:
        command.extend(["--sim-mode", args.sim_mode])
    package = ROOT / "runtime/src/roboenv_runtime/spectator"
    sources = [Path(__file__), *sorted(package.glob("*.py"))]
    atomic_write_json(args.out / "spec.json", {
        "command": command, "timeout_s": args.timeout, "task": task.as_dict(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    })
    atomic_write_json(args.out / "result.json", {"passed": False, "error": "probe did not finish"})
    from roboenv.provisioning.acceptance import acceptance_environment
    from roboenv.provisioning.gpu import gpu_environment, graphics_apis, resolve_gpu

    env = {**acceptance_environment(), "MUJOCO_GL": "egl",
           "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src"))}
    selection = resolve_gpu(graphics_apis(task.backend, task.backend_kwargs))
    if selection is not None:
        env.update(gpu_environment(selection))
    with (args.out / "native.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            atomic_write_json(args.out / "result.json", {
                "passed": False, "error": "probe interrupted",
            })
            raise
    report = finish_report(args.out, code)
    print(json.dumps(report))
    return 0 if code == 0 and report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
