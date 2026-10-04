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
from dataclasses import asdict
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
    if adapter.name in {"libero", "robocasa"}:
        import mujoco

        model, data = native.sim.model._model, native.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        state = {"integration": state, "camera_position": model.cam_pos,
                 "camera_rotation": model.cam_quat, "camera_fov": model.cam_fovy,
                 "offscreen": [model.vis.global_.offwidth, model.vis.global_.offheight]}
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


def probe(args):
    resolved = resolve_task(args.task, project=ROOT)
    task = resolved.task
    robot = args.out / "robot"
    shutil.copytree(resolved.robot, robot)
    (robot / "bindings.yaml").write_text(yaml.safe_dump(resolved.bindings.as_dict()))
    report = {"passed": False, "task": task.id, "clock": args.clock,
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
            from PIL import Image

            result = {}
            for name in cameras:
                reading = call(name, "read")
                for field in ("file", "depth_file"):
                    if field in reading:
                        with Image.open(reading[field]) as image:
                            result[name + "/" + field] = np.asarray(image).copy()
            return result

        # Establish ordinary renderer variability and initialize existing sensors.
        first, baseline = images(), images()
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
        report["agent_image_mae"] = {}
        for name in baseline:
            natural = np.abs(first[name].astype(float) - baseline[name]).mean()
            change = np.abs(after[name].astype(float) - baseline[name]).mean()
            report["agent_image_mae"][name] = {"baseline": natural, "spectator": change}
            if change > natural + 1:
                raise ValueError(f"{name}: agent image changed beyond baseline variability")
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
        report.update(passed=True, captures=captures, max_capture_s=max(latency),
                      native_success=adapter.score().completed)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--interpreter", type=Path)
    parser.add_argument("--clock", choices=("stepped", "realtime"), default="stepped")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--views", type=Path, help="JSON list of ViewSpec fields")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu < 0 or not 0 < args.seconds <= 300 or not 0 < args.timeout < float("inf"):
        parser.error("Use 0 < seconds <= 300 and a finite positive timeout")
    args.out = args.out.resolve()
    if args.views:
        args.views = args.views.resolve()
    if args.native:
        return probe(args)
    args.out.mkdir(parents=True, exist_ok=False)
    task = resolve_task(args.task, project=ROOT).task
    interpreter = resolve_interpreter(ROOT, task.backend, args.interpreter)
    command = [str(interpreter), str(Path(__file__).resolve()), args.task, "--native",
               "--out", str(args.out), "--gpu", str(args.gpu), "--clock", args.clock,
               "--seconds", str(args.seconds)]
    if args.views:
        command.extend(["--views", str(args.views)])
    package = ROOT / "runtime/src/roboenv_runtime/spectator"
    sources = [Path(__file__), *sorted(package.glob("*.py"))]
    atomic_write_json(args.out / "spec.json", {
        "command": command, "timeout_s": args.timeout, "task": asdict(task),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    })
    atomic_write_json(args.out / "result.json", {"passed": False, "error": "probe did not finish"})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_GL": "egl",
           "MUJOCO_EGL_DEVICE_ID": str(args.gpu), "ROBOTWIN_RENDER_DEVICE": "cuda:0",
           "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src"))}
    env.pop("OMNIGIBSON_GPU_ID", None)
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
    report = json.loads((args.out / "result.json").read_text())
    print(json.dumps(report))
    return 0 if code == 0 and report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
