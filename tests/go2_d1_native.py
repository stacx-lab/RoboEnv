"""Bounded native Go2–D1 controller checks; run with the isolated Isaac interpreter."""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.controllers import (
    DEFAULT,
    matrix_quat,
    quat_matrix,
)
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.environment import Go2Env

CASES = {
    "forward": [0.05, 0, 0, 0.30, 0, 0],
    "sideways": [0, 0.04, 0, 0.30, 0, 0],
    "turn": [0, 0, 0.12, 0.30, 0, 0],
    "posture": [0, 0, 0, 0.32, 0.06, -0.06],
}


def command(values):
    return {
        "mode": "locomotion",
        "velocity": dict(zip(("vx", "vy", "wz"), values[:3], strict=True)),
        "posture": dict(zip(("height", "roll", "pitch"), values[3:], strict=True)),
    }


def run(out, assets):
    out.mkdir(parents=True, exist_ok=False)
    world = Go2Env(asset_manifest=assets)
    native = world.native
    try:
        report = {"passed": True, "task_success": None, "cases": {}}
        for name, values in CASES.items():
            world.reset(0)
            history = []
            for i in range(650):
                if i in (0, 150, 600):
                    world.set_target("base", command(values if i == 150 else DEFAULT.tolist()))
                world.advance()
                if i % 25 == 0:
                    history.append(native.snapshot())
                if world.done:
                    raise ValueError(f"{name}: robot fell")
            (out / f"{name}-record.json").write_text(json.dumps(world.record(), indent=2))
            (out / f"{name}-trace.json").write_text(json.dumps(history, indent=2))
            first, last = history[12]["root"], history[23]["root"]
            duration = (history[23]["steps"] - history[12]["steps"]) * 0.02
            velocity = (np.array(last[:2]) - first[:2]) / duration

            def yaw(p):
                return np.arctan2(quat_matrix(p[3:])[1, 0], quat_matrix(p[3:])[0, 0])

            yaw_rate = (yaw(last) - yaw(first)) / duration
            r = quat_matrix(last[3:])
            posture = [last[2], np.arctan2(r[2, 1], r[2, 2]), np.arcsin(-r[2, 0])]
            measured = [*velocity.tolist(), float(yaw_rate), *map(float, posture)]
            tolerance = [0.025, 0.025, 0.04, 0.015, 0.025, 0.025]
            passed = bool(np.all(np.abs(np.array(measured) - values) < tolerance))
            report["cases"][name] = {
                "command": values,
                "measured": measured,
                "tolerance": tolerance,
                "passed": passed,
            }
            report["passed"] &= passed
            (out / "report.json").write_text(json.dumps(report, indent=2))
            print(name, measured, passed, flush=True)
        world.reset(0)
        for _ in range(150):
            world.advance()
        q = native.arm_target.copy()
        q[0] += 0.15
        desired = native.model.arm_fk(q)
        target = native.model.arm_ik(
            [*desired[:3, 3], *matrix_quat(desired[:3, :3])], world.read_device("arm")["positions"]
        )
        world.set_target("arm", {"target": target})
        world.set_target("gripper", {"target": 0.8})
        for _ in range(100):
            world.advance()
        actual = native.model.arm_fk(world.read_device("arm")["positions"])
        error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        report["arm_position_error_m"] = error
        report["passed"] &= error < 0.01 and not world.done
        before = world.steps
        report["cameras"] = {
            name: world.read_camera(name, out / "images")
            for name in ("camera_head", "camera_wrist")
        }
        assert world.steps == before
        assert world.read_camera("camera_head", out / "images") == report["cameras"]["camera_head"]
        (out / "arm-record.json").write_text(json.dumps(world.record(), indent=2))
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print("GO2_NATIVE_COMPLETE", report["passed"], flush=True)
        if not report["passed"]:
            raise ValueError("Native controller acceptance failed; inspect report.json")
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--assets", default=".cache/isaaclab/go2-d1/assets.json")
    args = parser.parse_args()
    run(args.out, args.assets)
