"""Measured high-speed Go2 validation; run explicitly in the Isaac interpreter."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env
from roboenv_runtime.adapters.isaaclab.go2_d1.kinematics import quat_matrix

CASES = {
    "forward": (0, 1.0),
    "backward": (0, -1.0),
    "left": (1, 0.5),
    "right": (1, -0.5),
    "ccw": (2, math.pi / 6),
    "cw": (2, -math.pi / 6),
}


def run(out, cases, device="cuda:0"):
    out.mkdir(parents=True, exist_ok=False)
    world = Go2Env(realtime=True, command_timeout_s=None, cameras=False, device=device)
    try:
        results = {}
        for name in cases:
            axis, target = CASES[name]
            world.reset(0)
            rows = []
            for i in range(900):
                velocity = [0.0, 0.0, 0.0]
                if 150 <= i < 750:
                    velocity[axis] = target
                if i in (0, 150, 750):
                    world.write(
                        "base",
                        {
                            "mode": "locomotion",
                            "velocity": dict(zip(("vx", "vy", "wz"), velocity, strict=True)),
                            "posture": {"height": 0.30, "roll": 0.0, "pitch": 0.0},
                        },
                    )
                world.advance()
                feedback = world.read_device("base")
                v = feedback["velocities"]
                pose = feedback["pose"]
                r = quat_matrix(pose["rotation_xyzw"])
                rows.append(
                    {
                        "time_s": world.steps * 0.02,
                        "command": velocity,
                        "measured": [v[0], v[1], v[5]],
                        "position": pose["translation_m"],
                        "posture": [
                            feedback["height_m"],
                            float(np.arctan2(r[2, 1], r[2, 2])),
                            float(np.arcsin(np.clip(-r[2, 0], -1, 1))),
                        ],
                    }
                )
                if world.done:
                    break
            steady = [r for r in rows if 11 <= r["time_s"] < 15]
            mean = (
                np.mean([r["measured"] for r in steady], axis=0) if steady else np.full(3, np.nan)
            )
            actual = np.array([r["measured"] for r in steady])
            # Report raw gait oscillation separately from cycle-averaged tracking.
            rms = float(np.sqrt(np.mean((actual[:, axis] - target) ** 2))) if steady else None
            pose_error = (
                np.max(
                    np.abs(np.array([r["posture"] for r in steady]) - [0.3, 0, 0]), axis=0
                ).tolist()
                if steady
                else None
            )
            end = [r for r in rows if r["time_s"] >= 17.5]
            stop_mean = np.mean([r["measured"] for r in end], axis=0) if end else np.full(3, np.nan)
            passed = (
                world.steps == 900
                and not world.done
                and np.isfinite(mean).all()
                and abs(mean[axis] - target) <= 0.1 * abs(target)
                and np.max(np.abs(np.delete(mean, axis))) < 0.1
                and np.max(np.abs(stop_mean)) < 0.08
                and pose_error[0] < 0.05
                and max(pose_error[1:]) < 0.20
            )
            results[name] = {
                "passed": bool(passed),
                "target": target,
                "mean": mean.tolist(),
                "raw_rms_error": rms,
                "max_posture_error": pose_error,
                "stop_mean": stop_mean.tolist(),
                "fallen": world.native.done,
                "steps": world.steps,
            }
            (out / f"{name}-trace.json").write_text(json.dumps(rows, indent=2))
            (out / f"{name}-record.json").write_text(json.dumps(world.record(), indent=2))
            (out / "report.json").write_text(json.dumps(results, indent=2))
            print(name, results[name], flush=True)
        if not all(r["passed"] for r in results.values()):
            raise ValueError("High-speed conformance failed; inspect the retained report")
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.out, args.cases, args.device)
