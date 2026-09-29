"""Replay a Go2–D1 owner journal in a fresh Isaac process, without state restoration."""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env


def replay(record, assets, out, video=None):
    if record.get("schema") != "roboenv/go2-d1-record/1":
        raise ValueError("Unsupported Go2–D1 record")
    world = Go2Env(
        asset_manifest=assets,
        cameras=video is not None,
        task_name=record.get("task_name", "go2-d1-control"),
        max_steps=record.get("max_steps"),
        execution_mode=record.get(
            "execution_mode", "realtime" if record.get("max_steps") is None else "stepped"
        ),
    )
    recorder = None
    try:
        world.reset(record["seed"])
        initial = world.record()
        for field in ("asset_sha256", "sources", "task_definition"):
            if initial[field] != record[field]:
                raise ValueError(f"Replay identity mismatch: {field}")
        if video is not None:
            from go2_task_video import TaskVideo

            recorder = TaskVideo(world, video)
        events = iter(record["events"])
        event = next(events, None)
        steps = record["final"]["steps"]
        for frame in range(steps + 1):
            while event is not None and event["frame"] == frame:
                if event["op"] == "write":
                    world.set_target(event["device"], event["args"])
                elif event["op"] == "stop":
                    world.stop(event["device"], event["reason"])
                else:
                    raise ValueError("Unknown replay event")
                event = next(events, None)
            if event is not None and (type(event["frame"]) is not int or event["frame"] < frame):
                raise ValueError("Replay events must be ordered by frame")
            if frame < steps:
                if world.done:
                    raise ValueError("Replay terminated before recorded final frame")
                world.advance()
                if recorder is not None and (world.steps % 2 == 0 or world.done):
                    recorder.capture()
        if event is not None:
            raise ValueError("Replay events exceed the final frame")
        final = world.record()
        if final["targets"] != record["targets"]:
            raise ValueError("Replay retained goals differ")
        actual, expected = final["final"], record["final"]
        root_error = float(np.max(np.abs(np.array(actual["root"]) - expected["root"])))
        joint_error = max(abs(actual["joints"][n] - q) for n, q in expected["joints"].items())
        task_match = all(
            actual["task"][k] == expected["task"][k]
            for k in ("stage", "completed", "failure", "hold", "lift_hold", "contact_loss")
        )
        object_error = 0.0
        object_velocity_error = 0.0
        if "object" in expected:
            object_error = float(
                np.max(np.abs(np.array(actual["object"]["pose"]) - expected["object"]["pose"]))
            )
            object_velocity_error = max(
                float(np.max(np.abs(np.array(actual["object"][k]) - expected["object"][k])))
                for k in ("linear_velocity", "angular_velocity")
            )
        result = {
            "passed": task_match
            and object_error < 0.005
            and object_velocity_error < 0.03
            and root_error < 0.005
            and joint_error < 0.01
            and actual["done"] == expected["done"]
            and actual["steps"] == steps,
            "root_max_error": root_error,
            "task_state_matched": task_match,
            "task_success": actual["task"]["completed"],
            "object_pose_max_error": object_error,
            "object_velocity_max_error": object_velocity_error,
            "joint_max_error_rad_or_m": joint_error,
            "root_tolerance": 0.005,
            "joint_tolerance": 0.01,
            "steps": steps,
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise ValueError("Replay final state differs from the recorded state")
        return result
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--assets", default=".cache/isaaclab/go2-d1/assets.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--video", type=Path, help="Record spectator MP4 during physical replay")
    args = parser.parse_args()
    replay(json.loads(args.record.read_text()), args.assets, args.out, args.video)
