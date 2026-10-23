"""Replay a T2 command record in fresh physics, without restoring saved state."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def compare(actual, expected):
    """Tight numeric replay tolerance with exact structural and outcome matching."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(compare(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(compare(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) in (float, int) and (
        type(actual) in (float, int) or isinstance(actual, (np.floating, np.integer))
    ):
        return bool(np.isclose(actual, expected, rtol=0, atol=1e-5))
    return type(actual) is type(expected) and actual == expected


def load_definition(record, task_file=None):
    """Preserve actor creation order, which sorted evidence JSON cannot retain."""
    definition = record["task_definition"]
    if task_file is None:
        if len(definition["items"]) + len(definition.get("articulated", {})) > 1:
            raise ValueError("Multi-object replay requires --task-file with the original task YAML")
        return definition
    task = yaml.safe_load(Path(task_file).read_text())
    if (task["backend_kwargs"]["task_name"] != record["task_name"]
            or ("task" in record and task["id"] != record["task"])
            or task["seed"] != record["seed"]
            or task["execution_mode"] != record["execution_mode"]
            or task["backend_kwargs"]["task_spec"] != definition):
        raise ValueError("Replay task YAML differs from the recorded task definition or instance")
    return task["backend_kwargs"]["task_spec"]


def replay(path, assets, out, task_file=None):
    from roboenv_runtime.adapters.isaaclab.robots.booster_t2.environment import T2Env

    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if record.get("schema") != "roboenv/booster-t2-record/1":
        raise ValueError("Unsupported T2 record")
    definition = load_definition(record, task_file)
    steps = record["final"]["steps"]
    if type(steps) is not int or steps < 0:
        raise ValueError("Invalid final frame")
    previous = -1
    for event in record["events"]:
        frame = event.get("frame")
        if type(frame) is not int or not previous <= frame <= steps:
            raise ValueError("Replay frames must be ordered and inside the episode")
        previous = frame
        if event.get("op") not in ("write", "stop"):
            raise ValueError("Unknown T2 replay operation")
    world = T2Env(
        task_name=record["task_name"],
        task_spec=definition,
        asset_directory=assets,
        max_steps=record["max_steps"],
        execution_mode=record["execution_mode"],
        stop_on_success=record["stop_on_success"],
        # Match the owner's scene construction, including camera prims.
        cameras=True,
        device_profiles=record["device_profiles"],
    )
    try:
        world.reset(record["seed"])
        initial = world.record()
        for key in ("sources", "task_definition", "device_profiles", "initialization"):
            if initial[key] != record[key]:
                raise ValueError(f"T2 replay identity mismatch: {key}")
        if not compare(initial["initial_state"], record["initial_state"]):
            raise ValueError("T2 initial physics state differs")
        events = iter(record["events"])
        event = next(events, None)
        for frame in range(steps + 1):
            while event is not None and event["frame"] == frame:
                if event["op"] == "write":
                    world.set_target(event["device"], event["args"])
                else:
                    world.stop(event["device"], event["reason"])
                event = next(events, None)
            if frame < steps:
                world.tick()
        final = world.record()
        passed = (
            compare(final["final"], record["final"])
            and final["native_success"] == record["native_success"]
            and final["budget"] == record["budget"]
        )
        report = {
            "schema": "roboenv/booster-t2-replay/1",
            "passed": passed,
            "record_sha256": hashlib.sha256(raw).hexdigest(),
            "steps": steps,
            "task_success": final["native_success"],
            "execution_mode": record["execution_mode"],
            "absolute_tolerance": 1e-5,
            "item_order": list(definition["items"]),
            "task_file_sha256": (hashlib.sha256(Path(task_file).read_bytes()).hexdigest()
                                 if task_file is not None else None),
        }
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not passed:
            raise ValueError("T2 command replay diverged")
        return report
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--task-file", type=Path,
                        help="Original resolved task YAML; required for multiple objects")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    replay(args.record, args.assets, args.out, args.task_file)
