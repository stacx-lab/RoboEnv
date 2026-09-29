"""Replay a SIMPLE owner record in a fresh process, without agent or final-state restore."""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.simple.backend import SimpleEnv


def replay(record):
    if record.get("schema") != "roboenv/simple-record/3":
        raise ValueError("Expected a SIMPLE owner record")
    mode = record.get("execution_mode")
    if mode not in ("stepped", "realtime"):
        raise ValueError("Replay requires execution_mode: stepped or realtime")
    metadata = record["initialization"]
    if metadata.get("control_protocol") != "simple-g1-metric-wbc/4":
        raise ValueError("Unsupported SIMPLE control protocol")
    if type(metadata.get("arm_gravity_compensation")) is not bool:
        raise ValueError("SIMPLE replay requires boolean arm_gravity_compensation")
    if len(record["frames"]) != record["budget"]["used"]:
        raise ValueError("SIMPLE replay frame count differs from budget")
    world = SimpleEnv(
        env_id=metadata["env_id"],
        arm_gravity_compensation=metadata["arm_gravity_compensation"],
        sim_mode=metadata["sim_mode"],
        dr_level=metadata["dr_level"],
        initialization=json.loads(metadata["state_dict_json"]),
        stabilize_steps=metadata["stabilize_steps"],
    )
    try:
        world.configure_execution_mode(mode)
        world.reset(metadata["seed"])
        for key in (
            "source_lock_sha256",
            "facts_sha256",
            "asset_pin_sha256",
            "loaded_state_sha256",
            "initialization_frames",
            "control_protocol",
            "arm_gravity_compensation",
        ):
            if world.initialization[key] != metadata[key]:
                raise ValueError(f"SIMPLE replay initialization changed: {key}")
        if world.implementation != record["implementation"]:
            raise ValueError("SIMPLE replay implementation or dependencies changed")
        # Replay accepted writes and stops without a wall-clock scheduler. Physics
        # advances solely at the recorded interval boundaries in synchronous execution.
        events = iter(record["commands"])
        event = next(events, None)
        for frame in range(len(record["frames"]) + 1):
            while event is not None and event["frame"] == frame:
                if event["op"] == "write":
                    world.control.write(event["device"], event["args"])
                elif event["op"] == "plan":
                    world.control.plan(event["device"])
                elif event["op"] == "stop":
                    world.control.stop(event["device"], reason=event["reason"])
                else:
                    raise ValueError("Unknown SIMPLE replay operation")
                event = next(events, None)
            if event is not None and (type(event["frame"]) is not int or event["frame"] < frame):
                raise ValueError("SIMPLE replay commands must have ordered frame indices")
            if frame == len(record["frames"]):
                break
            if world.control.targets != record["frames"][frame]:
                raise ValueError("SIMPLE replay composed targets differ from recorded interval")
            world.advance(world.control.targets)
        if event is not None:
            raise ValueError("SIMPLE replay command falls beyond the episode")
        if world.control.targets != record["pending_targets"]:
            raise ValueError("SIMPLE replay final retained goals differ")
        actual = world.record()
        errors = {}
        for key in ("state",):
            a, b = np.asarray(actual[key]), np.asarray(record[key])
            if a.shape != b.shape:
                raise ValueError("SIMPLE replay state dimensions changed")
            errors[key] = float(np.max(np.abs(a - b), initial=0))
        from roboenv_runtime.adapters.native_record import compare

        errors["controller"] = compare(record["controller_state"], actual["controller_state"],
                                       atol=1e-8, path="controller_state")
        passed = (
            all(value <= 1e-8 for value in errors.values())
            and actual["budget"] == record["budget"]
            and actual["native_success"] == record["native_success"]
            and actual["done"] == record["done"]
            and actual["reward"] == record["reward"]
        )
        return {
            "passed": passed,
            "verdict_matches": {
                key: actual[key] == record[key]
                for key in ("budget", "native_success", "done", "reward")
            },
            "max_absolute_errors": errors,
            "native_success": actual["native_success"],
            "budget": actual["budget"],
        }
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Replay output must be new")
    result = replay(json.loads(args.record.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
