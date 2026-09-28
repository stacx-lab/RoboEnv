"""Owner-only replay of a frozen LIBERO device command record at simulation frames.

Run in the prepared simulator interpreter, as a new process. Input is the owner's
post-submit observe result, never an agent instruction or public device artifact.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter

ROOT = Path(__file__).resolve().parents[1]


def replay(record):
    metadata = record["initialization"]
    adapter = LiberoAdapter(
        metadata["suite"],
        metadata["task_id"],
        init_state_index=metadata["init_state_index"],
        enable_render=False,
        max_steps=metadata["max_steps"],
    )
    try:
        adapter.reset(metadata["seed"])
        native = adapter._env
        for key in (
            "bddl_sha256",
            "init_state_sha256",
            "loaded_state_sha256",
            "controller",
            "control_frequency_hz",
            "initialization_frames",
        ):
            if native.initialization[key] != metadata[key]:
                raise ValueError(f"Replay initialization mismatch: {key}")
        for event in record["commands"]:
            frame = event["frame"]
            if (
                type(frame) is not int
                or frame < adapter.budget().used
                or frame > record["budget"]["used"]
            ):
                raise ValueError("Replay events must be ordered simulation frames")
            while adapter.budget().used < frame:
                native.tick()
            if event["op"] == "write":
                native.step(event["commands"])
            elif event["op"] == "stop":
                native.stop_part(event["part"])
            else:
                raise ValueError("Unknown replay operation")
        while adapter.budget().used < record["budget"]["used"]:
            native.tick()
        state = native.sim.get_state().flatten()
        expected = np.asarray(record["state"])
        if state.shape != expected.shape:
            raise ValueError("Replay model state dimension changed")
        difference = float(np.max(np.abs(state - expected)))
        success = native.task_completed()
        passed = (
            difference < 1e-8
            and success == record["native_success"]
            and adapter.budget().as_dict() == record["budget"]
        )
        return {
            "passed": passed,
            "max_state_difference": difference,
            "native_success": success,
            "budget": adapter.budget().as_dict(),
            "command_count": len(record["commands"]),
        }
    finally:
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = replay(json.loads(args.record.read_text()))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)
