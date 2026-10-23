"""Replay an owner ChemLab command record without restoring terminal state."""

import argparse
import hashlib
import json
from pathlib import Path

from roboenv_runtime.adapters.isaaclab.tasks.chemlab.environment import ChemLabEnvironment


def replay(record, *, asset_directory=None, render=False, out=None, record_sha256=None):
    if record.get("schema") != "roboenv/chemlab-record/1":
        raise ValueError("Unsupported ChemLab record")
    environment = ChemLabEnvironment(
        task_spec=record["task_definition"],
        device_profiles=record["device_profiles"],
        asset_directory=asset_directory,
        max_steps=record["max_steps"],
        execution_mode=record["execution_mode"],
        render=render,
    )
    environment.stop_on_success = record["stop_on_success"]
    try:
        if environment.implementation != record["sources"]:
            raise ValueError("Replay source identity differs")
        if environment.asset_identity != record["assets"]:
            raise ValueError("Replay asset identity differs")
        if environment.dependencies != record["dependencies"]:
            raise ValueError("Replay runtime dependencies differ")
        environment.reset(record["seed"])
        if environment.initial != record["initial_state"]:
            raise ValueError("Replay initial state differs")
        commands = iter(record["commands"])
        command = next(commands, None)
        frames = record["budget"]["used"]
        for frame in range(frames + 1):
            while command is not None and command["frame"] == frame:
                response = environment.invoke(
                    command["instrument"], command["capability"], command["arguments"]
                )
                if response.get("status") != command["status"]:
                    raise ValueError("Replay command outcome differs")
                command = next(commands, None)
            if command is not None and (
                type(command["frame"]) is not int or command["frame"] < frame
            ):
                raise ValueError("Replay commands must be ordered by frame")
            if frame < frames:
                environment.tick()
                if environment.frame != frame + 1:
                    raise ValueError("Replay terminated before its recorded frame")
        if command is not None:
            raise ValueError("Replay command exceeds its final frame")
        final = environment.record()
        matched = all(final[key] == record[key] for key in ("state", "budget", "native_success"))
        scene_matched = None
        if render:
            scene_matched = all(
                final[key] == record[key] for key in ("scene_mappings", "scene_evidence")
            )
            matched = matched and scene_matched
        result = {
            "passed": matched,
            "frames": frames,
            "native_success": final["native_success"],
            "rendered": render,
            "scene_matched": scene_matched,
        }
        if record_sha256:
            result["record_sha256"] = record_sha256
        # Isaac's application close exits the process. Seal evidence before
        # teardown, and never turn an exception into a zero-exit success.
        if out is not None:
            with Path(out).open("x") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        if not matched and render:
            raise ValueError("Rendered replay differs from recorded episode")
        return result
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--directory")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.record.read_bytes()
    result = replay(
        json.loads(raw),
        asset_directory=args.directory,
        render=args.render,
        out=args.out,
        record_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
