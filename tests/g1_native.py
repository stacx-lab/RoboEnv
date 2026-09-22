"""Explicit G1 native contract check; requires generated model and local assets."""

import argparse
import json
import traceback
from pathlib import Path

from roboenv_runtime.adapters.isaaclab.g1 import G1Env
from roboenv_runtime.adapters.isaaclab.g1_profiles import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Override the packaged G1 runtime model")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    env = G1Env(model_path=options.model, asset_manifest=options.assets)
    try:
        env.reset(0)
        before = {name: env.read_device(name) for name in NAMES}
        commands = [
            {
                "device": name,
                "args": (
                    {
                        "mode": "locomotion",
                        "velocity": {"vx": 0, "vy": 0, "wz": 0},
                        "posture": {"height": env.control.targets[name][3]},
                    }
                    if name == "base"
                    else {"target": env.control.targets[name]}
                ),
            }
            for name in NAMES
        ]
        # Invalid individual commands have no effects.
        try:
            env.control.write("unknown", {"target": []})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid command accepted")
        assert {name: env.read_device(name) for name in NAMES} == before
        for command in commands:
            env.control.write(command["device"], command["args"])
        after = {name: env.read_device(name) for name in NAMES}
        period = env.control_interval
        assert (
            abs(
                after["base"]["sample_time_s"]
                - before["base"]["sample_time_s"]
                - period * len(commands)
            )
            < 1e-8
        )
        # A single-component write retains every other target.
        retained = {name: list(value) for name, value in env.control.targets.items()}
        env.control.write("base", commands[0]["args"])
        assert env.control.targets == retained
        stamp = env.sample_time_s
        for name in NAMES:
            env.control.stop(name)
        assert env.sample_time_s == stamp
        assert env.control.targets["base"] == [0, 0, 0, retained["base"][3]]
        receipt = {
            "before": before,
            "after": after,
            "stop_sample_time_s": stamp,
            "done": env.done,
            "success": env.completed,
            "contract_passed": True,
        }
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print("G1 native device contract passed (task completion not asserted)", flush=True)
    except BaseException:
        traceback.print_exc()
        env.close(exit_code=1)
        raise
    env.close()


if __name__ == "__main__":
    main()
