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
        before = env.control.snapshot(list(NAMES))
        commands = [
            {
                "device": name,
                "args": (
                    {
                        "mode": "locomotion", "velocity": {"vx": 0, "vy": 0, "wz": 0},
                        "posture": {"height": env.control.targets[name][3]},
                    }
                    if name == "base"
                    else {"target": env.control.targets[name]}
                ),
            }
            for name in NAMES
        ]
        # Invalid batches must have no effect, including their valid prefix.
        invalid = commands + [{"device": "unknown", "args": {"target": []}}]
        try:
            env.control.batch_write(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid batch accepted")
        assert env.control.snapshot(list(NAMES)) == before
        after = env.control.batch_write(commands)
        period = env.model["control"]["execution"]["interval_s"]
        assert after["sequence"] == before["sequence"] + 1
        assert abs(after["sample_time_s"] - before["sample_time_s"] - period) < 1e-8
        assert len({state["sample_time_s"] for state in after["states"].values()}) == 1
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
