"""Exercise SIMPLE's realtime scheduler and base timeout over the public service."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import roboenv
import yaml
from roboenv.tasks.resolve import resolve_task
from simple_kinex import ROOT, tool


def check(out, sim_mode, arm_gravity_compensation=True):
    out.mkdir(parents=True, exist_ok=False)
    definition = yaml.safe_load(resolve_task("simple/g1-xmove-pick", project=ROOT).path.read_text())
    definition["backend_kwargs"]["sim_mode"] = sim_mode
    task = out / "task.yaml"
    task.write_text(yaml.safe_dump(definition, sort_keys=False))
    with roboenv.launch(task, project=ROOT, interpreter=sys.executable, realtime=True,
                        arm_gravity_compensation=arm_gravity_compensation) as run:
        with roboenv.connect(run.endpoint) as client:
            base = client.device("base")
            stamp = base.read()["sample_time_s"]
            base.write(mode="locomotion", velocity={"vx": 0.15, "vy": 0, "wz": 0},
                       posture={"height": 0.74, "roll": 0.03, "pitch": 0.05, "yaw": 0.08})
            time.sleep(0.85)
            assert base.read()["sample_time_s"] > stamp
        tool("finalize_episode").finalize(run.path, out / "owner")
    record = json.loads((out / "owner/owner-record.json").read_text())
    stops = [e for e in record["commands"] if e.get("reason") == "timeout"]
    assert len(stops) == 1 and stops[0]["device"] == "base", stops
    assert record["pending_targets"]["base"][:3] == [0, 0, 0]
    assert record["pending_targets"]["base"][3:] == [0.74, 0.03, 0.05, 0.08]
    assert record["budget"]["max"] is None
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME"}}
    subprocess.run([sys.executable, str(ROOT / "tools/replay_simple_devices.py"),
                    str(out / "owner/owner-record.json"), "--out", str(out / "replay.json")],
                   env=env, check=True, timeout=300)
    assert json.loads((out / "replay.json").read_text())["passed"]
    (out / "report.json").write_text(json.dumps({"passed": True, "timeout_stops": stops,
        "budget": record["budget"], "native_success": record["native_success"]}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco")
    parser.add_argument("--arm-gravity-compensation", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    check(args.out.resolve(), args.sim_mode, args.arm_gravity_compensation)
