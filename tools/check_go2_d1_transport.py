"""Exercise Go2–D1 public devices and owner recording through the real worker."""

import argparse
import json
from pathlib import Path

import roboenv
from roboenv.client import Client, RemoteError


def run(out, execution_mode="realtime", speed=0.05, task="isaaclab/go2-d1-control"):
    out.mkdir(parents=True, exist_ok=False)
    with roboenv.launch(task, execution_mode=execution_mode, timeout=180) as owner:
        public = owner.client
        private = Client(
            owner.record["socket"],
            session=owner.record["session"],
            generation=owner.record["generation"],
        )
        names = {d["name"] for d in public.devices()["devices"]}
        assert names == {"base", "arm", "gripper", "camera_head", "camera_wrist", "world"}
        command = {
            "mode": "locomotion",
            "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
            "posture": {"height": 0.30, "roll": 0.0, "pitch": 0.0},
        }
        world = public.device("world")
        started = world.read()["data"]["time_s"]
        for velocity, duration in [(0.0, 3.0), (speed, 2.6), (0.0, 0.4)]:
            command["velocity"]["vx"] = velocity
            receipt = public.device("base").write(**command, duration_s=duration)["data"]
            assert receipt["execution_id"] and receipt["elapsed_s"] == 0
            waited = world.wait(duration)["data"]
            assert duration - 1e-9 <= waited["elapsed_s"] <= duration + 0.02 + 1e-9
        assert world.read()["data"]["time_s"] >= started + 6.0 - 1e-9
        if execution_mode == "stepped":
            assert private.request("observe")["final"]["steps"] == 300
        before = private.request("observe")
        for name, args in [
            ("base", {**command, "duration_s": 0.1, "velocity": {"vx": True, "vy": 0, "wz": 0}}),
            ("arm", {"actions": [[10.0] * 6], "duration_s": 0.1}),
            ("gripper", {"actions": [-1], "duration_s": 0.1}),
        ]:
            try:
                public.device(name).call("write", args)
            except RemoteError:
                pass
            else:
                raise ValueError(f"{name}: invalid command accepted")
        readings = {
            name: public.device(name).read()
            for name in ("base", "arm", "gripper", "camera_head", "camera_wrist")
        }
        for name in ("base", "arm", "gripper"):
            public.device(name).stop()
        record = private.request("observe")
        if execution_mode == "stepped":
            assert record["final"]["steps"] == before["final"]["steps"]
            assert record["final"]["root"] == before["final"]["root"]
            assert record["events"][:-3] == before["events"]
        assert not record["final"]["done"]
        assert record["success_latched"] is False
        (out / "readings.json").write_text(json.dumps(readings, indent=2))
        (out / "record.json").write_text(json.dumps(record, indent=2))
        (out / "report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "execution_mode": execution_mode,
                    "steps": record["final"]["steps"],
                    "run_id": owner.record.get("id"),
                    "task_success": False,
                },
                indent=2,
            )
        )
        print("GO2_TRANSPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=["stepped", "realtime"], default="realtime")
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument(
        "--task",
        default="isaaclab/go2-d1-control",
        choices=["isaaclab/go2-d1-control", "isaaclab/go2-d1-reach", "isaaclab/go2-d1-pick-place"],
    )
    args = parser.parse_args()
    run(args.out, args.execution_mode, args.speed, args.task)
