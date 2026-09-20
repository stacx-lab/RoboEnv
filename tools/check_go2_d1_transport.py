"""Exercise Go2–D1 public devices and owner recording through the real worker."""

import argparse
import json
import time
from pathlib import Path

import roboenv
from roboenv.client import Client, RemoteError


def run(out, realtime=False, speed=0.05, task="isaaclab/go2-d1-control"):
    out.mkdir(parents=True, exist_ok=False)
    with roboenv.launch(task, realtime=realtime, timeout=180) as owner:
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
        if realtime:
            public.device("base").call("write", command)
            before = private.request("observe")["final"]["steps"]
            time.sleep(1.2)
            record = private.request("observe")
            assert record["final"]["steps"] > before
            assert any(e.get("reason") == "timeout" for e in record["events"])
        else:
            for i in range(300):
                command["velocity"]["vx"] = speed if 150 <= i < 280 else 0.0
                public.device("base").call("write", command)
            assert private.request("observe")["final"]["steps"] == 300
        before = private.request("observe")
        for name, args in [
            ("base", {**command, "velocity": {"vx": 5, "vy": 0, "wz": 0}}),
            ("arm", {"target": [10.0] * 6}),
            ("gripper", {"target": -1}),
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
        if not realtime:
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
                    "realtime": realtime,
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
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument(
        "--task",
        default="isaaclab/go2-d1-control",
        choices=["isaaclab/go2-d1-control", "isaaclab/go2-d1-reach", "isaaclab/go2-d1-pick-place"],
    )
    args = parser.parse_args()
    run(args.out, args.realtime, args.speed, args.task)
