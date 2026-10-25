"""Bounded T2 device/clock probe through the real worker, with owner evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import roboenv
import yaml
from roboenv.client import Client, RemoteError
from roboenv.tasks.resolve import resolve_task


def run(task, interpreter, out, mode, seed):
    out.mkdir(parents=True, exist_ok=False)
    resolved = resolve_task(task)
    value = resolved.task.as_dict()
    value.update(seed=seed, execution_mode=mode, stop_on_success=True)
    value["backend_kwargs"]["max_steps"] = 600
    selected = out / "task.yaml"
    selected.write_text(yaml.safe_dump(value, sort_keys=False))
    with roboenv.launch(
        selected,
        interpreter=interpreter,
        run_dir=out / "runs",
        recording={},
        allow_stop=True,
        timeout=1800,
    ) as owner:
        public = owner.client
        private = Client(owner.record["socket"], session=owner.record["session"], generation=0)
        facts = {d["name"]: d for d in public.devices()["devices"]}
        assert set(facts) == {
            "world",
            "base",
            "head",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "head_camera",
        }
        world = public.device("world")
        initial = private.request("observe")
        initial_head = public.device("head").read()["data"]["positions"]
        camera = public.device("head_camera").read()["data"]
        assert Path(camera["file"]).is_file() and Path(camera["depth_file"]).is_file()
        for name in ("head", "left_arm", "right_arm", "left_hand", "right_hand"):
            home = list(facts[name]["intrinsics"]["home"].values())
            if name == "head":
                home[0] = 0.2
            if name.endswith("_hand"):
                home = [0.3] * len(home)
            public.device(name).write(actions=[home], duration_s=0.5)
        world.wait(0.5)
        readings = {n: public.device(n).read()["data"] for n in facts if n != "world"}
        assert abs(readings["head"]["positions"][0] - initial_head[0]) > 0.05
        for name in ("left_hand", "right_hand"):
            assert max(readings[name]["positions"]) > 0.05
        before = private.request("observe")
        public.device("base").write(
            mode="locomotion", velocity={"vx": 0.1, "vy": 0.0, "wz": 0.0}, duration_s=1.0
        )
        world.wait(1.0)
        public.device("base").stop()
        world.wait(0.5)
        after = private.request("observe")
        displacement = float(
            np.linalg.norm(np.array(after["final"]["root"][:2]) - before["final"]["root"][:2])
        )
        assert displacement > 0.01, "Accepted base writes did not produce measured movement"
        for name, args in (
            (
                "base",
                {
                    "mode": "locomotion",
                    "velocity": {"vx": True, "vy": 0, "wz": 0},
                    "duration_s": 0.1,
                },
            ),
            ("head", {"actions": [[10, 10]], "duration_s": 0.1}),
        ):
            try:
                public.device(name).call("write", args)
            except RemoteError:
                pass
            else:
                raise ValueError("Invalid device command was accepted")
        before_read = private.request("observe")
        public.device("head_camera").read()
        after_read = private.request("observe")
        if mode == "stepped":
            assert before_read["final"] == after_read["final"], "Read advanced physics"
        for name in ("head", "left_arm", "right_arm", "left_hand", "right_hand"):
            before_stop = private.request("observe")["final"]["references"]
            public.device(name).stop()
            after_stop = private.request("observe")["final"]["references"]
            assert before_stop["velocity"] == after_stop["velocity"]
            for other in ("head", "left_hand", "right_hand"):
                if other != name or name.endswith("_hand"):
                    assert before_stop["joints"][other] == after_stop["joints"][other]
        public.request("simulation.stop")
        evidence = Path(owner.record["directory"]) / "evidence"
        record = json.loads((evidence / "final-observation.json").read_text())
        assert record["submitted"] and not record["final"]["fallen"]
        assert record["initialization"] == initial["initialization"]
        (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        (out / "report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "mode": mode,
                    "seed": seed,
                    "base_displacement_m": displacement,
                    "task_success": record["native_success"],
                    "owner": owner.record,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="isaaclab/t2-can-in-bowl")
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("realtime", "stepped"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.task, args.python, args.out, args.mode, args.seed)
