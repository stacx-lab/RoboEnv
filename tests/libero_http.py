"""Real HTTP episode isolation and fresh-process owner replay, without a model."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import roboenv
import yaml
from roboenv.client import Client, RemoteError
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    task = yaml.safe_load(resolve_task("libero/bowl-onto-plate").path.read_text())
    task["backend_kwargs"]["realtime"] = False
    task_path = out / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    launch_options = {"catalog": ROOT / "catalog", "port": 0, "interpreter": sys.executable}
    with roboenv.launch(task_path, **launch_options) as first:
        with roboenv.launch(task_path, **launch_options) as second:
            a, b = roboenv.connect(first.endpoint), roboenv.connect(second.endpoint)
            assert dict(a.task.context) == {}
            assert set(first.client.request("environment.describe")) == {
                "task",
                "devices",
                "session",
                "generation",
            }
            before_b = b.device("arm").read()
            public = Client(first.endpoint)
            owner = Client(
                first.record["socket"],
                session=first.record["session"],
                generation=first.record["generation"],
            )
            owner.allowed_ops = owner.allowed_ops | {"submit"}
            owner.request("observe", request_id="owner-private")
            for operation in ("observe", "score.peek", "api", "capture", "reset"):
                public.allowed_ops = public.allowed_ops | {operation}
                try:
                    public.request(operation)
                except RemoteError:
                    pass
                else:
                    raise AssertionError("Owner operation exposed over HTTP")
            try:
                public.operation("owner-private")
            except RemoteError:
                pass
            else:
                raise AssertionError("Owner cache was exposed")
            command = {
                "device": "arm",
                "action": "write",
                "args": {"target": a.device("arm").read()["positions"]},
            }
            before = owner.request("observe")["budget"]["used"]
            x = public.request("devices.call", command, request_id="deduplicated-motion")
            y = public.request("devices.call", command, request_id="deduplicated-motion")
            assert x == y
            assert owner.request("observe")["budget"]["used"] == before + 1
            assert b.device("arm").read() == before_b
            for camera in ("camera", "camera_wrist"):
                path = Path(a.device(camera).read()["file"])
                assert path.read_bytes().startswith(b"\x89PNG")
            # Closing a public connection leaves this episode intact.
            identity = (public.session, public.generation)
            a.close()
            a = roboenv.connect(first.endpoint)
            assert a.device("arm").read()["sample_time_s"] >= 0.3
            assert (public.session, public.generation) == identity
            verdict = owner.request("submit")
            record = owner.request("observe")
            (out / "owner-record.json").write_text(json.dumps(record, indent=2) + "\n")
            (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
            used = record["budget"]["used"]
            assert owner.request("observe")["budget"]["used"] == used
            endpoint = first.endpoint
            a.close()
            b.close()
    # A replacement episode on the same port must reject established clients.
    port = int(endpoint.rsplit(":", 1)[1])
    with roboenv.launch(
        task_path, catalog=ROOT / "catalog", port=port, interpreter=sys.executable
    ) as replacement:
        try:
            public.request("devices.list")
        except RemoteError:
            pass
        else:
            raise AssertionError("Old client reached a replacement episode")
        assert replacement.endpoint == endpoint
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/replay_libero_devices.py"),
            str(out / "owner-record.json"),
            "--out",
            str(out / "replay.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (out / "report.json").write_text(
        json.dumps({"passed": True, "model_calls": 0, "task_success_claimed": False}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    check(args.out)
