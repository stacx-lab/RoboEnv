"""Exercise a native clock through its real worker, then replay the frozen record."""

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import roboenv
import yaml
from check_native_transport import camera_name, owner_client, require, target
from finalize_episode import finalize
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def check(task, interpreter, out, *, timeout=900):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    resolved = resolve_task(task, project=ROOT)
    configured = replace(
        resolved.task, backend_kwargs={**resolved.task.backend_kwargs, "max_steps": 300}
    )
    path = out / "task.yaml"
    path.write_text(yaml.safe_dump(asdict(configured), sort_keys=False))
    report = {
        "passed": False,
        "task": configured.id,
        "checks": {},
        "scope": "native worker regression; no model or task-success claim",
    }
    try:
        with roboenv.launch(path, project=ROOT, interpreter=interpreter, timeout=timeout) as owner:
            private, public = owner_client(owner), owner.client
            before = private.request("observe")["budget"]["used"]
            time.sleep(0.5)
            after = private.request("observe")["budget"]["used"]
            require(after == before, "Wrong idle clock behavior")
            report["checks"]["idle_clock"] = True
            camera = camera_name(public)
            require(public.device(camera).read()["data"].get("file"), "Missing native camera")
            arm = public.device("left_arm")
            measured = arm.read()["data"]["positions"]
            goal = target(public, "left_arm", 0.05)
            frequency = 50
            require(
                arm.write([goal] * 16, freq_hz=frequency)["data"]["completed_actions"] == 16,
                "Sequence did not complete",
            )
            current = arm.read()["data"]["positions"]
            require(
                max(abs(a - b) for a, b in zip(current, measured, strict=True)) > 0.01,
                "Sequence produced no measured motion",
            )
            arm.stop()
            report["checks"]["camera_and_motion"] = True
            finalized = finalize(owner.path, out / "final")
            frozen = private.request("observe")
            time.sleep(0.2)
            require(private.request("observe") == frozen, "Submitted world did not freeze")
            report["checks"]["terminal_freeze"] = True
            report["native_success"] = finalized["verdict"]["completed"]
        command = [
            str(interpreter),
            str(ROOT / "tools/replay_native_devices.py"),
            str(out / "final/owner-record.json"),
            "--out",
            str(out / "replay.json"),
        ]
        report["replay_command"] = command
        with (out / "replay.log").open("x") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src")),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
            )
        replay = json.loads((out / "replay.json").read_text())
        require(replay.get("passed") is True, "Replay did not produce passing evidence")
        report["checks"]["fresh_process_replay"] = True
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    print(json.dumps(check(args.task, args.interpreter, args.out, timeout=args.timeout)))
