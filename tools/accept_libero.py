"""Run the reproducible LIBERO infrastructure gates without invoking a model.

Successful completion certifies device/control/transport/recording/replay checks.
Task-solving acceptance is a separate Kinex frozen-controller campaign.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src")),
        "ROBOENV_LIBERO_TEST_PYTHON": str(args.python),
        "ROBOENV_LIBERO_ACCEPTANCE": "1",
        "NODE": args.node,
        "KINEX_ROOT": str(args.kinex),
        "LIBERO_CONFIG_PATH": str(args.out / "native-config"),
        "MUJOCO_GL": "egl",
    }
    stages = [
        (
            "contracts",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_libero_devices.py",
                "-q",
                "-o",
                "tmp_path_retention_policy=all",
                "--basetemp",
                str(args.out / "contracts"),
            ],
            900,
        ),
        (
            "camera",
            [str(args.python), "tests/libero_camera.py", "--out", str(args.out / "camera")],
            180,
        ),
        ("http", [str(args.python), "tests/libero_http.py", "--out", str(args.out / "http")], 300),
        (
            "kinex",
            [
                str(args.python),
                "tests/libero_kinex.py",
                "--out",
                str(args.out / "kinex"),
                "--kinex",
                str(args.kinex),
                "--node",
                args.node,
            ],
            360,
        ),
    ]
    report = {
        "schema": "roboenv/libero-acceptance/1",
        "passed": False,
        "model_calls": 0,
        "full_task_attempted": False,
        "stages": [],
    }
    for name, command, timeout in stages:
        started = time.monotonic()
        failure = None
        with (args.out / f"{name}.log").open("x") as log:
            try:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout
                )
                exit_code = result.returncode
            except (subprocess.TimeoutExpired, OSError) as error:
                exit_code, failure = 1, str(error)
        report["stages"].append(
            {
                "name": name,
                "command": command,
                "seconds": time.monotonic() - started,
                "exit_code": exit_code,
                "error": failure,
            }
        )
        if exit_code:
            break
    report["passed"] = len(report["stages"]) == len(stages) and all(
        stage["exit_code"] == 0 for stage in report["stages"]
    )
    report["artifacts"] = {
        str(file.relative_to(args.out)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(args.out.rglob("*.json"))
        if file.is_file() and "site-packages" not in file.parts
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-sim/libero/bin/python")
    parser.add_argument("--kinex", type=Path, default=ROOT.parent / "kinex")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    args.out, args.python, args.kinex = (
        args.out.resolve(),
        args.python.expanduser().absolute(),
        args.kinex.resolve(),
    )
    report = run(args)
    print(json.dumps({"passed": report["passed"], "report": str(args.out / "report.json")}))
    raise SystemExit(0 if report["passed"] else 1)
