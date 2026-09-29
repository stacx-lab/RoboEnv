"""Bounded SIMPLE native/transport/Kinex checks and replay, without model calls."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINEX_TASKS = {
    "xmove": "simple/g1-xmove-pick",
    "transfer": "simple/g1-transfer-between-tables",
    "bend": "simple/g1-bend-pick",
}
TASKS = {
    "xmove": "simple/G1WholebodyXMovePickTeleop-v0",
    "transfer": "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    "bend": "simple/G1WholebodyBendPickTeleop-v0",
}


def run(args):
    args.out = args.out.resolve()
    args.python = args.python.absolute()
    args.out.mkdir(parents=True, exist_ok=False)
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env["PYTHONNOUSERSITE"] = "1"

    def stage(name, command, receipt):
        started = time.monotonic()
        error = None
        with (args.out / f"{name}.log").open("x") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                code = result.returncode
            except (subprocess.TimeoutExpired, OSError) as exc:
                code, error = 1, str(exc)
        passed = code == 0 and receipt.is_file() and json.loads(receipt.read_text())["passed"]
        return {
            "name": name,
            "command": command,
            "seconds": time.monotonic() - started,
            "passed": passed,
            "exit_code": code,
            "error": error,
        }

    def native(task):
        name = task + "-sync"
        directory = args.out / name
        command = [
            str(args.python),
            "tests/simple_native.py",
            "--out",
            str(directory),
            "--env-id",
            TASKS[task],
            "--sim-mode",
            args.sim_mode,
            "--kinex",
            str(args.kinex.resolve()),
        ]
        if not args.arm_gravity_compensation:
            command.append("--no-arm-gravity-compensation")
        checks = [stage(name, command, directory / "report.json")]
        if checks[-1]["passed"]:
            checks.append(
                stage(
                    name + "-replay",
                    [
                        str(args.python),
                        "tools/replay_simple_devices.py",
                        str(directory / "record.json"),
                        "--out",
                        str(directory / "replay.json"),
                    ],
                    directory / "replay.json",
                )
            )
        return checks

    cases = list(args.tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        checks = [item for group in pool.map(native, cases) for item in group]
    if all(check["passed"] for check in checks):
        record = args.out / (args.tasks[0] + "-sync") / "record.json"
        for case in ("commands", "state", "verdict", "gravity"):
            name = "reject-" + case
            checks.append(
                stage(
                    name,
                    [
                        str(args.python),
                        "tests/simple_replay_negative.py",
                        str(record),
                        "--case",
                        case,
                        "--out",
                        str(args.out / f"{name}.json"),
                    ],
                    args.out / f"{name}.json",
                )
            )
    if all(check["passed"] for check in checks) and not args.native_only:
        for task in cases:
            name = f"kinex-{task}-sync"
            command = [
                str(args.python),
                "tests/simple_kinex.py",
                "--out",
                str(args.out / name),
                "--kinex",
                str(args.kinex.resolve()),
                "--node",
                args.node,
                "--sim-mode",
                args.sim_mode,
                "--task",
                KINEX_TASKS[task],
            ]
            if not args.arm_gravity_compensation:
                command.append("--no-arm-gravity-compensation")
            checks.append(stage(name, command, args.out / name / "report.json"))
    report = {
        "schema": "roboenv/simple-acceptance/1",
        "passed": all(c["passed"] for c in checks),
        "arm_gravity_compensation": args.arm_gravity_compensation,
        "mode": "native-only" if args.native_only else "no-model-infrastructure",
        "model_calls": 0,
        "task_performance_claimed": False,
        "stages": checks,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--kinex", type=Path,
        default=Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")),
    )
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco_isaac")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per child stage")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument(
        "--arm-gravity-compensation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    raise SystemExit(0 if run(parser.parse_args()) else 1)
