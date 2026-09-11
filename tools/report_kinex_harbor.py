"""Collect real Harbor trial receipts without confusing bridge execution with task success."""

import argparse
import json
from pathlib import Path


def collect(directory, *, require_task_success=False):
    directory = Path(directory)
    job = json.loads((directory / "result.json").read_text())
    trials = []
    for path in sorted(directory.glob("*/result.json")):
        trial = {"name": path.parent.name, "passed": False, "native_success": False}
        try:
            result = json.loads(path.read_text())
            agent = path.parent / "agent"
            execution = json.loads((agent / "execution.json").read_text())
            integrity = json.loads((agent / "integrity.json").read_text())
            verifier = json.loads((path.parent / "verifier/submit.json").read_text())
            requested = json.loads((agent / "runner.json").read_text())["recording"]
            media = json.loads((agent / "media.json").read_text()) if requested else None
            checks = {
                "harbor_completed": bool(result.get("finished_at"))
                and result.get("exception_info") is None,
                "kinex_execution": execution.get("passed") is True,
                "integrity": integrity.get("passed") is True,
                "verifier": verifier.get("valid") is True,
                "requested_recording": not requested or media.get("passed") is True,
            }
            success = verifier.get("trial_success") is True
            trial.update(
                checks=checks,
                native_success=success,
                recording_verified=bool(requested) and checks["requested_recording"],
                mode=execution.get("mode"),
                passed=all(checks.values()) and (success or not require_task_success),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            trial["error"] = f"{type(error).__name__}: {error}"
        trials.append(trial)
    complete = bool(job.get("finished_at")) and len(trials) == job.get("n_total_trials")
    return {
        "schema": "roboenv/kinex-harbor-report/1",
        "passed": complete and bool(trials) and all(item["passed"] for item in trials),
        "task_success_required": require_task_success,
        "trials": trials,
        "fresh_process_replay_verified": False,
        "scope": "Harbor agent/verifier execution; native replay is a separate gate",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-task-success", action="store_true")
    args = parser.parse_args()
    report = collect(args.job, require_task_success=args.require_task_success)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
