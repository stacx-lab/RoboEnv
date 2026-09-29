"""Agent-neutral command line for the environment contract."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from roboenv.client import Client, RemoteError, UncertainResult, jsonable
from roboenv.validation import json_loads


def replay(path, client):
    """Execute a candidate in the agent workspace using only public environment functions."""
    import numpy as np

    namespace = {"np": np, "__name__": "__main__"}
    from roboenv.devices.client import Environment

    namespace["env"] = Environment(client)
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    # Public operations only: the program ran where an agent would, often over HTTP, where the
    # score and the owner's observation are not offered. The verifier alone judges the result.
    return {
        "program_completed": True,
        "simulation": client.request("simulation.status"),
        "certified": False,
    }


def verify(run, out, project=None):
    """Certify one local run from its launch record: owner socket, identity and journal."""
    from roboenv.verify import certify, refuse

    try:
        from roboenv.environment import _record, _record_path

        path = _record_path(run, project)
        record = _record(path, project)
        # The manifest beside run.json is the task as launch resolved it, seed and mode included.
        task = json_loads((path.parent / "environment.json").read_text())["task"]
        if task["id"] != record["task"]:
            raise ValueError("run record and launch manifest name different tasks")
        identity = {
            "socket_path": record["socket"],
            "session": record["session"],
            "generation": record["generation"],
            "task_id": task["id"],
            "seed": task["seed"],
            "budget_mode": task["budget_mode"],
            "journal": Path(record["directory"]) / "evidence/journal/actions.jsonl",
        }
    except Exception as exc:
        # A run that cannot be identified is not certified; stale outputs are still replaced.
        return refuse(out, f"{type(exc).__name__}: {exc}")
    return certify(out=out, **identity)


def main(argv=None):
    from roboenv.cli.workflows import COMMANDS, add_commands, execute

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)
    for name in ("describe", "devices", "observe", "score"):
        command = commands.add_parser(name)
        if name == "devices":
            command.add_argument("--name", help="Describe one device")
    device = commands.add_parser("device")
    device.add_argument("name")
    device.add_argument("action", nargs="?", default="read")
    device.add_argument("args", nargs="?", default="{}")
    device.add_argument("--request-id")
    device.add_argument("--timeout", type=float, default=600)
    operation = commands.add_parser("operation")
    operation.add_argument("request_id")
    program = commands.add_parser("replay")
    program.add_argument("program", type=Path)
    certification = commands.add_parser(
        "verify", help="Certify a finished local run: submit, then validate score and journal"
    )
    certification.add_argument("--run", required=True, help="Local run ID or run.json")
    certification.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory for submit.json, reward.json and the retained actions.jsonl",
    )
    certification.add_argument("--project", type=Path)
    sources = commands.add_parser("sources")
    sources.add_argument("operation", choices=("status", "sync", "check-updates"))
    sources.add_argument("--project", type=Path)
    sources.add_argument("--backends", nargs="+")
    privacy = commands.add_parser("audit-privacy")
    privacy.add_argument("--project", type=Path)
    tasks = commands.add_parser("tasks")
    tasks_commands = tasks.add_subparsers(dest="tasks_command", required=True)
    listing = tasks_commands.add_parser("list")
    listing.add_argument("--project", type=Path)
    listing.add_argument("--source", type=Path)
    listing.add_argument("--simulator", help="Filter tasks by simulator")
    discovery = tasks_commands.add_parser("discover")
    discovery.add_argument("--project", type=Path)
    discovery.add_argument("--out", required=True, type=Path)
    discovery.add_argument("--backends", nargs="+")
    for name in (
        "describe",
        "devices",
        "observe",
        "score",
        "device",
        "operation",
        "replay",
    ):
        target = commands.choices[name].add_mutually_exclusive_group()
        target.add_argument("--run", help="Local run ID or run.json")
        target.add_argument("--endpoint", help="HTTP service URL: http://HOST:PORT")
    args = parser.parse_args(argv)
    try:
        if args.command in COMMANDS:
            return execute(args)
        if args.command == "verify":
            report = verify(args.run, args.out, project=args.project)
            print(json.dumps(report, allow_nan=False))
            return 0 if report["valid"] else 1
        client = Client()
        if hasattr(args, "run"):
            from roboenv.environment import _connect as connect

            client = connect(
                args.run, endpoint=args.endpoint, owner=args.command in {"observe", "score"}
            )
        if args.command == "sources":
            from roboenv.provisioning.sources import check_updates, status, sync

            operation = {"status": status, "sync": sync, "check-updates": check_updates}[
                args.operation
            ]
            result = operation(args.project, args.backends)
            print(json.dumps(result))
            return (
                0 if result["up_to_date" if args.operation == "check-updates" else "ready"] else 1
            )
        elif args.command == "audit-privacy":
            from roboenv.privacy import audit
            from roboenv.project import project_root

            result = audit(project_root(args.project))
            print(json.dumps(result))
            return 0 if result["passed"] else 1
        elif args.command == "tasks":
            from roboenv.tasks.catalog import BACKENDS, discover, summary

            if args.tasks_command == "list":
                if args.simulator:
                    from roboenv.project import project_root
                    from roboenv.tasks.catalog import load_catalog

                    entries = load_catalog(
                        args.source or project_root(args.project) / "catalog/tasks"
                    )
                    ids = sorted(t.id for t, _ in entries.values() if t.backend == args.simulator)
                    result = {
                        "tasks": len(ids), "backends": {args.simulator: len(ids)}, "ids": ids,
                        "execution_modes": {key: entries[key][0].execution_mode for key in ids},
                    }
                else:
                    result = summary(args.project, args.source)
            else:
                result = discover(args.project, args.out, args.backends or BACKENDS)
        elif args.command == "describe":
            result = client.describe()
        elif args.command == "devices":
            result = client.device(args.name).describe() if args.name else client.devices()
        elif args.command == "device":
            result = client.device(args.name).call(
                args.action,
                json_loads(args.args),
                request_id=args.request_id,
                timeout=args.timeout,
            )
        elif args.command == "operation":
            result = client.operation(args.request_id)
        elif args.command == "replay":
            result = replay(args.program, client)
        else:
            result = client.request({"score": "score.peek", "observe": "observe"}[args.command])
        print(json.dumps(result, default=jsonable, allow_nan=False))
        return 0
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, (RemoteError, UncertainResult)):
            error.update(code=getattr(exc, "code", "outcome_unknown"), request_id=exc.request_id)
        if isinstance(exc, RemoteError):
            error["details"] = exc.details
        print(json.dumps(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
