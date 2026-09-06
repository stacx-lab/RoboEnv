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
    for name in client.api().get("tools", {}):
        namespace[name] = lambda *args, _name=name, **kwargs: client.call(_name, *args, **kwargs)
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    return {
        "program_completed": True,
        "score": client.request("score.peek"),
        "budget": client.request("observe").get("budget"),
        "certified": False,
    }


def main(argv=None):
    from roboenv.cli.workflows import COMMANDS, add_commands, execute

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)
    for name in ("describe", "devices", "observe", "scene", "score"):
        command = commands.add_parser(name)
        if name == "devices":
            command.add_argument("--name", help="Describe one device")
    api = commands.add_parser("api")
    api.add_argument("--name")
    call = commands.add_parser("call")
    call.add_argument("name")
    call.add_argument("args", nargs="?", default="{}")
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
    generation = tasks_commands.add_parser("generate")
    generation.add_argument("--project", type=Path)
    generation.add_argument(
        "--source", type=Path, help="Optional normalized catalog; otherwise discover pinned sources"
    )
    generation.add_argument("--deployments", required=True, type=Path)
    generation.add_argument("--wheel", required=True, type=Path)
    generation.add_argument("--out", required=True, type=Path)
    generation.add_argument("--backends", nargs="+")
    generation.add_argument("--ids", nargs="+")
    generation.add_argument("--seeds", nargs="+", type=int)
    generation.add_argument("--include-references", action="store_true")
    generation.add_argument("--check-images", action="store_true")
    validation = tasks_commands.add_parser("validate")
    validation.add_argument("task", type=Path)
    run = tasks_commands.add_parser("run")
    run.add_argument("bundle", type=Path)
    run.add_argument("--deployments", required=True, type=Path)
    run.add_argument("--agent", required=True)
    run.add_argument("--model")
    for name in (
        "describe",
        "devices",
        "observe",
        "scene",
        "score",
        "api",
        "call",
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
        client = Client()
        if hasattr(args, "run"):
            from roboenv.environment import connect

            client = connect(args.run, endpoint=args.endpoint)
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
            from roboenv.harbor.taskset import export_taskset, run_taskset, validate_harbor_task
            from roboenv.tasks.catalog import BACKENDS, discover, summary

            if args.tasks_command == "list":
                result = summary(args.project, args.source)
                if args.simulator:
                    from roboenv.project import project_root
                    from roboenv.tasks.catalog import load_catalog

                    entries = load_catalog(
                        args.source or project_root(args.project) / "catalog/tasks"
                    )
                    ids = sorted(t.id for t, _ in entries.values() if t.backend == args.simulator)
                    result = {"tasks": len(ids), "backends": {args.simulator: len(ids)}, "ids": ids}
            elif args.tasks_command == "discover":
                result = discover(args.project, args.out, args.backends or BACKENDS)
            elif args.tasks_command == "generate":
                result = export_taskset(
                    args.project,
                    args.source,
                    args.deployments,
                    args.out,
                    args.wheel,
                    backends=args.backends or BACKENDS,
                    identifiers=args.ids,
                    seeds=args.seeds,
                    include_references=args.include_references,
                    check_images=args.check_images,
                )
            elif args.tasks_command == "validate":
                result = validate_harbor_task(args.task)
            else:
                return run_taskset(
                    args.bundle, args.deployments, agent=args.agent, model=args.model
                )
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
        elif args.command == "scene":
            result = client.device("world").read()
        elif args.command == "api":
            result = client.api()
            if args.name:
                result = result["tools"][args.name]
        elif args.command == "call":
            result = client.call(args.name, **json_loads(args.args))
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
