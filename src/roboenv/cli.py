"""Agent-neutral command line for the environment contract."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from roboenv.client import Client, RemoteError, UncertainResult, jsonable
from roboenv.devices import invoke, list_devices
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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("describe", "devices", "observe", "scene", "score"):
        command = commands.add_parser(name)
        if name in {"describe", "devices"}:
            command.add_argument("--local", action="store_true", help="Read offline task files")
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
    device.add_argument("--local", action="store_true", help="Use a legacy local driver")
    device.add_argument("--request-id")
    device.add_argument("--timeout", type=float, default=600)
    operation = commands.add_parser("operation")
    operation.add_argument("request_id")
    program = commands.add_parser("replay")
    program.add_argument("program", type=Path)
    export = commands.add_parser("export-harbor")
    export.add_argument("task", type=Path)
    export.add_argument("--catalog", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--deployment", required=True, type=Path)
    export.add_argument("--wheel", required=True, type=Path)
    export.add_argument("--include-reference", action="store_true")
    sources = commands.add_parser("sources")
    sources.add_argument("operation", choices=("status", "sync"))
    sources.add_argument("--project", type=Path)
    sources.add_argument("--backends", nargs="+")
    privacy = commands.add_parser("audit-privacy")
    privacy.add_argument("--project", type=Path)
    tasks = commands.add_parser("tasks")
    tasks_commands = tasks.add_subparsers(dest="tasks_command", required=True)
    listing = tasks_commands.add_parser("list")
    listing.add_argument("--project", type=Path)
    listing.add_argument("--source", type=Path)
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
    args = parser.parse_args(argv)
    try:
        client = Client()
        if args.command == "sources":
            from .upstreams import status, sync

            result = (sync if args.operation == "sync" else status)(args.project, args.backends)
            print(json.dumps(result))
            return 0 if result["ready"] else 1
        elif args.command == "audit-privacy":
            from .privacy import audit
            from .project import project_root

            result = audit(project_root(args.project))
            print(json.dumps(result))
            return 0 if result["passed"] else 1
        elif args.command == "tasks":
            from .catalogue import BACKENDS, discover, summary
            from .taskset import export_taskset, run_taskset, validate_harbor_task

            if args.tasks_command == "list":
                result = summary(args.project, args.source)
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
            import os

            result = (
                client.describe()
                if not args.local
                else json.loads(
                    Path(os.environ.get("ROBOENV_MANIFEST", "/env/environment.json")).read_text()
                )
            )
        elif args.command == "devices":
            if args.local:
                result = list_devices()
                if args.name:
                    result = result[args.name]
            else:
                result = client.device(args.name).describe() if args.name else client.devices()
        elif args.command == "device":
            result = (
                invoke(args.name, args.action, json_loads(args.args))
                if args.local
                else client.device(args.name).call(
                    args.action,
                    json_loads(args.args),
                    request_id=args.request_id,
                    timeout=args.timeout,
                )
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
        elif args.command == "export-harbor":
            from roboenv.harbor import export_task

            result = export_task(
                args.task,
                args.catalog,
                args.out,
                None,
                args.wheel,
                deployment_path=args.deployment,
                include_reference=args.include_reference,
            )
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
