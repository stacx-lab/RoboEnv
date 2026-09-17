"""User workflows, composed from the same package APIs used by Python callers."""

import json
import sys
from pathlib import Path

COMMANDS = {"prepare", "doctor", "launch", "status", "stop", "export"}


def add_commands(commands):
    for name in sorted(COMMANDS):
        parser = commands.add_parser(
            name,
            help={
                "prepare": "Install or reuse one isolated simulator environment",
                "doctor": "Explain simulator readiness and next steps",
                "launch": "Start a task and wait for its devices",
                "status": "Inspect a running environment",
                "stop": "Stop a locally owned environment",
                "export": "Export one task to Harbor",
            }[name],
        )
        parser.add_argument("--project", type=Path)
        parser.add_argument("--json", action="store_true", help="Machine-readable output")
        if name in {"prepare", "doctor"}:
            parser.add_argument("simulator")
        if name == "prepare":
            parser.add_argument(
                "--prefix", type=Path, help="New environment (default: .venv-sim/SIMULATOR)"
            )
        if name in {"launch", "export"}:
            parser.add_argument("task", help="simulator/task_name or task.yaml path")
            parser.add_argument("--catalog", type=Path)
        if name == "launch":
            clock = parser.add_mutually_exclusive_group()
            clock.add_argument(
                "--realtime", dest="realtime", action="store_true",
                help="Run wall-clock-paced physics on a supported backend",
            )
            clock.add_argument(
                "--sync", dest="realtime", action="store_false",
                help="Advance one control interval per write on a supported backend",
            )
            parser.set_defaults(realtime=None)
            gravity = parser.add_mutually_exclusive_group()
            gravity.add_argument(
                "--arm-gravity-compensation", dest="arm_gravity_compensation",
                action="store_true", help="Enable SIMPLE arm gravity feedforward (default)",
            )
            gravity.add_argument(
                "--no-arm-gravity-compensation", dest="arm_gravity_compensation",
                action="store_false", help="Disable SIMPLE arm gravity feedforward",
            )
            parser.set_defaults(arm_gravity_compensation=None)
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--interpreter", type=Path)
            parser.add_argument("--timeout", type=float, default=900)
            parser.add_argument("--detach", action="store_true")
            parser.add_argument(
                "--port",
                type=int,
                default=0,
                help="HTTP port (default: automatically select a free port)",
            )
        if name in {"status", "stop"}:
            parser.add_argument("run", nargs="?" if name == "status" else None)
        if name == "export":
            parser.add_argument("--out", required=True, type=Path)
            parser.add_argument("--deployment", type=Path)
            parser.add_argument("--wheel", type=Path)


def emit(value, *, machine=False):
    if machine:
        print(json.dumps(value, default=str), flush=True)
    elif "next_steps" in value:
        print(f"Sources and recipe: {'ready' if value['install_preflight_passed'] else 'blocked'}")
        print(f"Registered interpreter: {'present' if value['installed'] else 'missing'}")
        for blocker in value.get("blockers", []):
            print(f"  {blocker}")
        for step in value["next_steps"]:
            print(f"Next: {step}")
        if value.get("downloaded_assets_verified"):
            print("Kitchen assets: downloaded and content-verified; simulator boot unverified")
        else:
            print(f"Assets: {value['asset_instructions']}")
    elif value.get("schema") == "roboenv/local-run/1":
        print(
            f"Ready: {value['task']}\nEndpoint: {value['endpoint']}\nDevices: {value['devices']}",
            flush=True,
        )
    else:
        print(json.dumps(value, indent=2, default=str), flush=True)


def execute(args):
    if args.command in {"prepare", "doctor"}:
        from roboenv.provisioning.setup import doctor, prepare

        prepared = {}
        if args.command == "prepare":

            def progress(message):
                print(f"[prepare] {message}", file=sys.stderr, flush=True)

            prepared = (
                prepare(
                    args.simulator, project=args.project, install_to=args.prefix, progress=progress
                )
                or {}
            )
        result = doctor(args.simulator, project=args.project)
        if "downloaded_assets_verified" in prepared:
            result["downloaded_assets_verified"] = prepared["downloaded_assets_verified"]
        emit(result, machine=args.json)
        return 0 if result["install_preflight_passed"] else 1
    if args.command == "launch":
        from roboenv.environment import launch

        environment = launch(
            args.task,
            project=args.project,
            catalog=args.catalog,
            interpreter=args.interpreter,
            timeout=args.timeout,
            port=args.port,
            host=args.host,
            realtime=args.realtime,
            arm_gravity_compensation=args.arm_gravity_compensation,
            log_stream=sys.stderr,
        )
        emit(environment.record, machine=args.json)
        if args.detach:
            environment.stop_logs()
            return 0
        try:
            return environment.process.wait()
        except KeyboardInterrupt:
            return 130
        finally:
            environment.close()
    if args.command in {"status", "stop"}:
        from roboenv.environment import status, stop

        result = (status if args.command == "status" else stop)(args.run, project=args.project)
    else:
        from roboenv.harbor.workflow import export

        result = export(
            args.task,
            project=args.project,
            catalog=args.catalog,
            out=args.out,
            deployment=args.deployment,
            wheel=args.wheel,
        )
    emit(result, machine=args.json)
    return 0
