"""User workflows, composed from the same package APIs used by Python callers."""

import json
import sys
from pathlib import Path

COMMANDS = {"prepare", "doctor", "launch", "status", "stop"}


def add_commands(commands):
    for name in sorted(COMMANDS):
        parser = commands.add_parser(
            name,
            help={
                "prepare": "Install or reuse one isolated simulator environment",
                "doctor": "Explain simulator readiness and next steps",
                "launch": "Start a task and wait for its devices",
                "status": "Inspect a running environment",
                "stop": "Finalize a selected simulation or stop a locally owned environment",
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
        if name == "launch":
            parser.add_argument("task", help="simulator/task_name or task.yaml path")
            parser.add_argument("--catalog", type=Path)
            parser.add_argument(
                "--runtime-root", type=Path, help="Prepared image data root; no source checkout"
            )
            parser.add_argument("--run-dir", type=Path, help="Directory for retained run records")
            parser.add_argument(
                "--python-path",
                type=Path,
                action="append",
                default=[],
                help="Explicit SDK/runtime import path for a prepared deployment",
            )
            parser.add_argument(
                "--allow-stop",
                action="store_true",
                help="Allow attached clients to finalize this simulation",
            )
            parser.add_argument(
                "--advertise-host", help="Service hostname reachable by attached clients"
            )
            gravity = parser.add_mutually_exclusive_group()
            gravity.add_argument(
                "--arm-gravity-compensation",
                dest="arm_gravity_compensation",
                action="store_true",
                help="Enable SIMPLE arm gravity feedforward (default)",
            )
            gravity.add_argument(
                "--no-arm-gravity-compensation",
                dest="arm_gravity_compensation",
                action="store_false",
                help="Disable SIMPLE arm gravity feedforward",
            )
            parser.set_defaults(arm_gravity_compensation=None)
            parser.add_argument(
                "--gpu", type=int,
                help="Host GPU index for SIMPLE Isaac (default: first permitted GPU, else 0)",
            )
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--interpreter", type=Path)
            parser.add_argument("--timeout", type=float, default=900)
            parser.add_argument(
                "--execution-mode", choices=("stepped", "realtime"),
                help="Override the task catalog execution mode"
            )
            parser.add_argument("--detach", action="store_true")
            parser.add_argument(
                "--record", nargs="?", const=True, type=Path, metavar="DIR",
                help="Save experimental spectator MP4s (default: inside the run directory)"
            )
            parser.add_argument(
                "--record-fps", type=float, help="Simulation frames per second (default: 10)"
            )
            parser.add_argument(
                "--record-views", type=Path, help="JSON list of spectator view definitions"
            )
            parser.add_argument(
                "--port",
                type=int,
                default=0,
                help="HTTP port (default: automatically select a free port)",
            )
        if name in {"status", "stop"}:
            target = parser.add_mutually_exclusive_group()
            target.add_argument(
                "run", nargs="?", help="Run reference or endpoint (default: ROBOENV_ENDPOINT)"
            )
            target.add_argument("--endpoint", help="HTTP service URL: http://HOST:PORT")
            target.add_argument(
                "--port", type=int, help="Select the local service at http://127.0.0.1:PORT"
            )


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
        if value.get("recording"):
            print(f"Recording: {value['recording']} (experimental)", flush=True)
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

        if args.record is None and (args.record_fps is not None or args.record_views is not None):
            raise ValueError("--record-fps and --record-views require --record")
        recording = None
        if args.record is not None:
            recording = {}
            if args.record is not True:
                recording["directory"] = args.record
            if args.record_fps is not None:
                recording["fps"] = args.record_fps
            if args.record_views is not None:
                recording["views"] = json.loads(args.record_views.read_text())
        environment = launch(
            args.task,
            execution_mode=args.execution_mode,
            project=args.project,
            catalog=args.catalog,
            interpreter=args.interpreter,
            timeout=args.timeout,
            port=args.port,
            host=args.host,
            arm_gravity_compensation=args.arm_gravity_compensation,
            gpu=args.gpu,
            log_stream=sys.stderr,
            runtime_root=args.runtime_root,
            run_dir=args.run_dir,
            allow_stop=args.allow_stop,
            advertise_host=args.advertise_host,
            python_path=args.python_path,
            recording=recording,
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

        result = (status if args.command == "status" else stop)(
            args.run, endpoint=args.endpoint, port=args.port, project=args.project
        )
    emit(result, machine=args.json)
    return 0
