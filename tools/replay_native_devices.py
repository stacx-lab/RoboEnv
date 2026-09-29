"""Replay source-native device commands in a fresh simulator process, owner only."""

import argparse
import json
from pathlib import Path

from roboenv_runtime.adapters.native_record import compare, digest
from roboenv_runtime.core.adapter import make_adapter

TOLERANCES = {"robotwin": 1e-5, "robodojo": 1e-4, "behavior": 1e-4}


def replay(record, *, out=None):
    if record.get("schema") != "roboenv/native-device-episode/1":
        raise ValueError("Unsupported native episode record")
    backend = record["backend"]
    if backend not in TOLERANCES:
        raise ValueError("Unsupported native replay backend")
    mode = record.get("execution_mode")
    if mode not in ("stepped", "realtime"):
        raise ValueError("Replay requires execution_mode: stepped or realtime")
    budget = record["budget"]
    final_frame = budget["used"]
    if (type(final_frame) is not int or final_frame < 0
            or (budget["max"] is not None and final_frame > budget["max"])):
        raise ValueError("Invalid recorded action-frame budget")
    adapter = make_adapter(backend, **record["environment"])
    report = {"schema": "roboenv/native-replay/1", "passed": False}
    try:
        adapter.configure_execution_mode(mode)
        adapter.reset(record["initialization"]["requested_seed"])
        native = adapter._env
        # Optional native imports can expose vendored distribution metadata.
        # Compare at the same lifecycle boundary as the recorded world.
        if native.implementation_identity != record["implementation"]:
            raise ValueError("Replay implementation or installed dependencies differ")
        if adapter.embodiment != record["embodiment"]:
            raise ValueError("Replay embodiment differs")
        # Work/time spent in an expert prepass is diagnostic. Its candidate-free
        # final initialization, source rig and task configuration must match.
        volatile = {
            "wall_time_s",
            "expert_physics_steps",
            "physics_steps",
            "loaded_state_sha256",
            "state_sha256",
            "native_clock_start_s",
            "native_context_start_index",
        }
        expected = {
            key: value for key, value in record["initialization"].items() if key not in volatile
        }
        actual = {key: value for key, value in native.initialization.items() if key not in volatile}
        compare(expected, actual, atol=1e-10, path="initialization")
        initial_error = compare(
            record["initial_state"],
            native.initial_state,
            atol=TOLERANCES[backend],
            path="initial_state",
        )
        # Reproduce the recorded control frames explicitly. This changes only
        # who schedules the same native interval; it never restores final state.
        for event in record["commands"]:
            frame = event["frame"]
            if type(frame) is not int or frame < native.control.frame or frame > final_frame:
                raise ValueError("Replay commands are not in native-frame order")
            while native.control.frame < frame:
                native.control.tick()
            if event["op"] == "write":
                commands = event["commands"]
                if not isinstance(commands, dict) or not commands:
                    raise ValueError("Empty native write group")
                for name, target in commands.items():
                    native.device_write(name, target)
            elif event["op"] == "stop" and set(event) == {"frame", "op", "device"}:
                native.device_stop(event["device"])
            else:
                raise ValueError("Unsupported recorded operation")
        while native.control.frame < final_frame:
            native.control.tick()
        final_error = compare(record["state"], native.owner_state(), atol=TOLERANCES[backend])
        compare(
            record["control_state"],
            {
                "targets": native.control.targets,
                "gripper_holds": getattr(native, "gripper_holds", {}),
            },
            atol=TOLERANCES[backend],
            path="control_state",
        )
        if (
            adapter.budget().as_dict() != budget
            or native.task_completed() != record["native_success"]
        ):
            raise ValueError("Native verdict or budget differs during replay")
        report = {
            "schema": "roboenv/native-replay/1",
            "passed": True,
            "record_sha256": digest(record),
            "backend": backend,
            "tolerance": TOLERANCES[backend],
            "max_initial_difference": initial_error,
            "max_final_difference": final_error,
            "budget": adapter.budget().as_dict(),
            "native_success": native.task_completed(),
            "commands": len(record["commands"]),
        }
        return report
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        # Isaac may exit Python directly from close(), even while unwinding an
        # exception. Persist the measured result first; callers must inspect it.
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x") as stream:
                stream.write(json.dumps(report, indent=2) + "\n")
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Native upstreams may change cwd while loading task assets.
    args.record = args.record.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Preserve existing evidence; select a fresh output path")
    try:
        report = replay(json.loads(args.record.read_text()), out=args.out)
    except Exception as error:
        report = {
            "schema": "roboenv/native-replay/1",
            "passed": False,
            "error": type(error).__name__ + ": " + str(error),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # This path also records validation/creation/cleanup failures outside replay.
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
