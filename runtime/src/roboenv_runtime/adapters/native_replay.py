"""Replay source-native device commands in a fresh simulator process, owner only."""

import argparse
import json
from pathlib import Path

from roboenv_runtime.adapters.native_record import compare, digest
from roboenv_runtime.core.adapter import make_adapter

TOLERANCES = {"robotwin": 1e-5, "robodojo": 1e-4, "behavior": 1e-4}


def replay(record, *, out=None, video_dir=None, height=1080, stride=1, cameras=None,
           recording_sha256=None):
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
    out = Path(out).resolve() if out is not None else None
    if out is not None and out.exists():
        raise ValueError("Preserve existing replay evidence; choose a fresh output")
    if video_dir is not None:
        if type(height) is not int or height < 2 or height % 2:
            raise ValueError("Render height must be a positive even integer")
        if type(stride) is not int or stride < 1:
            raise ValueError("Render stride must be a positive integer")
        video_dir = Path(video_dir).resolve()
        video_dir.mkdir(parents=True, exist_ok=False)
    video = None
    adapter = make_adapter(backend, **record["environment"])
    report = {"schema": "roboenv/native-replay/1", "passed": False}
    if recording_sha256 is not None:
        report["recording_sha256"] = recording_sha256
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
        if video_dir is not None and backend == "robodojo":
            restored = native.restore_replay_initial_state(record["initial_state"])
            if restored is not None:
                report["initial_state_restoration"] = restored
        initial_error = compare(
            record["initial_state"],
            native.initial_state,
            atol=TOLERANCES[backend],
            path="initial_state",
        )
        if video_dir is not None:
            from .native_video import NativeVideo
            video = NativeVideo(native, backend, video_dir, height=height, stride=stride, cameras=cameras)
            video.capture(force=True)
            compare(record["initial_state"], native.owner_state(), atol=TOLERANCES[backend],
                    path="state_after_renderer_setup")
            native.control.on_step = video.capture
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
        if video is not None:
            video.capture(force=True)
            video.close()
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
        report.update({
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
        })
        if video is not None:
            report.update(kind="native-command-replay", command_resimulation=True,
                          validation_scope="initial/final state, control targets, budget and verdict",
                          intermediate_states_verified=False, media=video.receipt())
            alignment = {"schema": "roboenv/native-camera-alignment/1", "passed": True,
                         "method": "Native camera pose and measured scaled render projection",
                         "original_episode_image_comparison": False,
                         "cameras": video.cameras, "frames_checked": video.count}
            (video_dir / "camera-alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
        return report
    except Exception as error:
        report.update(passed=False, error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if video is not None:
            native.control.on_step = None
            try:
                video.close()
            except Exception as error:
                report.update(passed=False, error=f"Media cleanup: {type(error).__name__}: {error}")
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
    parser.add_argument("--video-dir", type=Path, help="Fresh directory on the selected artifact storage")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--stride", type=int, default=1, help="Render every N native control frames")
    parser.add_argument("--cameras", nargs="+", help="Native device camera names; default first two")
    args = parser.parse_args()
    # Native upstreams may change cwd while loading task assets.
    args.record = args.record.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Preserve existing evidence; select a fresh output path")
    try:
        report = replay(json.loads(args.record.read_text()), out=args.out,
                        video_dir=args.video_dir, height=args.height, stride=args.stride, cameras=args.cameras)
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
