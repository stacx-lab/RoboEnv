"""Portable owner recording inspection, export and deferred rendering."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA = "roboenv/recording/1"
FINALIZE_TIMEOUT_S = 20


def configuration(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("Recording configuration must be a dictionary")
    config = {"buffer_frames": 256, "stride": 1, **(value or {})}
    if set(config) != {"buffer_frames", "stride"}:
        raise ValueError("Recording accepts buffer_frames and stride")
    if any(type(item) is not int or item < 1 for item in config.values()):
        raise ValueError("Recording buffer_frames and stride must be positive integers")
    return config


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def inspect_recording(directory):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        return {"complete": False, "reason": "owner recording manifest missing"}
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        return {"complete": False, "reason": "Unsupported recording schema"}
    required = {"commands.jsonl", "timeline.jsonl", "terminal.json"}
    if manifest["playback_kind"] == "recorded-native-integration-states":
        required |= {"model.mjb", "model.xml", "states.f64"}
    elif (manifest["playback_kind"] == "recorded-native-scene-states"
          and manifest["backend"] == "robodojo"):
        required |= {"scene-layout.json", "states.f64", "native-episode.json", "fluid-surfaces.bin"}
    elif manifest["playback_kind"] == "native-command-replay":
        required.add("native-episode.json")
    else:
        raise ValueError("Unsupported recording playback kind")
    files = manifest.get("files", {})
    invalid = sorted(required - files.keys())
    for name, expected in files.items():
        file = directory / name
        if (Path(name).name != name or file.is_symlink() or not file.is_file()
                or file_digest(file) != expected):
            invalid.append(name)
    frames = manifest.get("frames", 0)
    complete = (
        manifest.get("complete") is True and manifest.get("sealed") is True
        and manifest.get("dropped_events", 0) == 0 and manifest.get("capture_errors", 0) == 0
        and not invalid and frames > 0
    )
    if complete:
        timeline = [json.loads(line)
                    for line in (directory / "timeline.jsonl").read_text().splitlines()]
        complete = len(timeline) == frames and all(
            row["index"] == index and (index == 0 or row["frame"] > timeline[index - 1]["frame"])
            for index, row in enumerate(timeline)
        )
        complete = complete and timeline[-1]["frame"] == manifest["final_frame"]
        terminal = json.loads((directory / "terminal.json").read_text())
        complete = complete and (
            terminal["status"]["budget"]["used"] == manifest["final_frame"]
            and terminal["verdict"]["completed"] is manifest["native_success"]
            and terminal["seed"] == manifest["seed"]
            and terminal["task"] == manifest["task"]
            and terminal["status"]["session"] == manifest["session"]
            and terminal["status"]["generation"] == manifest["generation"]
        )
        if manifest["playback_kind"] != "native-command-replay":
            complete = complete and (
                manifest.get("states") == frames
                and (directory / "states.f64").stat().st_size
                == frames * manifest["state_width"] * 8
            )
        if manifest["playback_kind"] == "recorded-native-scene-states":
            complete = complete and manifest.get("state_spec") == "robodojo-scene/2"
            # Surface topology varies per frame, so these are length-prefixed
            # native arrays rather than a fixed-width integration vector.
            size = (directory / "fluid-surfaces.bin").stat().st_size
            with (directory / "fluid-surfaces.bin").open("rb") as stream:
                for _ in range(frames):
                    prefix = stream.read(8)
                    length = int.from_bytes(prefix, "little")
                    if len(prefix) != 8 or not 0 < length <= size - stream.tell():
                        complete = False
                        break
                    stream.seek(length, 1)
                complete = complete and stream.tell() == size
        if manifest["playback_kind"] != "recorded-native-integration-states":
            native = json.loads((directory / "native-episode.json").read_text())
            complete = complete and (
                native["native_success"] is manifest["native_success"]
                and native["budget"]["used"] == manifest["final_frame"]
                and native["backend"] == manifest["backend"]
                and native["seed"] == manifest["seed"]
                and native["task"] == manifest["task"]
                and [json.loads(line) for line in
                     (directory / "commands.jsonl").read_text().splitlines()] == native["commands"]
            )
    return {
        "format": SCHEMA,
        "complete": complete,
        "invalid_files": invalid,
        "backend": manifest["backend"],
        "playback_kind": manifest["playback_kind"],
        "intermediate_states_recorded": manifest["intermediate_states_recorded"],
        "frames": frames,
        "states": manifest.get("states", 0),
        "dropped_events": manifest.get("dropped_events", 0),
        "capture_errors": manifest.get("capture_errors", 0),
        "error": manifest.get("error"),
        "video_available": False,
        "replay_supported": complete,
    }


def export_recording(directory, output):
    status = inspect_recording(directory)
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in {"manifest.json", *manifest.get("files", {})}:
        if Path(name).name != name or (directory / name).is_symlink():
            raise ValueError("Recording files must be regular files inside the recording")
        shutil.copy2(directory / name, output / name)
    return status


def render_recording(directory, output, *, image=None, interpreter=None, gpu="0",
                     height=1080, stride=1, cameras=None, allow_command_replay=False,
                     timeout=28800):
    """Run only when the caller has released or isolated control resources.

    image selects the prepared simulator image. Without it, interpreter must
    belong to the matching prepared simulator environment. No live RPC is used.
    """
    status = inspect_recording(directory)
    if not status["complete"]:
        raise ValueError("Cannot render incomplete recording evidence")
    if status["playback_kind"] == "native-command-replay" and not allow_command_replay:
        raise ValueError("Command replay must be requested explicitly")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    options = ["--height", str(height), "--stride", str(stride)]
    if cameras:
        options += ["--cameras", *cameras]
    if allow_command_replay:
        options.append("--allow-command-replay")
    if image is not None and interpreter is not None:
        raise ValueError("Choose a simulator image or interpreter")
    if (output / "replay").exists():
        raise ValueError("Preserve existing replay; choose a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    if image is None:
        command = [str(interpreter or sys.executable), "-m", "roboenv_runtime.recording_render",
                   str(directory), str(output / "replay"), *options]
    else:
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True,
        ).strip()
        for name in ("home", "tmp", "cache"):
            (output / "runtime" / name).mkdir(parents=True)
        command = ["docker", "run", "--rm", "--pull=never", "--network=none",
                   "--user", "0:0" if status["backend"] in {"robodojo", "behavior"}
                   else f"{os.getuid()}:{os.getgid()}",
                   "--shm-size", "8g", "--gpus", f"device={gpu}",
                   "-e", "HOME=/output/runtime/home", "-e", "TMPDIR=/output/runtime/tmp",
                   "-e", "XDG_CACHE_HOME=/output/runtime/cache",
                   "-v", f"{directory}:/recording:ro", "-v", f"{output}:/output",
                   "--entrypoint", "/opt/sim-env/bin/python", image_id,
                   "-m", "roboenv_runtime.recording_render",
                   "/recording", "/output/replay", *options]
        write_json(output / "runtime.json", {"image_id": image_id, "requested_image": image})
    subprocess.run(command, check=True, timeout=timeout)
    report = json.loads((output / "replay/replay.json").read_text())
    if report.get("passed") is not True:
        raise ValueError(f"Offline replay failed: {report.get('error')}")
    return report
