"""Offline rendering entry point; no live environment service or agent is started."""

import argparse
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from roboenv.recording import file_digest, inspect_recording


@contextmanager
def native_environment(backend):
    """Validate pinned image sources without writing their Git object stores."""
    from roboenv.provisioning.sources import load_lock, selected_sources

    from roboenv_runtime.paths import repo_root

    root, lock = load_lock(repo_root())
    trusted = [root / lock["sources"][name]["path"]
               for name in selected_sources(lock, [backend])]
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    environment = {"GIT_CONFIG_COUNT": str(count + len(trusted))}
    for index, checkout in enumerate(trusted, start=count):
        environment[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        environment[f"GIT_CONFIG_VALUE_{index}"] = str(checkout)
    depot = Path(os.environ.get("ROBOENV_ASSET_CACHE", root / ".cache/assets"))
    if (backend == "behavior" and "OMNIGIBSON_DATA_PATH" not in os.environ
            and (depot / "behavior/datasets").is_dir()):
        environment["OMNIGIBSON_DATA_PATH"] = str(depot / "behavior/datasets")
    previous = {key: os.environ.get(key) for key in environment}
    previous.update({key: os.environ.get(key) for key in
                     ("GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES")})
    with tempfile.TemporaryDirectory(prefix="roboenv-replay-git-") as temporary:
        try:
            os.environ.update(environment)
            objects = [subprocess.check_output([
                "git", "-C", str(checkout), "rev-parse", "--path-format=absolute",
                "--git-path", "objects",
            ], text=True).strip() for checkout in trusted]
            alternates = os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            if alternates:
                objects.append(alternates)
            os.environ["GIT_OBJECT_DIRECTORY"] = temporary
            os.environ["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = os.pathsep.join(objects)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def render(directory, output, *, height=1080, stride=1, cameras=None,
           allow_command_replay=False):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    status = inspect_recording(directory)
    if not status["complete"]:
        raise ValueError("Cannot render incomplete recording evidence")
    if status["playback_kind"] == "recorded-native-integration-states":
        from roboenv_runtime.mujoco_recording import replay

        report = replay(directory, output, height, stride=stride, cameras=cameras)
    elif status["playback_kind"] == "recorded-native-scene-states":
        from roboenv_runtime.adapters.robodojo.recording import replay

        with native_environment(status["backend"]):
            report = replay(directory, output, height, stride=stride, cameras=cameras)
    else:
        if not allow_command_replay:
            raise ValueError("Command replay must be requested explicitly")
        from roboenv_runtime.adapters.native_replay import replay

        record = json.loads((directory / "native-episode.json").read_text())
        with native_environment(status["backend"]):
            report = replay(record, out=output / "replay.json", video_dir=output,
                            height=height, stride=stride, cameras=cameras,
                            recording_sha256=file_digest(directory / "manifest.json"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--allow-command-replay", action="store_true")
    args = parser.parse_args(argv)
    report = render(args.recording, args.output, height=args.height,
                    stride=args.stride, cameras=args.cameras,
                    allow_command_replay=args.allow_command_replay)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
