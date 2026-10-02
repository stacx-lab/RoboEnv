"""Recorded MuJoCo state playback; recording is owned by the worker."""

from pathlib import Path

from roboenv_runtime.mujoco_recording import replay

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=768)
    args = parser.parse_args()
    print(json.dumps(replay(args.recording, args.output, args.size)))
