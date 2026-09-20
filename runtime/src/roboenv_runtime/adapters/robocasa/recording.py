"""Private owner recording and native state playback, independent of the agent."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


class Video:
    def __init__(self, path, size, fps):
        self.log = Path(path).with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{size[1]}x{size[0]}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stderr=self.log)

    def append(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=120)
        self.log.close()
        if code:
            raise RuntimeError(f"Video encoder exited {code}")


class OwnerRecording:
    """Record every control-interval state and two native camera streams.

    Files stay in the worker's private evidence tree. Playback restores recorded
    integration states; it is state playback, not a claim of command resimulation.
    """

    def __init__(self, world, directory):
        import mujoco

        self.world = world
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model = world.sim.model._model
        self.data = world.sim.data._data
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.zeros(mujoco.mj_stateSize(self.model, self.spec))
        self.started = time.monotonic()
        self.count = 0
        self.closed = False
        self.last_video_frame = None
        self.frames = []
        context = getattr(world.sim, "_render_context_offscreen", None)
        visual = context.vopt if context is not None else mujoco.MjvOption()
        visual_fields = ("geomgroup", "sitegroup", "jointgroup", "tendongroup", "actuatorgroup",
                         "flexgroup", "skingroup", "flags", "label", "frame", "bvh_depth", "flex_layer")
        visual_options = {name: np.asarray(getattr(visual, name)).tolist()
                          for name in visual_fields if hasattr(visual, name)}
        self.cameras = list(dict.fromkeys((world._camera_names[0], world._camera_names[-1])))
        self.control_hz = float(world.env.control_freq)
        self.stride = max(1, round(self.control_hz / 10))
        mujoco.mj_saveModel(self.model, str(self.directory / "model.mjb"), None)
        (self.directory / "model.xml").write_text(world.sim.model.get_xml())
        self.states = (self.directory / "states.f64").open("wb")
        self.timeline = (self.directory / "timeline.jsonl").open("w")
        self.commands = (self.directory / "commands.jsonl").open("w")
        self.videos = [Video(self.directory / f"camera_{i}.mp4", world._camera_size,
                             self.control_hz / self.stride) for i in range(len(self.cameras))]
        self.manifest = {
            "schema": "roboenv/robocasa-owner-recording/1", "complete": False,
            "mujoco_version": mujoco.__version__, "task": world.task_name,
            "seed": world.seed, "episode": world.episode_meta(),
            "cameras": self.cameras, "camera_size": list(world._camera_size),
            "control_hz": self.control_hz, "video_hz": self.control_hz / self.stride,
            "state_spec": int(self.spec), "state_width": self.state.size,
            "state_dtype": "float64", "byteorder": __import__("sys").byteorder,
            "playback_kind": "recorded-native-integration-states",
            "visual_options": visual_options,
            "model_sha256": sha256(self.directory / "model.mjb"),
        }
        write_json(self.directory / "manifest.json", self.manifest)
        self.capture(force_video=True)

    def command(self, operation, value):
        self.commands.write(json.dumps({"frame": self.world._step_count,
            "wall_seconds": time.monotonic() - self.started, "operation": operation,
            "value": value}) + "\n")
        self.commands.flush()

    def capture(self, force_video=False):
        import mujoco

        if self.closed:
            return
        mujoco.mj_getState(self.model, self.data, self.state, self.spec)
        self.states.write(self.state.tobytes())
        frame = self.world._step_count
        row = {"index": self.count, "frame": frame, "simulation_seconds": float(self.data.time),
               "wall_seconds": time.monotonic() - self.started}
        self.timeline.write(json.dumps(row) + "\n")
        self.count += 1
        if (force_video or frame % self.stride == 0) and frame != self.last_video_frame:
            for camera, video in zip(self.cameras, self.videos, strict=True):
                video.append(self.world.render(camera))
            self.frames.append(row)
            self.last_video_frame = frame

    def close(self):
        if self.closed:
            return
        self.capture(force_video=True)
        self.closed = True
        for stream in (self.states, self.timeline, self.commands):
            stream.close()
        for video in self.videos:
            video.close()
        write_json(self.directory / "video-frames.json", self.frames)
        self.manifest.update(complete=True, states=self.count, video_frames=len(self.frames),
            final_frame=self.world._step_count, native_success=bool(self.world.task_completed()),
            states_sha256=sha256(self.directory / "states.f64"),
            files={p.name: sha256(p) for p in sorted(self.directory.iterdir())
                   if p.is_file() and p.name != "manifest.json"})
        write_json(self.directory / "manifest.json", self.manifest)


def replay(directory, output, size=768):
    """Restore and check every recorded state in a fresh native renderer."""
    import mujoco

    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("Incomplete recording or different MuJoCo version")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Recording integrity failed: {name}")
    states = np.fromfile(directory / "states.f64", dtype=np.float64)
    if states.size != manifest["states"] * manifest["state_width"]:
        raise ValueError("Truncated state recording")
    states = states.reshape(manifest["states"], manifest["state_width"])
    timeline = [json.loads(line) for line in (directory / "timeline.jsonl").read_text().splitlines()]
    if len(timeline) != len(states):
        raise ValueError("State timeline length differs")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    model.vis.global_.offwidth = max(size, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(size, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    spec = mujoco.mjtState(manifest["state_spec"])
    renderer = mujoco.Renderer(model, height=size, width=size)
    visual = mujoco.MjvOption()
    for name, value in manifest["visual_options"].items():
        if isinstance(value, list):
            getattr(visual, name)[:] = value
        else:
            setattr(visual, name, value)
    videos = [Video(output / f"camera_{i}.mp4", (size, size), manifest["control_hz"])
              for i in range(len(manifest["cameras"]))]
    difference = 0.0
    restored = np.zeros(manifest["state_width"])
    try:
        for state in states:
            mujoco.mj_setState(model, data, state, spec)
            mujoco.mj_getState(model, data, restored, spec)
            difference = max(difference, float(np.max(np.abs(restored - state))))
            if not np.isfinite(restored).all() or difference > 1e-12:
                raise ValueError("Restored native state differs from original episode")
            mujoco.mj_forward(model, data)
            for camera, video in zip(manifest["cameras"], videos, strict=True):
                renderer.update_scene(data, camera=camera, scene_option=visual)
                video.append(renderer.render())
    finally:
        for video in videos:
            video.close()
        renderer.close()
    write_json(output / "frames.json", timeline)
    result = {"schema": "roboenv/robocasa-state-playback/1", "passed": True,
              "kind": manifest["playback_kind"], "command_resimulation": False,
              "states_verified": len(states), "max_state_difference": difference,
              "native_render_size": [size, size], "recording_sha256": sha256(directory / "manifest.json"),
              "native_success": manifest["native_success"]}
    write_json(output / "replay.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=768)
    args = parser.parse_args()
    print(json.dumps(replay(args.recording, args.output, args.size)))
