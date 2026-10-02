"""Render recorded MuJoCo states without executing commands or native physics."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from roboenv.recording import file_digest, write_json

from roboenv_runtime.adapters.native_video import Video


def replay(directory, output, size=768, *, stride=1, cameras=None):
    import mujoco

    if type(size) is not int or size < 2 or size % 2 or type(stride) is not int or stride < 1:
        raise ValueError("Render size must be positive and even; stride must be positive")
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("Incomplete recording or different MuJoCo version")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_digest(directory / name) != expected:
            raise ValueError(f"Recording integrity failed: {name}")
    states = np.memmap(directory / "states.f64", mode="r", dtype=np.float64)
    if states.size != manifest["states"] * manifest["state_width"]:
        raise ValueError("Truncated state recording")
    states = states.reshape(manifest["states"], manifest["state_width"])
    timeline = [json.loads(line)
                for line in (directory / "timeline.jsonl").read_text().splitlines()]
    if len(timeline) != len(states):
        raise ValueError("State timeline length differs")
    names = list(cameras) if cameras is not None else manifest["cameras"]
    if not names or len(set(names)) != len(names) or set(names) - set(manifest["cameras"]):
        raise ValueError("Select distinct recorded cameras")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    model.vis.global_.offwidth = max(size, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(size, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    spec = mujoco.mjtState(manifest["state_spec"])
    visual = mujoco.MjvOption()
    for name, value in manifest["visual_options"].items():
        if isinstance(value, list):
            getattr(visual, name)[:] = value
        else:
            setattr(visual, name, value)
    fps = manifest["control_hz"] / (manifest.get("stride", 1) * stride)
    videos = []
    frames = []
    difference = 0.0
    restored = np.empty(manifest["state_width"])
    with ExitStack() as cleanup:
        renderer = mujoco.Renderer(model, height=size, width=size)
        cleanup.callback(renderer.close)
        for index in range(len(names)):
            video = Video(output / f"camera_{index}.mp4", size, size, fps)
            cleanup.callback(video.close)
            videos.append(video)
        for index, state in enumerate(states):
            mujoco.mj_setState(model, data, state.copy(), spec)
            mujoco.mj_getState(model, data, restored, spec)
            difference = max(difference, float(np.max(np.abs(restored - state))))
            if not np.isfinite(restored).all() or difference > 1e-12:
                raise ValueError("Restored native state differs from original episode")
            if index % stride and index != len(states) - 1:
                continue
            mujoco.mj_forward(model, data)
            for camera, video in zip(names, videos, strict=True):
                renderer.update_scene(data, camera=camera, scene_option=visual)
                video.append(renderer.render())
            frames.append({**timeline[index], "source_index": index, "index": len(frames)})
    write_json(output / "frames.json", frames)
    camera_info = [{"name": name, "file": f"camera_{i}.mp4", "width": size, "height": size}
                   for i, name in enumerate(names)]
    alignment = {
        "passed": True,
        "method": "Recorded MuJoCo camera definitions and verified integration states",
        "original_episode_image_comparison": False,
        "frames_checked": len(frames), "cameras": camera_info,
    }
    write_json(output / "camera-alignment.json", alignment)
    result = {
        "schema": "roboenv/state-playback/1", "passed": True,
        "kind": "recorded-native-integration-states", "command_resimulation": False,
        "validation_scope": "every recorded native integration state",
        "intermediate_states_verified": True,
        "states_verified": len(states), "max_state_difference": difference,
        "native_render_size": [size, size],
        "recording_sha256": file_digest(directory / "manifest.json"),
        "native_success": manifest["native_success"],
        "media": {
            "cameras": camera_info, "frames": len(frames), "fps": fps,
            "waiting_policy": (
                "Recorded states retained; wall-clock gaps without native steps omitted"
            ),
            "files": {name: file_digest(output / name)
                      for name in ["frames.json", *(c["file"] for c in camera_info)]},
        },
    }
    write_json(output / "replay.json", result)
    return result
