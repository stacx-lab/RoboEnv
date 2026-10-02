"""Record and render native liquid scene states without resimulating commands."""

import io
import json
from pathlib import Path

import numpy as np
from roboenv.recording import file_digest, inspect_recording, write_json

from roboenv_runtime.adapters.native_record import compare
from roboenv_runtime.adapters.native_video import NativeVideo
from roboenv_runtime.core.adapter import make_adapter


def pack(state, layout):
    values = []

    def visit(value, shape):
        if isinstance(shape, dict):
            if not isinstance(value, dict) or set(value) != set(shape):
                raise ValueError("Native scene fields changed during recording")
            for key, child in shape.items():
                visit(value[key], child)
        elif isinstance(shape, list):
            if not isinstance(value, list) or len(value) != len(shape):
                raise ValueError("Native scene dimensions changed during recording")
            for child, template in zip(value, shape, strict=True):
                visit(child, template)
        elif shape is None:
            if value is not None:
                raise ValueError("Native scene optional field changed during recording")
        elif isinstance(shape, (int, float)) and isinstance(value, (int, float)):
            values.append(value)
        else:
            raise ValueError("Unsupported native scene state field")

    visit(state, layout)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite native scene state")
    return result


def unpack(vector, layout):
    values = iter(vector)

    def visit(shape):
        if isinstance(shape, dict):
            return {key: visit(child) for key, child in shape.items()}
        if isinstance(shape, list):
            return [visit(child) for child in shape]
        return None if shape is None else type(shape)(next(values))

    result = visit(layout)
    if next(values, None) is not None:
        raise ValueError("Native scene state width differs")
    return result


class RoboDojoSource:
    def __init__(self, world):
        self.world = world

    @property
    def frame(self):
        return self.world.control.frame

    def prepare(self, directory):
        self.layout = self.world.scene_state()
        write_json(directory / "scene-layout.json", self.layout)
        return {
            "playback_kind": "recorded-native-scene-states", "state_spec": "robodojo-scene/2",
            "state_files": ["states.f64", "fluid-surfaces.bin"],
            "state_width": pack(self.layout, self.layout).size,
            "control_hz": 1 / self.world.control_interval,
            "cameras": [name for name, fact in self.world.device_facts.items()
                        if fact["type"] in {"rgb", "rgbd"}],
        }

    def capture(self):
        surfaces = self.world.fluid_surface_state()
        if any(not np.isfinite(value).all() for value in surfaces.values()):
            raise ValueError("Nonfinite native fluid surface")
        buffer = io.BytesIO()
        np.savez(buffer, **surfaces)
        return {"states.f64": pack(self.world.scene_state(), self.layout).tobytes(),
                "fluid-surfaces.bin": buffer.getvalue()}


def surface_states(directory):
    with (directory / "fluid-surfaces.bin").open("rb") as stream:
        while prefix := stream.read(8):
            length = int.from_bytes(prefix, "little")
            with np.load(io.BytesIO(stream.read(length)), allow_pickle=False) as archive:
                yield {name: archive[name] for name in archive.files}


def compare_surfaces(expected, actual):
    if set(expected) != set(actual):
        raise ValueError("Recorded fluid surface fields differ")
    difference = 0.0
    for key, left in expected.items():
        right = actual[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Recorded fluid surface dimensions differ: {key}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Recorded fluid surface is not finite: {key}")
        error = float(np.max(np.abs(left.astype(np.float64) - right), initial=0))
        tolerance = 0 if np.issubdtype(left.dtype, np.integer) else 1e-4
        if error > tolerance:
            raise ValueError(f"Recorded fluid surface differs: {key}: {error} > {tolerance}")
        difference = max(difference, error)
    return difference


def replay(directory, output, height=1080, *, stride=1, cameras=None):
    directory, output = Path(directory), Path(output)
    status = inspect_recording(directory)
    if not status["complete"] or status["playback_kind"] != "recorded-native-scene-states":
        raise ValueError("Incomplete native scene recording")
    if (type(stride) is not int or stride < 1
            or type(height) is not int or height < 2 or height % 2):
        raise ValueError("Render height must be positive and even; stride must be positive")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["state_spec"] != "robodojo-scene/2":
        raise ValueError("Unsupported RoboDojo scene state")
    layout = json.loads((directory / "scene-layout.json").read_text())
    timeline = [json.loads(line)
                for line in (directory / "timeline.jsonl").read_text().splitlines()]
    states = np.memmap(directory / "states.f64", mode="r", dtype=np.float64).reshape(
        manifest["states"], manifest["state_width"])
    names = list(cameras) if cameras is not None else manifest["cameras"][:2]
    if not names or len(set(names)) != len(names) or set(names) - set(manifest["cameras"]):
        raise ValueError("Select distinct recorded cameras")
    record = json.loads((directory / "native-episode.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "roboenv/state-playback/1", "passed": False,
              "recording_sha256": file_digest(directory / "manifest.json")}
    adapter = make_adapter("robodojo", **record["environment"])
    video = None
    try:
        adapter.configure_execution_mode(record["execution_mode"])
        adapter.reset(record["initialization"]["requested_seed"])
        native = adapter._env
        if native.implementation_identity != record["implementation"]:
            raise ValueError("Scene playback implementation, dependencies or assets differ")
        if adapter.embodiment != record["embodiment"]:
            raise ValueError("Scene playback embodiment differs")
        volatile = {"physics_steps", "wall_time_s", "state_sha256"}
        compare({k: v for k, v in record["initialization"].items() if k not in volatile},
                {k: v for k, v in native.initialization.items() if k not in volatile},
                atol=0, path="initialization")
        compare(manifest["control_hz"], 1 / native.control_interval,
                atol=1e-10, path="recorded_control_hz")
        physics_steps = native.physics_steps
        native.prepare_fluid_surface_playback()
        video = NativeVideo(native, "robodojo", output, height=height,
                            stride=manifest["stride"] * stride, cameras=names)
        difference = 0.0
        pairs = zip(states, surface_states(directory), strict=True)
        for index, (vector, surface) in enumerate(pairs):
            state = unpack(vector, layout)
            native.restore_scene_state(state)
            native.restore_fluid_surfaces(surface)
            native.control.frame = timeline[index]["frame"]
            if index % stride == 0 or index == len(states) - 1:
                video.capture(force=True)
            difference = max(difference, compare(state, native.scene_state(), atol=1e-4,
                                                 path=f"recorded_scene[{index}]"))
            difference = max(difference, compare_surfaces(surface, native.fluid_surface_state()))
            if native.physics_steps != physics_steps:
                raise ValueError("Scene rendering advanced native physics")
        video.close()
        media = video.receipt()
        media.update(kind="recorded-native-scene-states", waiting_policy=(
            "Recorded scene frames retained; wall-clock gaps without native steps omitted"))
        write_json(output / "camera-alignment.json", {
            "passed": True, "method": "Recorded scene state and measured native camera projection",
            "original_episode_image_comparison": False, "frames_checked": video.count,
            "cameras": video.cameras,
        })
        report.update(passed=True, kind="recorded-native-scene-states", command_resimulation=False,
                      validation_scope="every recorded native scene state; sealed owner verdict",
                      intermediate_states_verified=True, states_verified=len(states),
                      fluid_surface_states_verified=True,
                      fluid_surface_rendering="recorded-meshes-in-session-layer",
                      max_state_difference=difference, native_steps_advanced=0,
                      native_success=manifest["native_success"],
                      native_verdict_source="sealed-owner-evidence",
                      media=media)
        return report
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if video is not None:
            try:
                video.close()
            except Exception as error:
                report.update(passed=False, error=f"Media cleanup: {type(error).__name__}: {error}")
        write_json(output / "replay.json", report)
        adapter.close()
