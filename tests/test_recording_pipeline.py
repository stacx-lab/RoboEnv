"""Recording failures stay outside control; sealed evidence round-trips on CPU."""

import json
import os
import signal
import struct

import pytest
from roboenv.recording import export_recording, inspect_recording, write_json
from roboenv_runtime.recording import Recording


class StateSource:
    def __init__(self):
        self.frame = 0

    def prepare(self, directory):
        (directory / "model.mjb").write_bytes(b"fixture model")
        (directory / "model.xml").write_text("fixture model")
        return {"state_width": 1}

    def capture(self):
        return struct.pack("d", self.frame)


def finish(recording, *, frame=4, success=False, native_record=None):
    identity = recording.manifest
    write_json(recording.directory.parent / "terminal.json", {
        "seed": identity["seed"], "task": identity["task"],
        "status": {"budget": {"used": frame}, "session": "fixture", "generation": 0},
        "verdict": {"completed": success},
    })
    if recording.manifest["playback_kind"] != "recorded-native-integration-states":
        write_json(recording.directory.parent / "native-episode.json", {
            **(native_record or {}),
            "backend": identity["backend"], "task": identity["task"], "seed": identity["seed"],
            "native_success": success, "budget": {"used": frame},
            "commands": [{"frame": 0, "op": "write", "commands": {"arm": [0.5]}}],
        })
    recording.seal(frame, frame / 20, success)
    recording.finish()


def create(tmp_path, backend="robotwin", mode="stepped", **kwargs):
    source = StateSource() if backend in {"libero", "robocasa"} else None
    return Recording(tmp_path / "recording", {
        "backend": backend, "task": "fixture", "seed": 0, "session": "fixture",
        "generation": 0, "execution_mode": mode, "embodiment": "fixture",
    }, source=source, **kwargs)


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_state_and_command_recordings_export_and_detect_tampering(tmp_path, backend, mode):
    recording = create(tmp_path, backend, mode)
    for frame in range(5):
        if recording.source is not None:
            recording.source.frame = frame
        recording.capture(frame, frame / 20)
    # Sealing an already captured final state does not duplicate a frame.
    finish(recording)
    status = inspect_recording(recording.directory)
    assert status["complete"] is True
    assert status["frames"] == 5 and status["dropped_events"] == 0
    assert status["video_available"] is False
    assert status["intermediate_states_recorded"] is (recording.source is not None)
    exported = tmp_path / "export"
    assert export_recording(recording.directory, exported) == status
    assert inspect_recording(exported) == status
    assert json.loads((exported / "manifest.json").read_text())["execution_mode"] == mode
    if recording.source is not None:
        assert struct.unpack("5d", (exported / "states.f64").read_bytes()) == (0, 1, 2, 3, 4)
    with pytest.raises(FileExistsError):
        export_recording(recording.directory, exported)
    with (exported / "timeline.jsonl").open("a") as stream:
        stream.write("tampered\n")
    assert inspect_recording(exported)["complete"] is False


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_stalled_writer_drops_events_without_waiting(tmp_path, mode):
    recording = create(tmp_path, mode=mode, buffer_frames=1)
    os.kill(recording.writer.pid, signal.SIGSTOP)
    try:
        for frame in range(100):
            recording.capture(frame, frame / 20)
        assert recording.counters[1] > 0
        assert recording.writer.is_alive()
    finally:
        os.kill(recording.writer.pid, signal.SIGCONT)
        finish(recording, frame=99)
    status = inspect_recording(recording.directory)
    assert status["complete"] is False and status["dropped_events"] > 0


def test_dead_writer_is_reported_without_raising_into_capture(tmp_path):
    recording = create(tmp_path)
    recording.writer.kill()
    recording.writer.join()
    recording.capture(0, 0)
    finish(recording)
    status = inspect_recording(recording.directory)
    assert status["complete"] is False
    assert status["error"] == "recording writer did not finish"


def test_state_capture_failure_invalidates_only_the_recording(tmp_path, monkeypatch):
    recording = create(tmp_path, "libero")

    def fail():
        raise RuntimeError("fixture capture failed")

    monkeypatch.setattr(recording.source, "capture", fail)
    recording.capture(0, 0)
    finish(recording, success=True)
    status = inspect_recording(recording.directory)
    assert status["complete"] is False and status["capture_errors"] > 0
    terminal = json.loads((recording.directory / "terminal.json").read_text())
    assert terminal["verdict"]["completed"] is True


@pytest.mark.parametrize("failure", [None, "state", "surface", "physics", "implementation"])
def test_liquid_scene_playback_uses_recorded_states_and_sealed_verdict(
    tmp_path, monkeypatch, failure,
):
    import copy
    import shutil
    from types import SimpleNamespace

    import numpy as np
    from roboenv_runtime.adapters import native_video
    from roboenv_runtime.adapters.robodojo import recording as scene

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real video encoder")
    state = {"pose": [0.0], "particles": [[[0.0, 0.0, 0.0]], None, None]}
    surfaces = {"bottle/points": np.zeros((1, 3), dtype=np.float32)}
    native = SimpleNamespace(
        scene_state=lambda: copy.deepcopy(state), control=SimpleNamespace(frame=0),
        fluid_surface_state=lambda: copy.deepcopy(surfaces),
        prepare_fluid_surface_playback=lambda: None,
        physics_steps=0, control_interval=0.04, prepare_replay_materials=lambda: [],
        initialization={"requested_seed": 0},
        implementation_identity={
            "version": "changed" if failure == "implementation" else "fixture"},
        task_completed=lambda: False,
        device_facts={"head": {"type": "rgb", "intrinsics": {"color": {
            "width": 4, "height": 4, "projection": {"fx": 4, "fy": 4, "cx": 2, "cy": 2},
        }}}},
    )
    source = scene.RoboDojoSource(native)
    recording = Recording(tmp_path / "recording", {
        "backend": "robodojo", "task": "fixture", "seed": 0, "session": "fixture",
        "generation": 0, "execution_mode": "stepped", "embodiment": "fixture",
    }, source=source)
    for frame in range(4):
        state["pose"][0] = frame * 0.25
        state["particles"][0][0][0] = frame * 0.1
        # Native isosurface vertex counts change while particle counts stay fixed.
        surfaces["bottle/points"] = np.full((frame + 1, 3), frame, dtype=np.float32)
        native.control.frame = frame
        recording.capture(frame, frame * 0.04)
    finish(recording, frame=3, success=True, native_record={
        "environment": {}, "initialization": {"requested_seed": 0}, "execution_mode": "stepped",
        "implementation": {"version": "fixture"}, "embodiment": "fixture",
    })
    status = inspect_recording(recording.directory)
    assert status["complete"] and status["intermediate_states_recorded"]
    assert status["states"] == 4 and status["playback_kind"] == "recorded-native-scene-states"
    adapter = SimpleNamespace(_env=native, embodiment="fixture", reset=lambda _: None,
                              configure_execution_mode=lambda _: None, close=lambda: None)
    monkeypatch.setattr(scene, "make_adapter", lambda *args, **kwargs: adapter)

    def restore(snapshot):
        state.clear()
        state.update(copy.deepcopy(snapshot))
        if failure == "state":
            state["pose"][0] += 0.01
        if failure == "physics":
            native.physics_steps += 1

    native.restore_scene_state = restore

    def restore_surfaces(snapshot):
        surfaces.clear()
        surfaces.update(copy.deepcopy(snapshot))
        if failure == "surface":
            surfaces["bottle/points"][0, 0] += 0.01

    native.restore_fluid_surfaces = restore_surfaces
    rendered = []

    class Camera:
        def __init__(self, world, *args):
            self.width, self.height = args[-3:-1]
            self.render_subframes = 4

        def read(self):
            rendered.append((native.control.frame, state["pose"][0],
                             len(surfaces["bottle/points"])))
            return np.full((self.height, self.width, 3), int(state["pose"][0] * 200), np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(native_video, "IsaacCamera", Camera)
    out = tmp_path / "replay"
    if failure:
        with pytest.raises(ValueError, match="differ|advanced native physics"):
            scene.replay(recording.directory, out, height=8)
        assert json.loads((out / "replay.json").read_text())["passed"] is False
        assert not (out / "camera-alignment.json").exists()
    else:
        report = scene.replay(recording.directory, out, height=8)
        assert report["passed"] and report["states_verified"] == 4
        assert report["fluid_surface_states_verified"] is True
        assert report["native_success"] is True and native.task_completed() is False
        assert report["native_verdict_source"] == "sealed-owner-evidence"
        assert report["command_resimulation"] is False and report["native_steps_advanced"] == 0
        assert report["max_state_difference"] == 0 and report["media"]["fps"] == 25
        assert rendered == [(0, 0.0, 1), (1, 0.25, 2), (2, 0.5, 3), (3, 0.75, 4)]
        assert inspect_recording(recording.directory) == status
        from roboenv.recording import file_digest

        # Updating a checksum cannot disguise a truncated surface frame stream.
        surface_file = recording.directory / "fluid-surfaces.bin"
        surface_file.write_bytes(surface_file.read_bytes()[:-1])
        manifest = json.loads((recording.directory / "manifest.json").read_text())
        manifest["files"][surface_file.name] = file_digest(surface_file)
        write_json(recording.directory / "manifest.json", manifest)
        assert inspect_recording(recording.directory)["complete"] is False


def test_liquid_capture_layout_changes_invalidate_media_without_changing_success(tmp_path):
    import copy
    from types import SimpleNamespace

    from roboenv_runtime.adapters.robodojo.recording import RoboDojoSource

    state = {"particles": [[0.0, 0.0, 0.0]]}
    native = SimpleNamespace(
        scene_state=lambda: copy.deepcopy(state), control=SimpleNamespace(frame=0),
        fluid_surface_state=lambda: {},
        control_interval=0.04, device_facts={})
    recording = Recording(tmp_path / "recording", {
        "backend": "robodojo", "task": "fixture", "seed": 0, "session": "fixture",
        "generation": 0, "execution_mode": "stepped", "embodiment": "fixture",
    }, source=RoboDojoSource(native))
    recording.capture(0, 0)
    state["particles"].append([1.0, 2.0, 3.0])
    recording.capture(1, 0.04)
    finish(recording, frame=1, success=True)
    assert inspect_recording(recording.directory)["complete"] is False
    assert recording.counters[5] > 0
    assert json.loads((recording.directory / "terminal.json").read_text())["verdict"]["completed"]


def test_command_replay_requires_explicit_opt_in_before_subprocess(tmp_path):
    from roboenv.recording import render_recording

    recording = create(tmp_path)
    recording.capture(0, 0)
    finish(recording, frame=0)
    with pytest.raises(ValueError, match="explicitly"):
        render_recording(recording.directory, tmp_path / "render", image="unused")
    assert not (tmp_path / "render").exists()


@pytest.mark.parametrize("field,value", [
    ("sealed", False), ("dropped_events", 1), ("capture_errors", 1),
])
def test_inspection_rejects_overclaimed_completeness(tmp_path, field, value):
    recording = create(tmp_path)
    recording.capture(0, 0)
    finish(recording, frame=0)
    path = recording.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    assert inspect_recording(recording.directory)["complete"] is False


@pytest.mark.parametrize("source_fps,speed", [(25, 1), (25, 4), (12.5, 1)])
def test_edit_decodes_real_media_and_preserves_frame_provenance(tmp_path, source_fps, speed):
    import shutil

    import numpy as np
    from roboenv.media import edit
    from roboenv.recording import file_digest
    from roboenv_runtime.adapters.native_video import Video

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real video editor")
    replay = tmp_path / "replay"
    replay.mkdir()
    cameras = [{"file": f"camera_{index}.mp4"} for index in range(2)]
    for camera in cameras:
        video = Video(replay / camera["file"], 16, 16, source_fps)
        try:
            for frame in range(6):
                video.append(np.full((16, 16, 3), frame * 40, dtype=np.uint8))
        finally:
            video.close()
    frames = [{"index": index, "frame": index * 2, "simulation_seconds": index / source_fps}
              for index in range(6)]
    write_json(replay / "frames.json", frames)
    files = ["frames.json", *(camera["file"] for camera in cameras)]
    write_json(replay / "replay.json", {
        "passed": True, "kind": "recorded-native-integration-states",
        "validation_scope": "every recorded native integration state",
        "media": {"cameras": cameras, "waiting_policy": "fixture",
                  "files": {name: file_digest(replay / name) for name in files}},
    })
    output = tmp_path / "edited.mp4"
    receipt = edit(replay, output, "Fixture: literal 100% [title]", speed=speed)
    assert receipt["streams"][0]["width"] == 32
    assert receipt["streams"][0]["height"] == 144
    assert float(receipt["format"]["duration"]) >= 3
    mapping = json.loads(output.with_suffix(".frames.json").read_text())
    assert [row["frame"] for row in mapping["frames"]] == [row["frame"] for row in frames]
    assert mapping["frames"][-1]["edited_seconds"] == pytest.approx(1 + 5 / (source_fps * speed))
    assert mapping["fps"] == source_fps * speed
    assert mapping["source_cadence_preserved"] is True
    from fractions import Fraction

    assert float(Fraction(receipt["streams"][0]["r_frame_rate"])) == source_fps * speed
    import subprocess

    pixels = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(output), "-vf", "crop=16:16:0:128",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    colors = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 16, 16).mean(axis=(1, 2))
    # Each original state must remain visible, including when explicitly accelerated.
    assert all(np.any(np.abs(colors - frame * 40) < 4) for frame in range(6))
    (replay / "camera_0.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        edit(replay, tmp_path / "other.mp4", "Invalid source")
    assert not (tmp_path / "other.mp4").exists()


@pytest.mark.parametrize("graceful", [True, False])
def test_supervisor_gives_only_normal_shutdown_time_to_seal_writer(
    tmp_path, monkeypatch, graceful
):
    import time
    from pathlib import Path

    from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([
        str(Path(__file__).parent), *os.environ.get("PYTHONPATH", "").split(os.pathsep),
    ]))
    supervisor = Supervisor(SupervisorConfig(
        run_dir=tmp_path, task="fixture", seed=0, session="fixture", budget_mode="practice",
        worker_module="test_recording_pipeline", terminate_grace_s=0.2,
    ))
    try:
        supervisor.start_worker()
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert time.monotonic() < deadline
            assert supervisor._worker.poll() is None
            time.sleep(0.02)
        supervisor.terminate_worker(reason="fixture shutdown", graceful=graceful)
        assert supervisor._worker.poll() == 0
        assert inspect_recording(tmp_path / "recording")["complete"] is graceful
    finally:
        supervisor.close()


def recording_worker():
    """CPU owner process used to exercise the real supervisor's signal boundary."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    recording = create(args.run_dir, "libero")
    recording.capture(0, 0)
    write_json(args.run_dir / "terminal.json", {
        "seed": 0, "task": "fixture", "status": {
            "budget": {"used": 0}, "session": "fixture", "generation": 0,
        }, "verdict": {"completed": False},
    })
    recording.seal(0, 0, False)

    def shutdown(signum, frame):
        recording.finish()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    (args.run_dir / "ready").touch()
    while True:
        signal.pause()


if __name__ == "__main__":
    recording_worker()
