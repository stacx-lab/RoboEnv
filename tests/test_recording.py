"""Real encoder timing/failure checks with small synthetic spectator images."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.recording import recording_config
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.worker import SimWorker
from roboenv_runtime.spectator import ViewSpec
from roboenv_runtime.spectator.recording import Recorder


class Renderer:
    views = (ViewSpec("overview", source="third_person", width=16, height=16),)

    def __init__(self):
        self.color = [255, 0, 0]
        self.closed = False

    def capture(self):
        return {"overview": np.full((16, 16, 3), self.color, dtype=np.uint8)}

    def close(self):
        self.closed = True


@pytest.fixture
def renderer(monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("optional FFmpeg unavailable")
    renderer = Renderer()
    monkeypatch.setattr("roboenv_runtime.spectator.recording.create_spectator_renderer",
                        lambda *_: renderer)
    return renderer


def manifest(path):
    return json.loads((path / "manifest.json").read_text())


def decode(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path / "overview.mp4"),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"],
        capture_output=True, check=True, timeout=10,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 16, 16, 3)


@pytest.mark.parametrize("value", [True, [], {"other": 1}, {"fps": False}, {"fps": 0},
                                   {"fps": float("nan")}, {"views": []}, {"views": [1]}])
def test_invalid_recording_configuration(value):
    with pytest.raises(ValueError):
        recording_config(value)


def test_defaults_and_missing_encoder_fail_before_renderer(monkeypatch, tmp_path):
    assert recording_config({}) == {"fps": 10, "views": None}
    assert recording_config(None) is None
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="FFmpeg"):
        Recorder(None, tmp_path / "video", {})
    assert not (tmp_path / "video").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_custom_directory_receives_video_and_metadata(tmp_path, renderer, monkeypatch, existing):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "chosen" / "episode"
    if existing:
        output.mkdir(parents=True)
    config = recording_config({"directory": "chosen/episode"})
    assert config["directory"] == str(output)
    worker = SimWorker(FakeShell(), task_id="toy", seed=0, run_dir=tmp_path / "run",
                       execution_mode="stepped", recording=config)
    worker.close()
    assert manifest(output)["status"] == "complete"
    assert len(decode(output)) == 1
    assert not (tmp_path / "run/recording").exists()


def test_custom_directory_preserves_existing_files(tmp_path):
    original = tmp_path / "overview.mp4"
    original.write_bytes(b"existing recording")
    for value in (tmp_path, original):
        with pytest.raises(ValueError, match="new or empty"):
            recording_config({"directory": value})
    assert original.read_bytes() == b"existing recording"


def test_odd_video_dimensions_rejected_before_native_setup(tmp_path):
    with pytest.raises(ValueError, match="even"):
        Recorder(None, tmp_path / "video", {"views": [
            {"name": "x", "source": "third_person", "height": 721},
        ]})


def test_unusable_codec_rejects_startup(tmp_path, renderer, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/bin/false")
    with pytest.raises(RuntimeError, match="non-zero exit"):
        Recorder(None, tmp_path / "video", {})
    assert renderer.closed


def test_encoder_disk_failure_reports_incomplete(tmp_path, renderer, monkeypatch):
    path = tmp_path / "video"

    def create(*_):
        (path / "overview.mp4").symlink_to("/dev/full")
        return renderer

    monkeypatch.setattr("roboenv_runtime.spectator.recording.create_spectator_renderer", create)
    recorder = Recorder(None, path, {})
    recorder.close(0.2)
    assert manifest(path)["status"] == "incomplete"
    assert manifest(path)["error"]


def test_capture_failure_stops_recording_without_failing_world(tmp_path, renderer, monkeypatch):
    worker = SimWorker(FakeShell(), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode="stepped", recording={})

    def failed():
        raise RuntimeError("render failure")

    monkeypatch.setattr(renderer, "capture", failed)
    try:
        worker.world.resolve(worker.world.wait(0.2))
        assert worker.world.time_s == pytest.approx(0.2)
        assert worker.world.failure is None
        assert worker.submit()["completed"] is False
        assert manifest(tmp_path / "recording")["status"] == "incomplete"
    finally:
        worker.close()


def test_video_uses_simulation_timestamps_and_holds_missing_frames(tmp_path, renderer):
    path = tmp_path / "video"
    recorder = Recorder(None, path, {})
    renderer.color = [0, 255, 0]
    recorder.tick(0.3)
    # No real-time sleep is needed to represent a gap in simulation time.
    recorder.close(0.5, capture=False)
    result = manifest(path)
    assert result["status"] == "complete" and result["encoded_frames"] == 6
    frames = decode(path).mean(axis=(1, 2))
    assert np.all(frames[:3, 0] > 240) and np.all(frames[3:, 1] > 240)
    mapping = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    assert mapping == [
        {"source_time_s": 0.0, "frame_start": 0, "frame_stop": 3},
        {"source_time_s": 0.3, "frame_start": 3, "frame_stop": 6},
    ]
    assert renderer.closed and not recorder.process.is_alive()


def test_stalled_encoder_drops_without_blocking_and_preserves_duration(tmp_path, renderer):
    path = tmp_path / "video"
    recorder = Recorder(None, path, {})
    os.kill(recorder.process.pid, signal.SIGSTOP)
    try:
        started = time.monotonic()
        for index in range(1, 21):
            recorder.tick(index / 10)
        assert time.monotonic() - started < 1
        assert recorder.summary["dropped_samples"] > 0
    finally:
        os.kill(recorder.process.pid, signal.SIGCONT)
        recorder.close(2, capture=False)
    assert manifest(path)["encoded_frames"] == 21
    assert len(decode(path)) == 21


@pytest.mark.parametrize("failure", ["death", "stall", "disk"])
def test_encoder_failures_are_bounded_and_report_incomplete(tmp_path, renderer, failure):
    path = tmp_path / "video"
    recorder = Recorder(None, path, {})
    if failure == "death":
        recorder.process.kill()
        recorder.process.join(timeout=2)
        recorder.tick(0.2)
    elif failure == "stall":
        os.kill(recorder.process.pid, signal.SIGSTOP)
        started = time.monotonic()
        recorder.close(0.2)
        assert time.monotonic() - started < 7
    else:
        # Force final manifest replacement to fail, as on an unwritable/full volume.
        (path / "manifest.json").unlink()
        (path / "manifest.json").mkdir()
        recorder.close(0.2)
    assert recorder.closed and renderer.closed and not recorder.process.is_alive()
    if failure != "disk":
        assert manifest(path)["status"] == "incomplete"
        assert manifest(path)["error"]


@pytest.mark.parametrize("boundary", ["submit", "reset", "close"])
def test_worker_recording_lifecycle_keeps_verdict_and_budget(tmp_path, renderer, boundary):
    worker = SimWorker(FakeShell(), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode="stepped", recording={})
    try:
        before = worker.adapter.budget().as_dict()
        worker.world.resolve(worker.world.wait(0.2))
        if boundary == "submit":
            result = worker.submit()
            assert result["budget"] == before and result["completed"] is False
        elif boundary == "reset":
            worker.reset({})
        else:
            worker.close()
        assert worker.world.failure is None
        result = manifest(tmp_path / "recording")
        expected = "interrupted" if boundary == "reset" else "complete"
        assert result["status"] == expected
        assert result["encoded_frames"] == 3
        worker.world.after_advance()  # a reset cannot restart the old recording
        assert manifest(tmp_path / "recording") == result
    finally:
        worker.close()


def test_cli_record_options_are_forwarded_and_require_record(monkeypatch, tmp_path, capsys):
    from roboenv.cli.main import main

    captured = {}

    def launch(task, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(record={"recording": "private"}, stop_logs=lambda: None)

    monkeypatch.setattr("roboenv.environment.launch", launch)
    path = tmp_path / "views.json"
    path.write_text('[{"name":"hand","source":"wrist","height":480}]')
    assert main(["launch", "x", "--record", "--record-fps", "5", "--record-views",
                 str(path), "--detach"]) == 0
    assert captured["recording"] == {"fps": 5, "views": json.loads(path.read_text())}
    destination = tmp_path / "my recordings"
    assert main(["launch", "x", "--record", str(destination), "--detach"]) == 0
    assert captured["recording"] == {"directory": destination}
    captured.clear()
    assert main(["launch", "x", "--record-fps", "5"]) == 1
    assert not captured
    assert "require --record" in capsys.readouterr().out


def test_supervisor_normal_shutdown_drains_encoder_children(tmp_path, renderer):
    from pathlib import Path

    from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

    # A fresh owner with real encoding children distinguishes PID shutdown from
    # signalling the entire group, which would kill FFmpeg before its drain.
    fixture = tmp_path / "owner.py"
    fixture.write_text('''
import json
import signal
import sys
import time
from pathlib import Path
from test_recording import Renderer
from roboenv_runtime.spectator import recording

if __name__ == "__main__":
    path = Path(sys.argv[1])
    recording.create_spectator_renderer = lambda *_: Renderer()
    recorder = recording.Recorder(None, path / "video", {})
    def stop(*_):
        recorder.close(0.3)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    (path / "ready.json").write_text(json.dumps([recorder.process.pid, *recorder.pids]))
    while True:
        time.sleep(0.01)
''')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    process = subprocess.Popen([sys.executable, str(fixture), str(tmp_path)],
                               env=env, start_new_session=True)
    supervisor = Supervisor(SupervisorConfig(
        run_dir=tmp_path / "supervisor", task="test", seed=0,
        budget_mode="practice", session="test",
    ))
    supervisor._worker, supervisor._worker_pgid = process, process.pid
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        children = json.loads((tmp_path / "ready.json").read_text())
        supervisor.terminate_worker(reason="normal shutdown")
        assert process.returncode == 0
        assert manifest(tmp_path / "video")["status"] == "complete"
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
