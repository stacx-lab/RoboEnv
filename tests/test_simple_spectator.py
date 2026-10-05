"""Native-frame recording keeps SIMPLE's cameras and physics under their owner."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer

NAMES = ("camera_head_left", "camera_head_right")


def simple_adapter(mode):
    specs = {
        name: {"native": "head_stereo_" + side, "width": 640, "height": 360}
        for name, side in zip(NAMES, ("left", "right"), strict=True)
    }
    images = {spec["native"]: np.full((360, 640, 3), [255, 0, 0], np.uint8)
              for spec in specs.values()}
    calls = []

    def render(which):
        calls.append(which)
        return images

    world = SimpleNamespace(
        sim_mode=mode, model={"cameras": specs},
        env=SimpleNamespace(
            mujoco=SimpleNamespace(render=lambda: render("mujoco")),
            isaac=SimpleNamespace(render=lambda: render("isaac")),
        ),
    )
    return SimpleNamespace(name="simple", _env=world), images, calls


@pytest.mark.parametrize("mode,native", [("mujoco", "mujoco"), ("mujoco_isaac", "isaac")])
def test_simple_default_frames_are_native_detached_and_read_only(mode, native):
    adapter, images, calls = simple_adapter(mode)
    renderer = create_spectator_renderer(adapter)
    assert [(v.name, v.source, v.width, v.height) for v in renderer.views] == [
        (name, name, 640, 360) for name in NAMES
    ]
    assert not calls  # setup does not render, step or change the camera graph
    result = renderer.capture()
    assert calls == [native]
    images["head_stereo_left"][:] = [0, 255, 0]
    assert np.all(result[NAMES[0]][..., 0] == 255)
    assert np.all(renderer.capture()[NAMES[0]][..., 1] == 255)
    renderer.close()
    renderer.close()
    assert calls == [native, native]
    assert list(adapter._env.model["cameras"]) == list(NAMES)
    assert adapter._env.env.isaac.render()["head_stereo_left"].shape == (360, 640, 3)


def test_simple_explicit_size_resamples_without_resizing_source_cameras():
    adapter, images, calls = simple_adapter("mujoco_isaac")
    renderer = create_spectator_renderer(adapter, [ViewSpec("left", source=NAMES[0], height=720)])
    result = renderer.capture()
    assert list(result) == ["left"] and result["left"].shape == (720, 1280, 3)
    assert images["head_stereo_left"].shape == (360, 640, 3)
    assert adapter._env.model["cameras"][NAMES[0]]["width"] == 640
    assert calls == ["isaac"]
    renderer.close()


@pytest.mark.parametrize("view,match", [
    (ViewSpec("wrist", source="wrist"), "Unknown camera"),
    (ViewSpec("fixed", pose=(0, 0, 0, 0, 0, 0, 1)), "fixed spectator poses"),
])
def test_unsupported_simple_views_fail_before_rendering(view, match):
    adapter, _, calls = simple_adapter("mujoco_isaac")
    with pytest.raises(ValueError, match=match):
        create_spectator_renderer(adapter, [view])
    assert not calls


@pytest.mark.parametrize("frame", [np.zeros((360, 640, 3)), np.zeros((0,), np.uint8)])
def test_invalid_native_buffers_fail_explicitly(frame):
    adapter, images, _ = simple_adapter("mujoco_isaac")
    renderer = create_spectator_renderer(adapter)
    images["head_stereo_left"] = frame
    with pytest.raises(ValueError, match="shape/type"):
        renderer.capture()
    renderer.close()


def test_launch_accepts_simple_recording_before_native_startup(tmp_path, monkeypatch):
    from roboenv import environment
    from roboenv.provisioning import gpu

    selection = {"index": 0, "uuid": "GPU-test", "vulkan_index": 0, "cuda_index": 0}
    monkeypatch.setattr(gpu, "resolve_gpu", lambda value: selection)
    monkeypatch.setattr("roboenv.provisioning.sources.status", lambda *a: {"ready": True})
    seen = []

    def capture(command, **kwargs):
        seen.append(kwargs["env"])
        raise OSError("stop before native startup")

    monkeypatch.setattr(environment.subprocess, "Popen", capture)
    with pytest.raises(OSError, match="before native startup"):
        environment.launch("simple/g1-xmove-pick", project=Path(__file__).resolve().parents[1],
                           interpreter=sys.executable, recording={}, run_dir=tmp_path)
    assert len(seen) == 1
    record = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert record["recording"]
    shutil.rmtree(seen[0]["ROBOENV_SOCKET_ROOT"])


def test_simple_stereo_recording_encodes_both_views_and_final_sample(tmp_path):
    from roboenv_runtime.spectator.recording import Recorder

    tmp_path = tmp_path / "video"
    if not shutil.which("ffmpeg"):
        pytest.skip("optional FFmpeg unavailable")
    adapter, images, _ = simple_adapter("mujoco_isaac")
    recorder = Recorder(adapter, tmp_path, {"views": [
        {"name": name, "source": name, "height": 36} for name in NAMES
    ]})
    for frame in images.values():
        frame[:] = [0, 255, 0]
    recorder.close(0.3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["encoded_frames"] == 4
    assert manifest["dropped_samples"] == 0
    for name in NAMES:
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(tmp_path / (name + ".mp4")),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1",
        ], capture_output=True, check=True, timeout=10)
        frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, 36, 64, 3)
        assert len(frames) == 4
        assert frames[0, ..., 0].mean() > 240 and frames[-1, ..., 1].mean() > 240
    mapping = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert [row["source_time_s"] for row in mapping] == [0.0, 0.3]


@pytest.mark.parametrize("code", [0, -11])
def test_native_probe_does_not_certify_a_shutdown_crash(tmp_path, code):
    tool = run_path(str(Path(__file__).resolve().parents[1] / "tools/check_spectator.py"))
    (tmp_path / "result.json").write_text(json.dumps({"passed": True}))
    report = tool["finish_report"](tmp_path, code)
    assert report["isolation_passed"] is True
    assert report["native_exit_code"] == code and report["passed"] is (code == 0)
    assert json.loads((tmp_path / "result.json").read_text()) == report
    if code:
        assert "after isolation checks" in report["error"]
