"""Native shutdown cannot erase a replay verdict or hide a failed comparison."""

import json
from types import SimpleNamespace

import pytest
from helpers import load_tool
from roboenv_runtime.adapters import native_replay as tool


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_replay_persists_result_before_native_shutdown_exits(monkeypatch, tmp_path, mismatch, mode):
    state = {"pose": [0.0]}
    budget = {"used": 0, "max": 1 if mode == "stepped" else None}
    native = SimpleNamespace(
        implementation_identity={"version": "fixture"},
        initialization={"requested_seed": 0},
        initial_state=state,
        owner_state=lambda: {"pose": [1.0]} if mismatch else state,
        control=SimpleNamespace(frame=0, targets={"arm": [0.0]}),
        task_completed=lambda: False,
    )

    def close():
        raise SystemExit(0)

    configured = []
    adapter = SimpleNamespace(
        _env=native, embodiment="fixture", reset=lambda seed: None, close=close,
        configure_execution_mode=configured.append,
        budget=lambda: SimpleNamespace(as_dict=lambda: budget),
    )
    monkeypatch.setattr(tool, "make_adapter", lambda *args, **kwargs: adapter)
    record = {
        "schema": "roboenv/native-device-episode/1",
        "backend": "robodojo",
        "budget": budget,
        "environment": {},
        "initialization": native.initialization,
        "implementation": native.implementation_identity,
        "embodiment": "fixture",
        "initial_state": state,
        "commands": [],
        "state": state,
        "control_state": {"targets": native.control.targets, "gripper_holds": {}},
        "native_success": False, "execution_mode": mode,
    }
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exit_info:
        tool.replay(record, out=out, recording_sha256="fixture recording")
    assert exit_info.value.code == 0
    assert configured == [mode]
    report = json.loads(out.read_text())
    assert report["recording_sha256"] == "fixture recording"
    assert report["passed"] is not mismatch
    if mismatch:
        assert "Replay state differs" in report["error"]
    else:
        assert report["record_sha256"] == tool.digest(record)


@pytest.mark.parametrize("name", ["native", "simple"])
@pytest.mark.parametrize("mode", [None, "invalid", True])
def test_replay_rejects_missing_or_invalid_mode_before_creating_simulator(monkeypatch, name, mode):
    module = tool if name == "native" else load_tool("replay_simple_devices")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid mode must be rejected before simulator creation")

    monkeypatch.setattr(module, "make_adapter" if name == "native" else "SimpleEnv", unexpected)
    schema = "roboenv/native-device-episode/1" if name == "native" else "roboenv/simple-record/3"
    record = {"schema": schema, "backend": "robodojo"}
    if mode is not None:
        record["execution_mode"] = mode
    with pytest.raises(ValueError, match="execution_mode"):
        module.replay(record)


@pytest.mark.parametrize("backend", ["robotwin", "robodojo", "behavior"])
@pytest.mark.parametrize("failure", [None, "state", "render"])
def test_native_video_streams_control_frames_and_rejects_failed_replays(
    monkeypatch, tmp_path, backend, failure
):
    import shutil
    import subprocess

    import numpy as np
    from roboenv_runtime.adapters import native_video
    from roboenv_runtime.adapters.device_control import DeviceControl

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real video encoder")
    root = tmp_path / "render"
    initial = {"pose": [0.0]}
    position = [0.0]
    capture_frames = []
    restored_states = []
    projection = {"fx": 4, "fy": 4, "cx": 1.5, "cy": 1.0} if backend == "robotwin" else {
        "fx": 4, "fy": 4, "cx": 2, "cy": 1.5,
    }
    native = SimpleNamespace(
        implementation_identity={"version": "fixture"},
        initialization={"requested_seed": 0},
        initial_state={"pose": [0.1]} if backend == "robodojo" else initial,
        owner_state=lambda: {"pose": list(position)},
        control_interval=0.04,
        task_completed=lambda: False,
        prepare_replay_materials=lambda: [],
        device_facts={
            name: {
                "type": "rgb",
                "intrinsics": {"color": {"width": 4, "height": 3, "projection": projection}},
            }
            for name in ("head", "wrist")
        },
    )

    def restore_initial_state(state):
        restored_states.append(state)
        native.initial_state = state
        return {"source": "archived-native-initial-state", "final_state_restored": False}

    native.restore_replay_initial_state = restore_initial_state

    def advance():
        position[0] += native.control.targets["arm"][0]

    native.control = DeviceControl(
        {"arm": [(-1, 1)]},
        {"arm": [0.0]},
        lambda _: None,
        advance,
        interval_s=0.04,
        exhausted=lambda: native.control.frame >= 5,
    )
    native.device_write = native.control.write
    native.device_stop = lambda name: native.control.stop(name, position)
    adapter = SimpleNamespace(
        _env=native,
        embodiment="fixture",
        reset=lambda _: None,
        configure_execution_mode=lambda _: None,
        close=lambda: None,
        budget=lambda: SimpleNamespace(as_dict=lambda: {"used": native.control.frame, "max": 5}),
    )
    monkeypatch.setattr(tool, "make_adapter", lambda *args, **kwargs: adapter)

    class Camera:
        def __init__(self, world, *args):
            self.width, self.height = args[-3:-1]
            self.render_subframes = 4
            if backend == "robotwin":
                np.testing.assert_allclose(
                    args[-1][:2, 2], [(self.width - 1) / 2, (self.height - 1) / 2],
                )

        def read(self):
            if failure == "render" and native.control.frame:
                raise RuntimeError("fixture renderer failed")
            capture_frames.append(native.control.frame)
            return np.full((self.height, self.width, 3), native.control.frame * 60, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(native_video, "SapienCamera", Camera)
    monkeypatch.setattr(native_video, "IsaacCamera", Camera)
    record = {
        "schema": "roboenv/native-device-episode/1",
        "backend": backend,
        "execution_mode": "stepped",
        "budget": {"used": 3, "max": 5},
        "environment": {},
        "initialization": native.initialization,
        "implementation": native.implementation_identity,
        "embodiment": "fixture",
        "initial_state": initial,
        "commands": [
            {"frame": 0, "op": "write", "commands": {"arm": [0.2]}},
            {"frame": 2, "op": "write", "commands": {"arm": [0.5]}},
        ],
        "state": {"pose": [8.0 if failure == "state" else 0.9]},
        "control_state": {"targets": {"arm": [0.5]}, "gripper_holds": {}},
        "native_success": False,
    }
    out = root / "replay.json"
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            tool.replay(record, out=out, video_dir=root, height=12, stride=2)
        report = json.loads(out.read_text())
        assert report["passed"] is False
        if backend == "robodojo":
            assert restored_states == [initial]
            assert report["initial_state_restoration"]["final_state_restored"] is False
        assert not (root / "camera-alignment.json").exists()
        return
    report = tool.replay(record, out=out, video_dir=root, height=12, stride=2)
    assert restored_states == ([initial] if backend == "robodojo" else [])
    if backend == "robodojo":
        assert report["initial_state_restoration"]["source"] == "archived-native-initial-state"
    assert report["passed"] and report["kind"] == "native-command-replay"
    assert report["intermediate_states_verified"] is False
    assert capture_frames == [0, 0, 2, 2, 3, 3]
    assert report["media"]["frames"] == 3
    assert report["media"]["fps"] == 12.5
    frames = json.loads((root / "frames.json").read_text())
    assert [f["frame"] for f in frames] == [0, 2, 3]
    assert frames[-1]["simulation_seconds"] == pytest.approx(0.12)
    for name, digest in report["media"]["files"].items():
        assert native_video.file_digest(root / name) == digest
    metadata = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(root / "camera_0.mp4")]
        )
    )
    assert (metadata["streams"][0]["width"], metadata["streams"][0]["height"]) == (16, 12)
    assert int(metadata["streams"][0]["nb_frames"]) == 3
    alignment = json.loads((root / "camera-alignment.json").read_text())
    assert alignment["passed"] and alignment["original_episode_image_comparison"] is False
    with pytest.raises(ValueError, match="Preserve existing"):
        tool.replay(record, out=out, video_dir=root)
