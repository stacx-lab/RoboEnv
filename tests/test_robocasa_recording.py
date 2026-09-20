"""Private native state/video evidence round-trips without changing physics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_state_recording_replay_and_tamper_detection(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from roboenv_runtime.adapters.robocasa.recording import OwnerRecording, replay

    monkeypatch.setenv("MUJOCO_GL", "egl")
    xml = '''<mujoco><visual><global offwidth="128" offheight="128"/></visual>
    <worldbody><light pos="0 0 3"/><camera name="left" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8"/>
    <camera name="wrist" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8"/>
    <geom type="plane" size="3 3 .1"/><body pos="0 0 1"><joint name="hinge"/>
    <geom type="box" size=".2 .1 .1" mass="1"/></body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)

    def render(camera):
        renderer.update_scene(data, camera=camera)
        return renderer.render()

    world = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=model, get_xml=lambda: xml),
        data=SimpleNamespace(_data=data)), _camera_names=("left", "wrist"), _camera_size=(64, 64),
        env=SimpleNamespace(control_freq=20), task_name="toy", seed=0, _step_count=0,
        episode_meta=lambda: {}, task_completed=lambda: False, render=render)
    before = data.qpos.copy()
    recording = OwnerRecording(world, tmp_path / "recording")
    assert np.array_equal(data.qpos, before)
    for i in range(4):
        recording.command("write", {"arm": [i / 10]})
        data.qvel[0] = .1
        mujoco.mj_step(model, data)
        world._step_count += 1
        recording.capture()
    recording.close()
    renderer.close()
    result = replay(tmp_path / "recording", tmp_path / "replay", size=128)
    assert result["passed"] and result["states_verified"] == 6
    assert result["max_state_difference"] == 0
    assert result["command_resimulation"] is False
    manifest = json.loads((tmp_path / "recording/manifest.json").read_text())
    assert manifest["video_frames"] == 3 and manifest["final_frame"] == 4
    with (tmp_path / "recording/states.f64").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        replay(tmp_path / "recording", tmp_path / "invalid", size=128)
