"""Calibration convention and render-buffer regression checks (no GPU)."""

from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, ExecutionContext
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics


@pytest.mark.parametrize("width,height,fov", [(256, 256, 45), (320, 240, 37), (641, 359, 70)])
def test_integer_pixel_centers_span_the_declared_field_of_view(width, height, fov):
    color = intrinsics(width, height, fov, False)["color"]
    k = color["projection"]
    assert k["cx"] == (width - 1) / 2
    assert k["cy"] == (height - 1) / 2
    assert color["fov"]["vertical_deg"] == pytest.approx(fov)
    # Symmetric rays at the outer pixel edges, not at a half-pixel-shifted center.
    left, right = (-0.5 - k["cx"]) / k["fx"], (width - 0.5 - k["cx"]) / k["fx"]
    assert left == -right
    assert color["fov"]["horizontal_deg"] == pytest.approx(np.degrees(2 * np.arctan(right)))


def test_robocasa_rebuilds_multisample_buffers_once_per_model():
    calls = []
    quality = SimpleNamespace(offsamples=4)
    context = SimpleNamespace(
        gl_ctx=SimpleNamespace(make_current=lambda: calls.append("current")),
        con=SimpleNamespace(free=lambda: calls.append("free")),
        _set_mujoco_context_and_buffers=lambda: calls.append("buffers"),
    )
    world = RoboCasaEnv.__new__(RoboCasaEnv)
    world.env = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=SimpleNamespace(vis=SimpleNamespace(quality=quality))),
        _render_context_offscreen=context,
    ))
    world._pixel_center_rendering()
    assert quality.offsamples == 0
    assert calls == ["current", "free", "buffers"]
    world._pixel_center_rendering()
    assert len(calls) == 3
    quality.offsamples = 4  # A new/reset model needs new buffers again.
    world._pixel_center_rendering()
    assert len(calls) == 6


def test_camera_read_returns_handler_payload_without_calibration_metadata():
    device = BoundDevice(
        DeviceDefinition("camera", "rgb", {"test": [1]}, {"mount": [2]}),
        {"read": Action(lambda args, execution: {"sample_time_s": 3.0})},
        implementation="fixture",
    )
    result = device.execute("read", {}, ExecutionContext(read_only=True))
    assert result == {"sample_time_s": 3.0}


@pytest.mark.parametrize("mismatch", [False, True])
def test_offline_sapien_camera_preserves_integer_pixel_centers(mismatch):
    from roboenv_runtime.adapters.native_video import SapienCamera

    expected = np.array([[8.0, 0, 3.5], [0, 8.0, 2.5], [0, 0, 1]])
    native = expected.copy()
    native[:2, 2] += 0.5
    measured = native.copy()
    measured[0, 0] += 1 if mismatch else 0
    settings, poses, removed = [], [], []
    pose = object()
    source = SimpleNamespace(get_near=lambda: 0.1, get_far=lambda: 100,
                             get_global_pose=lambda: pose)
    clone = SimpleNamespace(
        set_perspective_parameters=lambda *args: settings.append(args),
        entity=SimpleNamespace(set_pose=poses.append), take_picture=lambda: None,
        get_intrinsic_matrix=lambda: measured, get_picture=lambda _: np.ones((6, 8, 4)),
    )
    world = SimpleNamespace(_camera=lambda _: source, task=SimpleNamespace(
        _update_render=lambda: None, scene=SimpleNamespace(
            add_camera=lambda *args: clone, update_render=lambda: None,
            remove_camera=removed.append,
        ),
    ))
    camera = SapienCamera(world, "wrist", 8, 6, expected)
    try:
        assert settings == [(0.1, 100, 8.0, 8.0, 4.0, 3.0, 0.0)]
        if mismatch:
            with pytest.raises(ValueError, match="projection differs"):
                camera.read()
        else:
            assert camera.read().shape == (6, 8, 3)
        assert poses == [pose]
    finally:
        camera.close()
    assert removed == [clone]
