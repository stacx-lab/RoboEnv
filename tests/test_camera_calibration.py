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
        {"read": Action(lambda args, execution: {"timestamp": 3.0})},
        implementation="fixture",
    )
    result = device.execute("read", {}, ExecutionContext(read_only=True))
    assert result == {"timestamp": 3.0}
