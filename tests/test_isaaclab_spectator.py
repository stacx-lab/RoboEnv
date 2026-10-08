"""Isaac Lab view composition and render-only boundary, without GPU imports."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.isaaclab.common.spectator import fit_view, look_at
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.spectator import sources as go2_sources
from roboenv_runtime.adapters.isaaclab.robots.yam.spectator import sources as yam_sources
from roboenv_runtime.adapters.urdf_geometry import matrix
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer
from roboenv_runtime.spectator.isaaclab import Engine


def test_look_at_optical_axes_and_horizon():
    eye, target = np.array([-1.1, -0.15, 1.85]), np.array([0.5, 0, 1.05])
    transform = matrix(look_at(eye, target))
    local = transform[:3, :3].T @ (target - eye)
    np.testing.assert_allclose(local[:2], 0, atol=1e-12)
    assert local[2] > 0
    assert transform[2, 0] == pytest.approx(0)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize(
    "eye,target", [([0, 0, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, 0]), ([np.nan, 0, 0], [1, 0, 0])]
)
def test_degenerate_look_at_is_rejected(eye, target):
    with pytest.raises(ValueError):
        look_at(eye, target)


def test_fit_view_contains_off_axis_custom_targets_with_margin():
    points = np.array([[0, 0, 0], [8, 7, 2], [-2, -4, 0]])
    pose = fit_view([3.6, -4, 2.6], [1.2, 0, 0.45], points)
    transform = matrix(pose)
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    ndc = local[:, :2] / local[:, 2, None] / (np.tan(np.deg2rad(25)) * np.array([16 / 9, 1]))
    assert np.all(np.abs(ndc) <= 0.8 + 1e-10)
    assert np.all(local[:, 2] > 0)


def test_control_view_follows_xy_without_roll_pitch_or_heading():
    position = np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    env = SimpleNamespace(
        task_name="go2-d1-control",
        native=SimpleNamespace(
            root_pose=lambda: position.copy(), scene=SimpleNamespace(spectator_pose=lambda: None)
        ),
    )
    source = go2_sources(env)["third_person"]
    before = source.pose()
    position[:3] += [2, -3, 0.1]
    after = source.pose()
    np.testing.assert_allclose(np.array(after[:3]) - before[:3], [2, -3, 0])
    np.testing.assert_allclose(after[3:], before[3:])


def test_yam_view_uses_front_opening_and_no_public_calibration():
    source = yam_sources(SimpleNamespace(profiles={"camera_top": "???"}))["third_person"]
    transform = matrix(source.pose())
    assert transform[0, 3] < 0  # outside the open front, not behind an opaque wall
    assert abs(transform[1, 3]) < 0.65
    assert (source.width, source.height, source.fov_y) == (1280, 720, 50.0)


def test_registered_engine_uses_dedicated_source_and_only_render(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "isaaclab_physx.renderers.isaac_rtx_renderer_utils",
        SimpleNamespace(ensure_isaac_rtx_render_update=lambda: calls.append("pump")),
    )
    world = SimpleNamespace(
        native=SimpleNamespace(sim=SimpleNamespace(render=lambda: calls.append("render"))),
        spectator_sources=lambda: yam_sources(None),
    )
    adapter = SimpleNamespace(name="isaaclab", environment=world)
    monkeypatch.setattr(Engine, "open", lambda self, views: calls.append(views))
    renderer = create_spectator_renderer(adapter)
    assert [(v.name, v.width, v.height) for v in renderer.views] == [("third_person", 1280, 720)]
    assert renderer.views[0].fov_y == 50.0
    renderer._engine.render()
    assert calls[-2:] == ["render", "pump"]
    params = renderer._engine.source_attributes(renderer._engine.sources["third_person"])
    assert params["horizontalAperture"] / params["verticalAperture"] == pytest.approx(16 / 9)
    assert np.rad2deg(2 * np.arctan(params["verticalAperture"] / (2 * params["focalLength"]))) \
        == pytest.approx(renderer.views[0].fov_y)
    view = ViewSpec("custom", pose=(1, 2, 3, 0, 0, 0, 1))
    np.testing.assert_allclose(
        renderer._engine.view_transform(view), matrix(view.pose) @ np.diag([1, -1, -1, 1])
    )
    renderer.close()


def test_second_renderer_fails_explicitly():
    with pytest.raises(ValueError, match="only one"):
        Engine(SimpleNamespace(environment=SimpleNamespace(_spectator_opened=True)))
