"""Spectator contract and native-boundary isolation, without GPU dependencies."""

import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.spectator import CameraSource, SpectatorRenderer, ViewSpec
from roboenv_runtime.spectator.isaac import Engine as IsaacEngine
from roboenv_runtime.spectator.mujoco import Engine as MujocoEngine
from roboenv_runtime.spectator.sapien import Engine as SapienEngine


class Engine:
    sources = {"third_person": CameraSource(None, 640, 480)}

    def __init__(self):
        self.closed = False

    def open(self, views):
        self.buffers = {v.name: np.zeros((v.height, v.width, 3), np.uint8) for v in views}

    def capture(self, views):
        return self.buffers

    def close(self):
        self.closed = True


def test_views_have_independent_resolution_and_detached_buffers():
    engine = Engine()
    renderer = SpectatorRenderer(engine, [
        ViewSpec("overview", source="third_person", height=360),
        ViewSpec("detail", source="third_person", width=1024, height=768),
        ViewSpec("outside", pose=(1, 2, 3, 0, 0, 0, 1), height=720),
    ])
    assert [(v.width, v.height) for v in renderer.views] == [(480, 360), (1024, 768), (1280, 720)]
    frames = renderer.capture()
    engine.buffers["overview"][:] = 255
    assert not frames["overview"].any()
    assert engine.sources["third_person"].width == 640
    renderer.close()
    renderer.close()
    with pytest.raises(RuntimeError, match="closed"):
        renderer.capture()


@pytest.mark.parametrize("kwargs", [
    {"source": "third_person", "height": 0},
    {"source": "third_person", "height": None},
    {"source": "third_person", "width": True},
    {"source": "third_person", "fov_y": float("nan")},
    {"source": "third_person", "pose": (0, 0, 0, 0, 0, 0, 1)},
    {"pose": (0, 0, 0, 0, 0, 0, 0)},
    {},
])
def test_invalid_views_fail_before_native_effects(kwargs):
    with pytest.raises(ValueError):
        ViewSpec("view", **kwargs)


@pytest.mark.parametrize("views", [[], [ViewSpec("x", source="unknown")],
                                      [ViewSpec("x", source="third_person")] * 2])
def test_unknown_and_duplicate_views_close_setup(views):
    engine = Engine()
    with pytest.raises(ValueError):
        SpectatorRenderer(engine, views)
    assert engine.closed


def test_native_shape_failures_and_thread_affinity_are_explicit():
    engine = Engine()
    renderer = SpectatorRenderer(engine, None)
    errors = []

    def wrong_thread():
        try:
            renderer.capture()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert errors and "owner thread" in errors[0]
    engine.buffers["third_person"] = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match="shape"):
        renderer.capture()
    renderer.close()


def test_sapien_follows_wrist_without_updating_source_camera_or_task():
    calls = []
    source = SimpleNamespace(get_width=lambda: 4, get_height=lambda: 4,
                             get_global_pose=lambda: "old camera pose")
    world = SimpleNamespace(
        _camera=lambda name: source,
        task=SimpleNamespace(
            scene=SimpleNamespace(update_render=lambda: calls.append("sync")),
            robot=SimpleNamespace(left_camera=SimpleNamespace(get_pose=lambda: "live wrist")),
        ),
    )
    engine = SapienEngine(SimpleNamespace(_env=world))
    engine.cameras["wrist"] = SimpleNamespace(
        entity=SimpleNamespace(set_pose=lambda pose: calls.append(pose)),
        take_picture=lambda: calls.append("picture"),
        get_picture=lambda name: np.ones((4, 4, 4)),
    )
    frames = engine.capture([ViewSpec("wrist", source="left_wrist", width=4, height=4)])
    assert calls == ["live wrist", "sync", "picture"]
    assert frames["wrist"].shape == (4, 4, 3)
    assert np.all(frames["wrist"] == 255)


def test_behavior_view_roles_follow_mounts_not_camera_numbers():
    parents = {"camera_0": "right_arm", "camera_1": "trunk", "camera_2": "left_arm"}
    sensors = {name: SimpleNamespace(image_width=640, image_height=480) for name in parents}
    world = SimpleNamespace(
        cameras={name: name for name in parents},
        device_facts={name: {"extrinsics": {"mount": {"parent": parent}}}
                      for name, parent in parents.items()},
        robot=SimpleNamespace(sensors=sensors),
    )
    engine = IsaacEngine(SimpleNamespace(_env=world, name="behavior"))
    assert engine.sources["left_wrist"].native is sensors["camera_2"]
    assert engine.sources["third_person"].native is sensors["camera_1"]
    assert engine.sources["right_wrist"].native is sensors["camera_0"]


@pytest.mark.parametrize("fail_write", [False, True])
def test_behavior_capture_edits_usd_before_rendering(monkeypatch, fail_write):
    calls = []
    editing = False

    @contextmanager
    def editing_usd():
        nonlocal editing
        assert not editing
        editing = True
        calls.append("enter")
        try:
            yield
        finally:
            editing = False
            calls.append("sync")

    def set_transform(value):
        assert editing, "USD edit outside editing_usd"
        assert np.asarray(value).shape == (4, 4)
        calls.append("write")
        if fail_write:
            raise RuntimeError("camera write failed")

    def render():
        assert not editing
        calls.append("render")

    def tensor(value):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(value)))

    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(
        sim=SimpleNamespace(editing_usd=editing_usd, render=render),
    ))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Gf=SimpleNamespace(Matrix4d=lambda value: value),
    ))
    engine = IsaacEngine.__new__(IsaacEngine)
    engine.backend = "behavior"
    sensor = SimpleNamespace(get_position_orientation=lambda: (
        tensor([1, 2, 3]), tensor([0, 0, 0, 1]),
    ))
    engine.sources = {"left_wrist": CameraSource(sensor, 4, 4)}
    views = [ViewSpec("wrist", source="left_wrist", width=4, height=4),
             ViewSpec("fixed", pose=(0, 0, 0, 0, 0, 0, 1), width=4, height=4)]
    engine.cameras = {v.name: SimpleNamespace(Set=set_transform) for v in views}
    engine.annotators = {v.name: SimpleNamespace(
        get_data=lambda: np.full((4, 4, 4), 42, np.uint8),
    ) for v in views}

    if fail_write:
        with pytest.raises(RuntimeError, match="camera write failed"):
            engine.capture(views)
        assert calls == ["enter", "write", "sync"]
    else:
        frames = engine.capture(views)
        assert calls == ["enter", "write", "write", "sync", "render"]
        assert set(frames) == {"wrist", "fixed"}
        assert all(frame.shape == (4, 4, 3) and np.all(frame == 42)
                   for frame in frames.values())
    assert not editing


def test_mujoco_restores_owner_context_on_render_failure():
    calls = []
    engine = MujocoEngine.__new__(MujocoEngine)
    engine.abc = False
    engine.gl = SimpleNamespace(make_current=lambda: calls.append("spectator"))
    engine.world = SimpleNamespace(sim=SimpleNamespace(_render_context_offscreen=SimpleNamespace(
        gl_ctx=SimpleNamespace(make_current=lambda: calls.append("owner")),
    )))
    with pytest.raises(RuntimeError, match="render failed"), engine._current():
        raise RuntimeError("render failed")
    assert calls == ["spectator", "owner"]
