"""ABC discovery and control contracts, with optional CPU MuJoCo checks."""

import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.devices.client import Environment
from roboenv.tasks.abc import discover_tasks, entries
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.abc.adapter import frame_budget

ROOT = Path(__file__).resolve().parents[1]


def test_abc_registry_discovery_preserves_tasks_and_rigs(tmp_path):
    source = ROOT / "third_party/abc"
    if not (source / "abc_sim/task_specs.py").exists():
        pytest.skip("ABC source is not initialized")
    native = list(entries(source))
    tasks = list(discover_tasks(ROOT))
    assert len(native) == len(tasks) == 33
    assert sum(bool(t.get("evaluator_name")) for t in native) == 29
    assert {task.robot for task, _ in tasks} == {"abc/abc-yam"}
    for task, _ in tasks:
        path = tmp_path / "task.yaml"
        import yaml

        path.write_text(yaml.safe_dump(task.as_dict()))
        resolved = resolve_task(path, project=ROOT)
        assert resolved.bindings.backend == "abc"
        assert len(resolved.bindings.devices) == 8
        assert task.backend_kwargs["task_name"] in {t["name"] for t in native}
    # Discovery must remain usable without simulator dependencies.
    tree = ast.parse((ROOT / "src/roboenv/tasks/abc.py").read_text())
    assert not any(
        isinstance(n, ast.Import) and any(a.name.startswith(("abc_sim", "mujoco")) for a in n.names)
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("interval", "expected"),
    [(1 / 30, 9000), (0.002 * 17, 8823), (0.0002 * 170, 8823), (0.02, 15000)],
)
def test_stepped_budget_uses_control_intervals(interval, expected):
    assert frame_budget(interval) == expected
    assert expected * interval <= 300 + 1e-10
    assert (expected + 1) * interval > 300


@pytest.mark.parametrize("interval", [0, -1, math.nan, math.inf])
def test_invalid_native_timing_is_rejected(interval):
    with pytest.raises(ValueError):
        frame_budget(interval)


def test_task_refresh_replaces_snapshot_without_mutating_previous_instruction():
    value = {"id": "abc/multi-drawer-search", "instruction": "find the cup", "context": {}}
    transport = SimpleNamespace(describe=lambda: {"task": value}, devices=lambda: {"devices": []})
    env = Environment(transport)
    old = env.task
    value["instruction"] = "find the bottle"
    assert env.refresh_task().instruction == "find the bottle"
    assert old.instruction == "find the cup"
    assert not env.task.context


def test_recipe_stays_simulator_only():
    recipe = json.loads((ROOT / "runtime/environments.json").read_text())["backends"]["abc"]
    assert recipe["source_only"] == ["third_party/abc"]
    lock = (ROOT / recipe["lock"]).read_text().lower()
    assert "mujoco==" in lock
    assert "torch==" not in lock and "mujoco-warp==" not in lock


def test_blender_cameras_share_one_render_per_control_frame(monkeypatch):
    from roboenv_runtime.adapters.abc.rendering import BlenderImages

    calls = []

    class Provider:
        def __init__(self, *args, **kwargs):
            assert kwargs["gpu_id"] is None  # Inherit the resolved CUDA mask.
            assert kwargs["depth"] is True
            calls.append("open")

        def initial_frames(self):
            return {name: ("initial-rgb", "initial-depth") for name in ("top", "left")}

        def frames_for_step(self, frame, timestamp):
            calls.append(frame)
            return {"top": ("next-rgb", "next-depth"), "left": ("next-rgb", "next-depth")}

        def close(self):
            calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        "abc_sim.rendering.blender.live",
        SimpleNamespace(BlenderCameraProvider=Provider),
    )
    world = SimpleNamespace(
        env=SimpleNamespace(model=None, data=None),
        control=SimpleNamespace(frame=0),
        timestamp=0,
        _camera_names=("top", "left"),
        blender_settings={"gpu_id": None},
    )
    images = BlenderImages(world)
    assert images.read()["top"] == images.read()["left"] == ("initial-rgb", "initial-depth")
    world.control.frame = 1
    assert images.read()["top"] == images.read()["left"] == ("next-rgb", "next-depth")
    images.close()
    assert calls == ["open", 1, "close"]


def test_blender_preparation_rejects_wrong_backend_before_effects(monkeypatch):
    from roboenv.provisioning import setup

    monkeypatch.setattr(setup, "audit", lambda *args: pytest.fail("must reject before effects"))
    with pytest.raises(ValueError, match="Renderer preparation"):
        setup.prepare("libero", renderer="blender")


def test_blender_asset_checks_preserve_conflicting_files(tmp_path):
    from roboenv.provisioning.abc_blender import _download

    target = tmp_path / "asset"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="differs from its pin"):
        _download("unused", target, "0" * 64)
    assert target.read_bytes() == b"existing"


def test_blender_missing_installation_is_explicit(monkeypatch, tmp_path):
    from roboenv.provisioning.abc_blender import paths

    monkeypatch.setenv("ROBOENV_ASSET_CACHE", str(tmp_path))
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.delenv("ABC_HDRI", raising=False)
    with pytest.raises(ValueError, match="prepare abc --renderer blender"):
        paths(tmp_path)


@pytest.mark.parametrize("velocity", [-0.5, 0.5])
@pytest.mark.parametrize("supplemental", [False, True])
def test_observations_use_current_geometry_without_changing_native_evaluation(
    velocity, supplemental,
):
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from roboenv_runtime.adapters.abc.backend import ABCEnv

    model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="left_lf_down">
              <geom type="box" size=".006 .002 .04"/>
            </body>
            <body name="left_rf_down" pos="0 .04 0">
              <joint name="finger" type="slide" axis="0 1 0"/>
              <geom type="box" size=".006 .002 .04" mass="1"/>
              <site name="finger_tip"/>
              <camera name="wrist"/>
            </body>
          </worldbody>
          <actuator><general joint="finger" dyntype="filter"/></actuator>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    data.qvel[0], data.ctrl[0] = velocity, 0.3
    mujoco.mj_forward(model, data)
    state_names = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
    after_step, evaluations = {}, []

    def evaluate():
        evaluations.append(float(data.geom_xpos[1, 1]))
        return {"success": False, "native_pad_y": evaluations[-1]}

    def step(action, *, render_obs):
        assert not render_obs
        for _ in range(17):
            mujoco.mj_step(model, data)
        after_step.update({key: getattr(data, key).copy() for key in state_names})
        after_step["time"] = float(data.time)
        return None, 0, False, False, {"task_eval": evaluate()}

    world = ABCEnv.__new__(ABCEnv)
    world.env = SimpleNamespace(model=model, data=data, step=step)
    world._action = np.zeros(14)
    world.control = SimpleNamespace(targets={"left_gripper": [0]})
    world.supplemental = (
        SimpleNamespace(evaluate=lambda current: evaluate()) if supplemental else None
    )
    world._advance()

    expected_pad_y = 0.04 + float(data.qpos[0])
    # Preserve native predicate timing even though its derived state is older.
    assert len(evaluations) == (2 if supplemental else 1)
    assert all(abs(value - expected_pad_y) > 1e-5 for value in evaluations)
    assert world.evaluation["native_pad_y"] == evaluations[-1]
    assert world.task_completed() is False
    for _ in range(2):
        read = world.device_read("left_gripper")
        assert read["aperture_m"] == pytest.approx(expected_pad_y - 0.004)
        assert read["timestamp"] == after_step["time"]
        assert data.site_xpos[0, 1] == pytest.approx(expected_pad_y)
        assert data.cam_xpos[0, 1] == pytest.approx(expected_pad_y)
    assert len(evaluations) == (2 if supplemental else 1)
    for key in state_names:
        np.testing.assert_array_equal(getattr(data, key), after_step[key])
    assert float(data.time) == after_step["time"]


def test_mujoco_depth_invalid_pixels_and_mode_restoration():
    import numpy as np
    from roboenv_runtime.adapters.abc.rendering import mujoco_rgbd

    class Renderer:
        depth = False
        fail = False

        def disable_depth_rendering(self):
            self.depth = False

        def enable_depth_rendering(self):
            self.depth = True

        def update_scene(self, data, camera):
            assert not self.depth

        def render(self):
            if self.depth:
                if self.fail:
                    raise RuntimeError("depth failed")
                return np.array([[.001, .5, np.nan], [49.99, np.inf, -1]], np.float32)
            return np.zeros((2, 3, 3), np.uint8)

    renderer = Renderer()
    model = SimpleNamespace(vis=SimpleNamespace(map=SimpleNamespace(znear=.01, zfar=50)),
                            stat=SimpleNamespace(extent=1))
    _, depth = mujoco_rgbd(renderer, model, None, "top")
    np.testing.assert_array_equal(depth, [[0, .5, 0], [0, 0, 0]])
    assert not renderer.depth
    renderer.fail = True
    with pytest.raises(RuntimeError, match="depth failed"):
        mujoco_rgbd(renderer, model, None, "top")
    assert not renderer.depth


def test_mujoco_pair_cache_invalidates_after_control_frame(monkeypatch):
    import numpy as np
    from roboenv_runtime.adapters.abc import rendering

    calls = []
    monkeypatch.setattr(rendering, "mujoco_rgbd", lambda *args: calls.append(args[-1]) or (
        np.zeros((2, 3, 3), np.uint8), np.ones((2, 3), np.float32)))
    images = object.__new__(rendering.MujocoImages)
    images.world = SimpleNamespace(control=SimpleNamespace(frame=0),
                                   env=SimpleNamespace(model=None, data=None))
    images.renderer, images.frame, images.images = None, None, {}
    first = images.read("top")
    assert images.read("top") is first
    images.read("left")
    images.world.control.frame += 1
    assert images.read("top") is not first
    assert calls == ["top", "left", "top"]


def blender_definition(filename, name, namespace):
    """Exercise source definitions without importing GPU-only dependencies."""
    source = ROOT / "third_party/abc/abc_sim/rendering/blender" / filename
    if not source.exists():
        pytest.skip("ABC source is not initialized")
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_blender_depth_preserves_optical_z_and_invalidates_background(tmp_path):
    import numpy as np
    from roboenv_runtime.bindings.backends.shared.depth import write_depth

    convert = blender_definition("render_usd.py", "optical_depth", {"np": np})
    depth = convert(np.array([[.5, .5, .5], [np.inf, 0, 100]], np.float32), .01, 100)
    np.testing.assert_array_equal(depth, [[.5, .5, .5], [0, 0, 0]])
    path = tmp_path / "depth.pgm"
    write_depth(path, depth)
    assert path.read_bytes() == b"P5\n3 2\n65535\n" + np.array(
        [[500, 500, 500], [0, 0, 0]], dtype=">u2").tobytes()


@pytest.mark.parametrize(
    "invalid", [None, "missing", "shape", "nonfinite", "negative", "dtype", "truncated"],
)
def test_blender_pair_protocol_and_failure_cleanup(tmp_path, invalid):
    import time

    import numpy as np
    from PIL import Image

    provider_type = blender_definition("live.py", "BlenderCameraProvider", {
        "CameraProvider": object, "np": np, "time": time, "Image": Image,
    })
    color = tmp_path / "rgb.bmp"
    Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(color)
    depth_file = tmp_path / "depth.npy"
    depth = np.ones((3, 2) if invalid == "shape" else (2, 3), np.float32)
    if invalid == "nonfinite":
        depth[0, 0] = np.nan
    elif invalid == "negative":
        depth[0, 0] = -1
    elif invalid == "dtype":
        depth = depth.astype(np.float64)
    np.save(depth_file, depth)
    if invalid == "truncated":
        depth_file.write_bytes(b"broken")
    lines = iter([str(color).encode() + b"\n",
                  b"" if invalid == "missing" else str(depth_file).encode() + b"\n"])
    provider = object.__new__(provider_type)
    provider._views = [{"name": "top", "width": 3, "height": 2, "matrix_pose": [0]*12}]
    provider.camera_names = ("top",)
    provider._depth, provider._geom_ids = True, []
    provider._data = SimpleNamespace(geom_xpos=np.zeros((0, 3)), geom_xmat=np.zeros((0, 9)))
    provider._io = SimpleNamespace(write=lambda _: None, flush=lambda: None,
                                   readline=lambda: next(lines))
    provider._calls, provider._render_s = 0, 0
    closed = []
    provider.close = lambda: closed.append(True)
    provider._failure = lambda message: message
    if invalid:
        with pytest.raises(RuntimeError, match="Blender render failed"):
            provider.frames_for_step(0, 0)
        assert closed == [True]
    else:
        rgb, result = provider.frames_for_step(0, 0)["top"]
        assert rgb.shape == (2, 3, 3)
        np.testing.assert_array_equal(result, depth)
        assert provider._calls == 1 and not closed
