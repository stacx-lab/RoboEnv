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
            calls.append("open")

        def initial_frames(self):
            return {"top": "initial", "left": "initial"}

        def frames_for_step(self, frame, timestamp):
            calls.append(frame)
            return {"top": "next", "left": "next"}

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
    assert images.read()["top"] == images.read()["left"] == "initial"
    world.control.frame = 1
    assert images.read()["top"] == images.read()["left"] == "next"
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
