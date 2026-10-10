"""Camera reads must await delivered frames, including under delayed rendering."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices
from roboenv_runtime.adapters.behavior.rendering import (
    ZERO_DELAY_SETTINGS,
    capture_launch_settings,
    render_for_capture,
)
from roboenv_runtime.spectator.isaac import Engine


@pytest.fixture
def pipeline(monkeypatch):
    state = {"renders": 0, "stamp": 1, "pixels": 0, "current": 2,
             "deliver": True, "present": True, "valid": True}

    def render():
        state["renders"] += 1
        # The GPU produces a delivery only on every third render attempt.
        if state["deliver"] and state["renders"] % 3 == 0:
            state["pixels"] = state["current"]
            state["stamp"] += 1

    def attribute(name):
        return SimpleNamespace(get=lambda: (
            state["stamp"] if name.endswith("Numerator") else int(state["valid"])
        ))

    core = ModuleType("omni.graph.core")
    core.get_node_by_path = lambda _: (
        SimpleNamespace(get_attribute=attribute) if state["present"] else None
    )
    graph = ModuleType("omni.graph")
    graph.core = core
    omni = ModuleType("omni")
    omni.graph = graph
    for name, module in (("omni", omni), ("omni.graph", graph), ("omni.graph.core", core)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(
        sim=SimpleNamespace(render=render),
    ))
    settings = ModuleType("carb.settings")
    settings.get_settings = lambda: SimpleNamespace(get=ZERO_DELAY_SETTINGS.get)
    carb = ModuleType("carb")
    carb.settings = settings
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", settings)
    return state


@pytest.mark.parametrize("kind", ["agent", "spectator"])
def test_first_read_after_motion_waits_for_delivery(pipeline, kind):
    state = pipeline
    world = BehaviorDevices.__new__(BehaviorDevices)

    def observation():
        return {"rgb": np.full((2, 3, 4), state["pixels"], np.uint8),
                "depth_linear": np.full((2, 3), state["pixels"], np.float32)}, {}

    world.robot = SimpleNamespace(sensors={"native": SimpleNamespace(get_obs=observation)})
    world.cameras = {"camera_0": "native"}
    engine = Engine.__new__(Engine)
    engine.backend = "behavior"
    for current in (2, 3):
        state["current"] = current
        if kind == "agent":
            color, depth = world.device_image("camera_0")
            assert np.all(color == current) and np.all(depth == current)
        else:
            engine.render()
            assert state["pixels"] == current
    assert state["renders"] == 6


@pytest.mark.parametrize("failure", ["present", "deliver", "valid"])
def test_missing_or_stalled_delivery_fails_explicitly(pipeline, failure):
    pipeline[failure] = False
    with pytest.raises(RuntimeError, match="BEHAVIOR camera"):
        render_for_capture()
    assert pipeline["renders"] == (0 if failure == "present" else 16)


def test_capture_rejects_missing_startup_settings(pipeline, monkeypatch):
    monkeypatch.setattr(sys.modules["carb.settings"], "get_settings",
                        lambda: SimpleNamespace(get=lambda _: None))
    with pytest.raises(RuntimeError, match="zero-delay startup"):
        render_for_capture()
    assert pipeline["renders"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_settings_are_scoped_and_preserve_other_options(monkeypatch, fail):
    original = {"active_gpu": 3, "extra_args": ["--existing=true"]}
    launcher = SimpleNamespace(DEFAULT_LAUNCHER_CONFIG=original)
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=launcher))
    try:
        with capture_launch_settings():
            config = launcher.DEFAULT_LAUNCHER_CONFIG
            assert config["active_gpu"] == 3
            assert config["extra_args"][0] == "--existing=true"
            assert len(config["extra_args"]) == 4
            assert original == {"active_gpu": 3, "extra_args": ["--existing=true"]}
            if fail:
                raise RuntimeError("boot failed")
    except RuntimeError:
        assert fail
    assert launcher.DEFAULT_LAUNCHER_CONFIG is original
