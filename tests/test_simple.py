"""SIMPLE catalog, control and replay-boundary checks without simulator imports."""

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.provisioning.simple_assets import extract
from roboenv.tasks.resolve import resolve_task
from roboenv.tasks.simple import entries
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.control import Control
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json"


class World:
    def __init__(self, realtime=False):
        self.model = json.loads(FACTS.read_text())
        self.steps = 0
        self.sample_time_s = 0.0
        self.done = False
        self.received = []
        self.control = Control(self, self.model, realtime=realtime)

    def read_device(self, name):
        values = {
            "positions": list(self.control.targets[name]),
            "sample_time_s": self.sample_time_s,
            "faults": [],
        }
        return values

    def require_active(self):
        if self.done:
            raise ValueError("terminated")

    def advance(self, targets):
        self.require_active()
        self.received.append(copy.deepcopy(targets))
        self.steps += 1
        self.sample_time_s = self.steps * 0.02


def base(vx=0.2, wz=0.3):
    return {
        "mode": "locomotion",
        "velocity": {"vx": vx, "vy": 0, "wz": wz},
        "posture": {"height": 0.74, "roll": 0.1, "pitch": 0.2, "yaw": 0.3},
    }


def test_imports_and_catalog_do_not_boot_simulator():
    code = """
import sys
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.simple import SimpleAdapter
resolve_task('simple/g1-xmove-pick')
assert SimpleAdapter.name == 'simple'
assert not {'simple', 'torch', 'mujoco', 'isaacsim', 'pinocchio'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_three_examples_and_twelve_native_teleop_registrations():
    source = ROOT / "third_party/SIMPLE"
    if not (source / "src/simple/envs/__init__.py").exists():
        pytest.skip("upstream discovery tested with selected checkout")
    found = list(entries(source))
    assert len(found) == 12
    assert all(item["env_id"].endswith("Teleop-v0") for item in found)
    for name in ("g1-xmove-pick", "g1-transfer-between-tables", "g1-bend-pick"):
        task = resolve_task("simple/" + name)
        assert task.task.budget_mode == "practice"
        assert len(task.bindings.devices) == 8
        assert task.task.backend_kwargs["sim_mode"] == "mujoco_isaac"


@pytest.mark.parametrize("realtime", [False, True])
def test_atomic_batch_retains_goals_and_reads_do_not_step(realtime):
    world = World(realtime)
    left = world.control.targets["left_arm"].copy()
    left[0] = 0.2
    before = copy.deepcopy(world.control.targets)
    with pytest.raises(ValueError):
        world.control.batch_write(
            [{"device": "base", "args": base()}, {"device": "left_arm", "args": {"target": [0]}}]
        )
    assert world.control.targets == before
    assert world.steps == 0
    world.control.batch_write(
        [{"device": "base", "args": base()}, {"device": "left_arm", "args": {"target": left}}]
    )
    assert world.steps == (0 if realtime else 1)
    world.control.snapshot(["base", "left_arm"])
    assert world.steps == (0 if realtime else 1)
    if realtime:
        world.control.tick()
    assert world.received[0]["base"] == [0.2, 0, 0.3, 0.74, 0.1, 0.2, 0.3]
    assert world.received[0]["right_arm"] == before["right_arm"]


@pytest.mark.parametrize(
    "command",
    [
        base(vx=0.51),
        base(wz=1.01),
        base(vx=float("nan")),
        {"mode": "locomotion", "target": [0, 0, 0]},
    ],
)
def test_invalid_metric_commands_fail_before_effect(command):
    world = World()
    with pytest.raises(ValueError):
        world.control.write("base", command)
    assert not world.received


@pytest.mark.parametrize("axis", ["height", "roll", "pitch", "yaw"])
@pytest.mark.parametrize("invalid", ["missing", "out-of-range", "nonfinite"])
def test_base_requires_valid_complete_posture(axis, invalid):
    world = World()
    command = base()
    if invalid == "missing":
        del command["posture"][axis]
    else:
        command["posture"][axis] = 100 if invalid == "out-of-range" else float("nan")
    before = copy.deepcopy(world.control.targets)
    with pytest.raises(ValueError):
        world.control.batch_write([
            {"device": "left_arm", "args": {"target": before["left_arm"]}},
            {"device": "base", "args": command},
        ])
    assert world.control.targets == before
    assert not world.received


def test_waist_has_no_independent_command_or_snapshot():
    world = World()
    before = copy.deepcopy(world.control.targets)
    with pytest.raises(ValueError):
        world.control.write("waist", {"target": [0, 0, 0]})
    with pytest.raises(ValueError):
        world.control.stop("waist")
    with pytest.raises(ValueError):
        world.control.snapshot(["base", "waist"])
    assert world.control.targets == before
    assert not world.received


def test_base_stop_retains_full_posture_and_arm_commands_without_stepping():
    world = World()
    world.control.write("base", base())
    before = copy.deepcopy(world.control.targets)
    world.control.stop("base")
    assert world.control.targets == {**before, "base": [0, 0, 0, *before["base"][3:]]}
    assert world.steps == 1


def test_catalog_routes_upper_body_through_measured_base_torso():
    resolved = resolve_task("simple/g1-xmove-pick")
    profiles = {spec.name: spec.resolved_profile(resolved.robot)
                for spec in resolved.bindings.devices}
    assert "waist" not in profiles
    assert not (ROOT / "catalog/robots/simple/simple-g1/devices/waist").exists()
    assert profiles["base"]["extrinsics"]["frames"] == ["torso"]
    for name in ("left_arm", "right_arm", "camera_head_left", "camera_head_right"):
        mount = profiles[name]["extrinsics"]["mount"]
        assert (mount["parent"], mount["parent_frame"]) == ("base", "torso")
    components = profiles["base"]["intrinsics"]["control"]["modes"]["locomotion"][
        "posture"]["components"]
    assert [(c["axis"], c["units"]) for c in components] == [
        ("height", "m"), ("roll", "rad"), ("pitch", "rad"), ("yaw", "rad"),
    ]


def test_realtime_timeout_retains_full_posture_and_upper_targets(monkeypatch):
    import roboenv_runtime.adapters.humanoid_control as control_module

    now = [0.0]
    monkeypatch.setattr(control_module.time, "monotonic", lambda: now[0])
    world = World(True)
    world.control.write("base", base())
    targets = copy.deepcopy(world.control.targets)
    now[0] = 0.6
    world.control.tick()
    assert world.received[-1] == {**targets, "base": [0, 0, 0, 0.74, 0.1, 0.2, 0.3]}
    world.done = True
    world.control.tick()
    assert world.steps == 1
    with pytest.raises(ValueError, match="terminated"):
        world.control.write("base", base())


def test_binding_plan_uses_shared_execution_group():
    resolved = resolve_task("simple/g1-xmove-pick")
    world = World()
    context = BindingContext("simple", "g1", SimpleNamespace(_env=world))
    service = default_registry().assemble(resolved.bindings, resolved.robot, context)
    service.call(
        {
            "device": "base",
            "action": "batch_write",
            "args": {"commands": [{"device": "base", "args": base()}]},
        }
    )
    assert world.steps == 1
    assert service.call(
        {"device": "base", "action": "snapshot", "args": {"devices": ["base", "right_arm"]}}
    )
    assert world.steps == 1


def test_realtime_profile_resolution_does_not_rewrite_catalog():
    path = ROOT / "catalog/robots/simple/simple-g1/devices/base/profile.yaml"
    before = path.read_bytes()
    resolved = resolve_task("simple/g1-xmove-pick", realtime=True)
    spec = next(s for s in resolved.bindings.devices if s.name == "base")
    execution = spec.resolved_profile(resolved.robot)["intrinsics"]["control"]["execution"]
    assert execution == {"mode": "realtime", "command_timeout_s": 0.5}
    assert path.read_bytes() == before


def test_score_uses_cached_verdict_without_evaluating_predicate():
    adapter = SimpleAdapter.__new__(SimpleAdapter)
    adapter._env = SimpleNamespace(completed=True, reward=0.4)
    assert adapter.score().completed is True
    assert adapter.score().raw_score == 0.4


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_archives_reject_traversal_before_writing(tmp_path, name):
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("good", "data")
        source.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "out/good").exists()


def test_extracted_inventory_is_pinned(tmp_path):
    import hashlib

    from roboenv.provisioning.simple_assets import verify_assets

    depot = tmp_path / ".cache/simple"
    depot.mkdir(parents=True)
    receipt = depot / "test.zip.json"
    receipt.write_text("[]\n")
    pin = tmp_path / "src/roboenv/assets/simple-g1.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"archives": [{
        "file": "test.zip", "inventory_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }]}))
    assert verify_assets(tmp_path) == depot / "data"
    receipt.write_text("[ ]\n")
    with pytest.raises(ValueError, match="inventory"):
        verify_assets(tmp_path)


def test_base_catalog_declares_atomic_actions():
    resolved = resolve_task("simple/g1-xmove-pick")
    base = next(spec for spec in resolved.bindings.devices if spec.name == "base")
    actions = base.resolved_profile(resolved.robot)["access"]["driver"]["actions"]
    assert {"batch_write", "snapshot"} <= set(actions)


def test_metric_hook_sets_current_observation_commands_and_rejects_nan():
    import ast

    import numpy as np

    path = (ROOT / "third_party/SIMPLE/third_party/decoupled_wbc/control/policy/"
            "g1_gear_wbc_policy.py")
    if not path.exists():
        pytest.skip("requires selected pinned WBC checkout")
    method = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "set_metric_command")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    policy = SimpleNamespace(cmd=np.zeros(3))
    namespace["set_metric_command"](policy, [0.2, -0.1, 0.3], 0.65, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(policy.cmd, [0.2, -0.1, 0.3])
    assert (policy.height_cmd, policy.roll_cmd, policy.pitch_cmd, policy.yaw_cmd) == (
        0.65, 0.1, 0.2, 0.3,
    )
    with pytest.raises(ValueError):
        namespace["set_metric_command"](policy, [0, 0, float("nan")], 0.65, [0, 0, 0])
    np.testing.assert_array_equal(policy.cmd, [0.2, -0.1, 0.3])


@pytest.mark.parametrize("maximum, done", [(2, True), (None, False)])
def test_native_advance_caches_verdict_and_applies_only_finite_horizon(maximum, done):
    from roboenv_runtime.adapters.simple.backend import SimpleEnv

    world = SimpleEnv.__new__(SimpleEnv)
    world.steps, world.max_steps, world.done = 1, maximum, False
    world.frames, world.images = [], {"stale": True}
    calls = []

    def step(targets):
        calls.append(targets)
        return {}, 0.25, False, False, {}

    world._step = step
    world.advance({"base": [0, 0, 0, 0.74, 0.1, 0.2, 0.3]})
    assert world.steps == 2 and world.done is done
    assert world.reward == 0.25 and world.completed is False
    assert len(calls) == 1 and not world.images
    if done:
        with pytest.raises(ValueError, match="terminated"):
            world.advance({})
        assert len(calls) == 1
