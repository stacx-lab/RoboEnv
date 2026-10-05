"""SIMPLE catalog, control and replay-boundary checks without simulator imports."""

import copy
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import DeviceError
from roboenv.provisioning.simple_assets import extract
from roboenv.tasks.resolve import resolve_task
from roboenv.tasks.simple import entries
from roboenv_runtime.adapters.simple import backend
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.control import Control
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json"


class World:
    def __init__(self):
        self.model = json.loads(FACTS.read_text())
        self.steps = 0
        self.timestamp = 0.0
        self.done = False
        self.received = []
        self.control = Control(self, self.model)

    def configure_execution_mode(self, mode):
        self.execution_mode = mode

    def read_device(self, name):
        values = {
            "positions": list(self.control.targets[name]),
            "timestamp": self.timestamp,
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
        self.timestamp = self.steps * 0.02


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


@pytest.mark.parametrize("sim_mode", ["mujoco_isaac", "mujoco"])
@pytest.mark.parametrize("existing", [None, "YES", "NO", ""])
def test_isaac_eula_default_preserves_explicit_environment(
    monkeypatch, tmp_path, sim_mode, existing
):
    for name in ("OMNI_KIT_ACCEPT_EULA", "SIMPLE_DATA_DIR", "SIMPLE_OFFLINE", "MUJOCO_GL"):
        monkeypatch.delenv(name, raising=False)
    if existing is not None:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", existing)
    env_id = "simple/G1WholebodyXMovePickTeleop-v0"
    monkeypatch.setattr(
        backend,
        "entries",
        lambda root: [{"env_id": env_id, "metadata": {"max_episode_steps": 1000}}],
    )
    monkeypatch.setattr(backend, "status", lambda root, names: {"ready": True})
    monkeypatch.setattr(backend, "verify_assets", lambda root: tmp_path)
    robot_sha256 = json.loads(FACTS.read_text())["robot_sha256"]
    monkeypatch.setattr(backend, "digest", lambda path: robot_sha256)

    world = backend.SimpleEnv(sim_mode=sim_mode)

    expected = "YES" if existing is None and sim_mode == "mujoco_isaac" else existing
    assert os.environ.get("OMNI_KIT_ACCEPT_EULA") == expected
    assert world.env is None  # Acceptance is configured before native simulator startup.


@pytest.mark.parametrize(
    "task_name,native_horizon",
    [("g1-xmove-pick", 800), ("g1-transfer-between-tables", 1200), ("g1-bend-pick", 800)],
)
def test_simple_stepped_budget_is_10000_for_each_task(
    monkeypatch, tmp_path, task_name, native_horizon
):
    task = resolve_task("simple/" + task_name).task
    env_id = task.backend_kwargs["env_id"]
    for name in ("SIMPLE_DATA_DIR", "SIMPLE_OFFLINE", "MUJOCO_GL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        backend,
        "entries",
        lambda root: [{"env_id": env_id, "metadata": {"max_episode_steps": native_horizon}}],
    )
    monkeypatch.setattr(backend, "status", lambda root, names: {"ready": True})
    monkeypatch.setattr(backend, "verify_assets", lambda root: tmp_path)
    robot_sha256 = json.loads(FACTS.read_text())["robot_sha256"]
    monkeypatch.setattr(backend, "digest", lambda path: robot_sha256)
    world = backend.SimpleEnv(env_id=env_id, sim_mode="mujoco")
    assert world.execution_mode == "realtime" and world.max_steps is None
    world.configure_execution_mode("stepped")
    assert world.max_steps == 10_000
    monkeypatch.setattr(world, "_step", lambda targets: ({}, 0.0, False, False, {}))
    world.steps = native_horizon - 1
    world.advance({})
    assert not world.done
    world.steps = 9998
    world.advance({})
    assert not world.done
    world.advance({})
    assert world.done and world.steps == 10_000 and world.timestamp == 200.0
    with pytest.raises(ValueError, match="terminated"):
        world.advance({})
    assert world.steps == 10_000
    world.configure_execution_mode("realtime")
    assert world.max_steps is None


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


def test_independent_writes_retain_goals_and_reads_do_not_step():
    world = World()
    initial = copy.deepcopy(world.control.targets)
    left = initial["left_arm"].copy()
    left[0] = 0.2
    assert world.control.write("base", base()) == {
        "accepted": True,
        # A write with no budget does not wait, so only a stepping world
        # advances its clock inside the call.
    }
    assert world.steps == 0
    with pytest.raises(ValueError):
        world.control.write("left_arm", {"target": [0]})
    assert world.control.targets["left_arm"] == initial["left_arm"]
    assert world.control.targets["base"] == [0.2, 0, 0.3, 0.74, 0.1, 0.2, 0.3]
    world.control.write("left_arm", {"target": left})
    world.read_device("left_arm")
    assert world.steps == 0
    world.control.tick()
    assert world.received[-1]["right_hand"] == initial["right_hand"]
    assert world.received[-1]["left_arm"] == left


@pytest.mark.parametrize(
    "command",
    [
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
@pytest.mark.parametrize("invalid", ["missing", "nonfinite"])
def test_base_requires_valid_complete_posture(axis, invalid):
    world = World()
    command = base()
    if invalid == "missing":
        del command["posture"][axis]
    else:
        command["posture"][axis] = float("nan")
    before = copy.deepcopy(world.control.targets)
    with pytest.raises(ValueError):
        world.control.write("base", command)
    assert world.control.targets == before
    assert not world.received


def test_waist_has_no_independent_command():
    world = World()
    before = copy.deepcopy(world.control.targets)
    with pytest.raises(ValueError):
        world.control.write("waist", {"target": [0, 0, 0]})
    with pytest.raises(ValueError):
        world.control.stop("waist")
    assert world.control.targets == before
    assert not world.received


def test_base_stop_retains_full_posture_and_arm_commands_without_stepping():
    world = World()
    world.control.write("base", base())
    before = copy.deepcopy(world.control.targets)
    world.control.stop("base")
    assert world.control.targets == {**before, "base": [0, 0, 0, *before["base"][3:]]}
    assert world.steps == 0


def test_catalog_routes_upper_body_through_measured_base_torso():
    resolved = resolve_task("simple/g1-xmove-pick")
    profiles = {
        spec.name: spec.resolved_profile(resolved.robot) for spec in resolved.bindings.devices
    }
    assert "waist" not in profiles
    assert not (ROOT / "catalog/robots/simple/simple-g1/devices/waist").exists()
    assert profiles["base"]["intrinsics"]["frames"] == {
        "torso": {"parent_frame": None, "translation_m": None, "rotation_xyzw": None}
    }
    for name in ("left_arm", "right_arm", "camera_head_left", "camera_head_right"):
        mount = profiles[name]["extrinsics"]["mount"]
        assert (mount["parent"], mount["parent_frame"]) == ("base", "torso")
    components = profiles["base"]["intrinsics"]["control"]["modes"]["locomotion"]["posture"][
        "components"
    ]
    assert [(c["axis"], c["units"]) for c in components] == [
        ("height", "m"),
        ("roll", "rad"),
        ("pitch", "rad"),
        ("yaw", "rad"),
    ]


def test_contract_bindings_have_only_basic_actions():
    resolved = resolve_task("simple/g1-xmove-pick")
    world = World()
    context = BindingContext("simple", "g1", SimpleNamespace(_env=world, control_interval_s=0.02))
    from roboenv_runtime.runtime.world import WorldSession

    owner = SimpleAdapter.__new__(SimpleAdapter)
    owner._env = world
    world.max_steps = None
    session = WorldSession(owner)
    context = BindingContext("simple", "g1", owner, world=session)
    service = default_registry().assemble(
        resolved.bindings,
        resolved.robot,
        context,
        motion_scheduler=session.schedule,
        resolve=session.resolve,
    )
    service.call({"device": "base", "action": "write", "args": {**base(), "duration_s": 0.02}})
    assert world.steps == 0
    session.resolve(session.wait(0.02))
    assert world.steps == 1
    for action in ("batch_write", "snapshot"):
        with pytest.raises(DeviceError, match="Action not declared"):
            service.call({"device": "base", "action": action, "args": {}})
    assert world.steps == 1


def test_hand_geometry_survives_live_export(tmp_path):
    from roboenv.devices.export import export_devices

    resolved = resolve_task("simple/g1-xmove-pick")
    world = World()
    service = default_registry().assemble(
        resolved.bindings,
        resolved.robot,
        BindingContext("simple", "g1", SimpleNamespace(_env=world, control_interval_s=0.02)),
    )
    descriptors = [service.describe(name) for name in ("left_hand", "right_hand")]
    export_devices(descriptors, tmp_path, {"endpoint": "http://127.0.0.1:8123"})
    for descriptor in descriptors:
        name = descriptor["name"]
        exported = yaml.safe_load((tmp_path / name / "profile.yaml").read_text())
        assert exported["intrinsics"] == world.control.profiles[name]["intrinsics"]
        assert exported["intrinsics"]["geometry"]["reference_frame"] == "grasp_reference"
        assert "thumb_tip" not in exported["intrinsics"]["frames"]
        assert exported["intrinsics"]["geometry"]["fingers"]["thumb"]["tip_frame"] == "thumb_tip"
        assert (tmp_path / name / "driver.py").is_file()


def test_binding_rejects_incorrect_fingertip_offset():
    resolved = resolve_task("simple/g1-xmove-pick")
    spec = next(spec for spec in resolved.bindings.devices if spec.name == "left_hand")
    # Resolving an episode no longer rewrites intrinsics, so state the override here.
    intrinsics = spec.resolved_profile(resolved.robot)["intrinsics"]
    intrinsics["geometry"]["fingers"]["thumb"]["tip_offset"][0] += 0.01
    bindings = replace(
        resolved.bindings,
        devices=tuple(
            replace(s, overrides={**s.overrides, "intrinsics": intrinsics})
            if s.name == "left_hand"
            else s
            for s in resolved.bindings.devices
        ),
    )
    world = World()
    with pytest.raises(ValueError, match="tip_offset"):
        default_registry().assemble(
            bindings,
            resolved.robot,
            BindingContext("simple", "g1", SimpleNamespace(_env=world, control_interval_s=0.02)),
        )
    assert world.steps == 0


def test_profile_resolution_does_not_rewrite_catalog():
    path = ROOT / "catalog/robots/simple/simple-g1/devices/base/profile.yaml"
    before = path.read_bytes()
    resolved = resolve_task("simple/g1-xmove-pick")
    spec = next(s for s in resolved.bindings.devices if s.name == "base")
    assert "execution" not in spec.resolved_profile(resolved.robot)["intrinsics"]["control"]
    assert path.read_bytes() == before


def test_score_uses_cached_verdict_without_evaluating_predicate():
    adapter = SimpleAdapter.__new__(SimpleAdapter)
    adapter._env = SimpleNamespace(completed=True, reward=0.4, arm_gravity_compensation=True)
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


def test_extracted_inventory_is_pinned(tmp_path, monkeypatch):
    import hashlib

    from roboenv.provisioning import simple_assets
    from roboenv.provisioning.simple_assets import verify_assets

    depot = tmp_path / ".cache/simple"
    depot.mkdir(parents=True)
    receipt = depot / "test.zip.json"
    receipt.write_text("[]\n")
    # The pin is the package's own file; this test's pin names one archive.
    pin = json.dumps(
        {
            "archives": [
                {
                    "file": "test.zip",
                    "inventory_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                }
            ]
        }
    ).encode()
    monkeypatch.setattr(simple_assets, "asset_pin_bytes", lambda: pin)
    assert verify_assets(tmp_path) == depot / "data"
    receipt.write_text("[ ]\n")
    with pytest.raises(ValueError, match="inventory"):
        verify_assets(tmp_path)


def test_metric_hook_sets_current_observation_commands_and_rejects_nan():
    import ast

    import numpy as np

    path = (
        ROOT / "third_party/SIMPLE/third_party/decoupled_wbc/control/policy/g1_gear_wbc_policy.py"
    )
    if not path.exists():
        pytest.skip("requires selected pinned WBC checkout")
    method = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "set_metric_command"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    policy = SimpleNamespace(cmd=np.zeros(3))
    namespace["set_metric_command"](policy, [0.2, -0.1, 0.3], 0.65, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(policy.cmd, [0.2, -0.1, 0.3])
    assert (policy.height_cmd, policy.roll_cmd, policy.pitch_cmd, policy.yaw_cmd) == (
        0.65,
        0.1,
        0.2,
        0.3,
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


def test_generator_reproduces_catalog_and_facts(tmp_path):
    if not (ROOT / ".cache/simple/data/robots/g1_sonic/g1_29dof_with_hand.xml").is_file():
        pytest.skip("requires prepared pinned robot geometry")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_simple_config.py"),
            "--out",
            str(tmp_path / "catalog"),
            "--facts-out",
            str(tmp_path / "g1.json"),
            "--bindings-out",
            str(tmp_path / "bindings"),
        ],
        check=True,
        cwd=ROOT,
    )
    assert (tmp_path / "g1.json").read_bytes() == FACTS.read_bytes()
    for directory in ("robots/simple", "tasks/simple"):
        original = ROOT / "catalog" / directory
        copied = tmp_path / "catalog" / directory
        files = {p.relative_to(original) for p in original.rglob("*") if p.is_file()}
        assert files == {p.relative_to(copied) for p in copied.rglob("*") if p.is_file()}
        for relative in files:
            assert (original / relative).read_bytes() == (copied / relative).read_bytes()

    assert not (tmp_path / "catalog/bindings").exists()
    assert (tmp_path / "bindings/simple/simple-g1.yaml").read_bytes() == (
        ROOT / "src/roboenv/devices/bindings/defaults/simple/simple-g1.yaml"
    ).read_bytes()


@pytest.mark.parametrize(
    "invalid", ["dimensions", "intrinsics", "half_pixel", "translation", "rotation"]
)
def test_isaac_camera_calibration_checks_actual_optical_mount(invalid):
    import numpy as np
    from roboenv_runtime.adapters.simple.cameras import validate_isaac
    from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

    facts = json.loads(FACTS.read_text())
    parent = np.array([0.3, -0.2, 0.7, 0, 0, np.sin(0.2), np.cos(0.2)])
    cameras = {}
    for spec in facts["cameras"].values():
        mount = spec["mount"]
        optical = matrix_pose(
            matrix(parent) @ matrix(mount["translation_m"] + mount["rotation_xyzw"])
        )
        projection = spec["projection"]
        calibration = [
            [projection["fx"], 0, projection["cx"] + 0.5],
            [0, projection["fy"], projection["cy"] + 0.5],
            [0, 0, 1],
        ]
        cameras[spec["native"]] = SimpleNamespace(
            get_resolution=lambda: (640, 360),
            get_intrinsics_matrix=lambda value=calibration: np.array(value),
            get_world_pose=lambda camera_axes, value=optical: (
                np.array(value[:3]),
                np.array([value[6], *value[3:6]]),
            ),
        )
    renderer = SimpleNamespace(robot=SimpleNamespace(prim_path="/Robot/G1"), cameras=cameras)
    world = SimpleNamespace(
        model=facts, env=SimpleNamespace(isaac=renderer), _pose=lambda name: parent
    )
    validate_isaac(world)
    camera = cameras[facts["cameras"]["camera_head_right"]["native"]]
    if invalid == "dimensions":
        camera.get_resolution = lambda: (360, 640)
    elif invalid in {"intrinsics", "half_pixel"}:
        value = camera.get_intrinsics_matrix()
        value[0, 0] += 1 if invalid == "intrinsics" else 0
        value[:2, 2] -= 0.5 if invalid == "half_pixel" else 0
        camera.get_intrinsics_matrix = lambda: value
    else:
        position, quaternion = camera.get_world_pose(camera_axes="ros")
        if invalid == "translation":
            position[0] += 0.01
        else:
            quaternion = np.array([1, 0, 0, 0])
        camera.get_world_pose = lambda camera_axes: (position, quaternion)
    with pytest.raises(ValueError, match="differ"):
        validate_isaac(world)


def test_joint_stop_keeps_measured_soft_limit_overshoot_without_stepping():
    world = World()
    original = copy.deepcopy(world.control.targets)
    measured = list(original["left_arm"])
    measured[0] = world.model["parts"]["left_arm"]["joints"][0]["limits"]["max"] + 0.01
    world.read_device = lambda name: {"positions": measured}
    world.control.stop("left_arm")
    assert world.steps == 0
    assert world.control.targets == {**original, "left_arm": measured}


def test_native_targets_are_not_rejected_by_invented_bounds():
    world = World()
    command = base(vx=5, wz=10)
    command["posture"] = dict.fromkeys(command["posture"], 100)
    world.control.write("base", command)
    assert world.control.targets["base"] == [5, 0, 10, 100, 100, 100, 100]
    target = [100] * len(world.control.targets["left_arm"])
    world.control.write("left_arm", {"target": target})
    world.control.tick()
    assert world.received[-1]["left_arm"] == target
