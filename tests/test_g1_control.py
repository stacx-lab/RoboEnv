"""CPU checks for six-device routing; native physics is checked separately."""

import copy
import json

import pytest
from roboenv_runtime.adapters.isaaclab.g1_assets import digest, load_bundle
from roboenv_runtime.adapters.isaaclab.g1_control import Control
from roboenv_runtime.adapters.isaaclab.g1_profiles import NAMES


def model():
    joint = {
        "name": "q",
        "type": "revolute",
        "limits": {"min": -1, "max": 1},
        "max_velocity": None,
        "max_effort": None,
    }
    return {
        "cameras": {},
        "control": {
            "mode": "motion",
            "body_frame": "pelvis",
            "modes": {
                "locomotion": {
                    "velocity": {
                        "frame": "body",
                        "components": [
                            {
                                "axis": axis,
                                "units": "rad/s" if axis == "wz" else "m/s",
                                "bounds": {"min": -0.5, "max": 0.5},
                            }
                            for axis in ("vx", "vy", "wz")
                        ],
                    },
                    "posture": {
                        "reference": "ground",
                        "components": [
                            {
                                "axis": "height",
                                "units": "m",
                                "bounds": {"min": 0.6, "max": 0.8},
                                "default": 0.75,
                            }
                        ],
                    },
                }
            },
            "execution": {"mode": "stepped", "interval_s": 0.02},
        },
        "parts": {
            name: {
                "joints": [{**joint, "name": name + "_q"}],
                "home": {name + "_q": 0},
                "chain": [],
                "mount": [0.04, 0, 0, 0, 0, 0, 1],
            }
            for name in NAMES[1:]
        },
    }


class Native:
    def __init__(self):
        self.sample_time_s = 0
        self.received = []
        self.done = False
        self.steps = 0
        self.max_steps = 10
        self.realtime = False
        self.control_interval = 0.02
        self.closed = False

    def require_active(self):
        if self.done or self.steps >= self.max_steps:
            raise ValueError("G1 episode has terminated")

    def read_device(self, name):
        return {"sample_time_s": self.sample_time_s, "positions": [0.2], "faults": []}

    def advance(self, targets):
        self.require_active()
        self.received.append(copy.deepcopy(targets))
        self.sample_time_s += 0.02
        self.steps += 1
        self.done = self.steps >= self.max_steps

    def close(self):
        self.closed = True

    def reset(self, seed):
        self.steps = 0
        self.sample_time_s = 0
        self.done = False
        self.control = Control(self, model(), realtime=self.realtime)


def test_policy_cannot_advertise_untracked_posture_or_other_frames():
    for mutation in ("pitch", "frame", "default"):
        facts = model()
        mode = facts["control"]["modes"]["locomotion"]
        if mutation == "pitch":
            mode["posture"]["components"].append(
                {"axis": "pitch", "units": "rad", "bounds": {"min": -0.2, "max": 0.2}, "default": 0}
            )
        if mutation == "frame":
            mode["velocity"]["frame"] = "heading"
        if mutation == "default":
            mode["posture"]["components"][0]["default"] = 2
        native = Native()
        with pytest.raises(ValueError):
            Control(native, facts)
        assert native.received == []


@pytest.mark.parametrize("realtime", [False, True])
def test_batch_validates_before_any_effect_and_preserves_other_targets(realtime):
    native = Native()
    control = Control(native, model(), realtime=realtime)
    valid = {"device": "left_arm", "args": {"target": [0.4]}}
    for invalid in (
        {"device": "right_arm", "args": {"target": [2]}},
        {"device": "missing", "args": {"target": [0]}},
        {"device": "base", "args": {"target": [0, 0, 0, 0.7]}},
        {
            "device": "base",
            "args": {
                "mode": "locomotion",
                "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": 2, "pitch": 0},
            },
        },
        {
            "device": "base",
            "args": {
                "mode": "stand",
                "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": 0.7},
            },
        },
        valid,
    ):
        with pytest.raises(ValueError):
            control.batch_write([valid, invalid])
    assert native.received == []
    receipt = control.batch_write([valid])
    assert receipt["sequence"] == 1
    if realtime:
        assert native.received == []
        assert receipt["sample_time_s"] == 0
        control.tick()
    assert native.sample_time_s == 0.02
    assert native.received[0]["right_arm"] == [0]
    assert control.snapshot(["left_arm", "right_arm"])["sequence"] == 1
    assert native.sample_time_s == 0.02
    control.write(
        "base",
        {
            "mode": "locomotion",
            "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
            "posture": {"height": 0.7, "roll": 0.1, "pitch": -0.2, "yaw": 0.3},
        },
    )
    if realtime:
        control.tick()
    control.stop("base")
    assert control.targets["base"] == [0, 0, 0, 0.7]
    assert native.sample_time_s == 0.04


@pytest.mark.parametrize("timeout", [None, 0.5])
def test_realtime_watchdog_retains_posture_and_only_base_writes_refresh_it(monkeypatch, timeout):
    from roboenv_runtime.adapters.isaaclab import g1_control

    now = [10.0]
    monkeypatch.setattr(g1_control.time, "monotonic", lambda: now[0])
    native = Native()
    data = model()
    control = Control(native, data, realtime=True, command_timeout_s=timeout)
    target = {
        "mode": "locomotion",
        "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
        "posture": {"height": 0.7},
    }
    control.write("base", target)
    first_deadline = control.base_deadline
    assert data["control"]["execution"] == {"mode": "stepped", "interval_s": 0.02}
    assert control.profiles["base"]["intrinsics"]["control"]["execution"] == {
        "mode": "realtime",
        "command_timeout_s": timeout,
    }
    now[0] = 10.2
    control.snapshot(["base"])
    control.write("left_arm", {"target": [0.4]})
    with pytest.raises(ValueError):
        control.write("base", {**target, "posture": {"height": 2}})
    assert control.base_deadline == first_deadline
    assert native.received == []
    control.tick()
    assert native.received[-1]["base"] == [0.1, 0, 0, 0.7]
    assert native.received[-1]["left_arm"] == [0.4]
    now[0] = 10.3
    control.write("base", target)
    assert control.base_deadline == (10.8 if timeout else None)
    now[0] = 10.6
    control.tick()
    assert native.received[-1]["base"][0] == 0.1
    now[0] = 10.8
    control.tick()
    assert native.received[-1]["base"] == ([0, 0, 0, 0.7] if timeout else [0.1, 0, 0, 0.7])
    assert native.received[-1]["left_arm"] == [0.4]
    # Stops replace goals but do not step. Upper body holds measured positions.
    before = native.steps
    control.stop("base")
    control.stop("left_arm")
    assert native.steps == before and control.base_deadline is None
    control.tick()
    assert native.received[-1]["base"] == [0, 0, 0, 0.7]
    assert native.received[-1]["left_arm"] == [0.2]


@pytest.mark.parametrize(
    "options",
    [
        {"realtime": "true"},
        {"realtime": 1},
        *(
            {"command_timeout_s": value}
            for value in (0, -1, True, "0.5", float("nan"), float("inf"))
        ),
    ],
)
def test_invalid_clock_settings_are_rejected_before_effects(options):
    native = Native()
    with pytest.raises(ValueError):
        Control(native, model(), **options)
    assert native.received == []


@pytest.mark.parametrize("early_terminal", [False, True])
def test_g1_realtime_uses_owner_clock_and_preserves_terminal_state(monkeypatch, early_terminal):
    from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
    from roboenv_runtime.runtime import world as world_module

    now = [0.0]
    monkeypatch.setattr(world_module.time, "monotonic", lambda: now[0])
    native = Native()
    native.realtime = True
    native.max_steps = 2
    native.control = Control(native, model(), realtime=True)
    adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
    adapter._env = native
    world = world_module.WorldSession(adapter)
    world.start()
    assert world.main_thread_clock and world._thread is None
    try:
        before = native.control.snapshot(["base", "left_arm"])
        native.control.write("left_arm", {"target": [0.4]})
        assert native.steps == 0
        now[0] = 0.021
        world.poll()
        assert adapter.budget().used == 1
        after = native.control.snapshot(["base", "left_arm"])
        assert after["sample_time_s"] > before["sample_time_s"]
        assert native.received[-1]["left_arm"] == [0.4]
        if early_terminal:
            native.done = True
        for _ in range(4):
            now[0] += 0.03
            world.poll()
        assert native.steps == (1 if early_terminal else 2)
        assert world.failure is None
        retained = copy.deepcopy(native.control.targets)
        with pytest.raises(ValueError, match="terminated"):
            native.control.write("left_arm", {"target": [0.5]})
        assert native.control.targets == retained
        assert adapter.budget().max == 2  # Preserve the original horizon.
        world.reset(3)
        assert not world._halt.is_set() and native.steps == 0
        now[0] += 0.03
        world.poll()
        assert native.steps == 1 and world.failure is None
    finally:
        world.close()
    assert native.closed


def test_asset_manifest_detects_changes_and_escape(tmp_path):
    asset = tmp_path / "asset"
    asset.write_bytes(b"fixture")
    value = {
        "schema": "roboenv/g1-assets/1",
        "roles": {key: "asset" for key in ("robot", "table", "object", "policy")},
        "files": {"asset": digest(asset)},
    }
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps(value))
    assert load_bundle(manifest)[1]["robot"] == str(asset)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_bundle(manifest)


def test_live_export_retains_execution_group(tmp_path):
    from roboenv.devices.export import export_devices
    from roboenv.devices.model import Action, BoundDevice, DeviceDefinition

    device = BoundDevice(
        DeviceDefinition("base", "base"),
        {"read": Action(lambda args, context: {})},
        implementation="fixture",
        execution_group="base",
    )
    export_devices([device.describe()], tmp_path, {"endpoint": "http://127.0.0.1:8123"})
    import yaml

    profile = yaml.safe_load((tmp_path / "base/profile.yaml").read_text())
    assert profile["access"]["driver"]["execution_group"] == "base"
    assert "kinex.base" in (tmp_path / "base/driver.py").read_text()


@pytest.mark.parametrize("realtime", [False, True])
def test_default_catalog_has_complete_read_only_camera_bindings(tmp_path, realtime):
    from pathlib import Path
    from types import SimpleNamespace

    from roboenv.devices.bindings.registry import BindingContext
    from roboenv.devices.model import DeviceError
    from roboenv.tasks.resolve import resolve_task
    from roboenv_runtime.bindings import default_registry

    root = Path(__file__).resolve().parents[1]
    task = resolve_task("isaaclab/g1-pick-place", project=root, realtime=realtime)
    assert {spec.name for spec in task.bindings.devices} == {
        *NAMES,
        "world",
        "camera_torso",
        "camera_scene",
    }
    assert not Path(task.task.backend_kwargs["asset_manifest"]).is_absolute()
    assert "model_path" not in task.task.backend_kwargs
    assert {path.name for path in task.robot.iterdir()} == {"workspace.yaml", "devices"}
    from roboenv.devices.bindings.spec import binding_path

    assert binding_path(task.robot) == root / "catalog/bindings/isaaclab/isaaclab-g1.yaml"
    data = json.loads((root / "runtime/src/roboenv_runtime/adapters/isaaclab/g1.json").read_text())
    assert digest(root / "src/roboenv/assets/isaaclab-g1.json") == data["asset_manifest_sha256"]
    world = Native()
    world.model = data
    world.control = Control(
        world,
        data,
        realtime=realtime,
        command_timeout_s=task.task.backend_kwargs.get("command_timeout_s", 0.5),
    )
    world.read_camera = lambda name, directory: {"sample_time_s": world.sample_time_s}
    context = BindingContext("isaaclab", "g1", SimpleNamespace(_env=world, run_dir=tmp_path))
    service = default_registry().assemble(task.bindings, task.robot, context)
    try:
        for name in ("camera_torso", "camera_scene"):
            assert (
                service.call({"device": name, "action": "read", "args": {}})["data"][
                    "sample_time_s"
                ]
                == 0
            )
            for action in ("write", "stop"):
                with pytest.raises(DeviceError):
                    service.call({"device": name, "action": action, "args": {}})
        assert world.received == []
        expected_execution = world.control.profiles["base"]["intrinsics"]["control"]["execution"]
        assert service.describe("base")["intrinsics"]["control"]["execution"] == expected_execution
        import yaml
        from roboenv.devices.export import export_devices

        export_devices(
            service.inventory()["devices"],
            tmp_path / "devices",
            {"endpoint": "http://127.0.0.1:8123"},
        )
        exported = yaml.safe_load((tmp_path / "devices/base/profile.yaml").read_text())
        assert exported["intrinsics"]["control"]["execution"] == expected_execution
        service.call(
            {
                "device": "base",
                "action": "write",
                "args": {
                    "mode": "locomotion",
                    "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
                    "posture": {"height": 0.75},
                },
            }
        )
        assert world.steps == (0 if realtime else 1)
        before = world.steps
        service.call({"device": "base", "action": "stop", "args": {}})
        assert service.call({"device": "base", "action": "unload", "args": {}})["data"]["unloaded"]
        assert world.control.targets["base"] == [0, 0, 0, 0.75]
        assert world.steps == before and world.control.base_deadline is None
    finally:
        service.close()


def test_g1_reset_clears_pending_motion_and_watchdog():
    from types import SimpleNamespace

    from roboenv_runtime.adapters.isaaclab.g1 import G1Env

    world = G1Env.__new__(G1Env)
    world.model = model()
    world.realtime = True
    world.command_timeout_s = 0.5
    world.control_interval = 0.02
    world.env = SimpleNamespace(allow_reset=False, max_episode_length=10, reset=lambda **_: None)
    world.read_device = lambda name: {"positions": [0.2]}
    world.steps = 10
    world.done = world.completed = True
    world.images = {"old": {}}
    world.control = Control(world, world.model, realtime=True)
    world.control.targets["base"] = [0.1, 0, 0, 0.7]
    world.control.base_deadline = 999
    world.reset(7)
    assert world.steps == 0 and not world.done and not world.completed
    assert world.images == {} and world.env.allow_reset is False
    assert world.control.realtime and world.control.base_deadline is None
    assert world.control.targets["base"] == [0, 0, 0, 0.75]
    assert world.control.targets["left_arm"] == [0.2]
