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
        self.control = Control(self, model())


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


def test_write_validates_before_effects_and_preserves_other_targets():
    native = Native()
    control = Control(native, model())
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
    ):
        with pytest.raises(ValueError):
            control.write(invalid["device"], invalid["args"])
    assert native.received == []
    receipt = control.write(valid["device"], valid["args"])
    assert receipt["accepted"] is True and receipt["elapsed_s"] >= 0
    assert native.received == []
    assert native.sample_time_s == 0
    control.tick()
    assert native.sample_time_s == 0.02
    assert native.received[0]["right_arm"] == [0]
    native.read_device("left_arm")
    native.read_device("right_arm")
    assert native.sample_time_s == 0.02
    control.write(
        "base",
        {
            "mode": "locomotion",
            "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
            "posture": {"height": 0.7},
        },
    )
    control.tick()
    control.stop("base")
    assert control.targets["base"] == [0, 0, 0, 0.7]
    assert native.sample_time_s == 0.04


def test_terminal_control_preserves_references_until_reset():
    native = Native()
    native.control = Control(native, model())
    native.control.write("left_arm", {"target": [0.4]})
    native.control.tick()
    native.done = True
    retained = copy.deepcopy(native.control.targets)
    with pytest.raises(ValueError, match="terminated"):
        native.control.write("left_arm", {"target": [0.5]})
    assert native.control.targets == retained and native.steps == 1
    native.reset(3)
    assert native.steps == 0


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


def test_live_export_uses_independent_device_driver(tmp_path):
    from roboenv.devices.export import export_devices
    from roboenv.devices.model import Action, BoundDevice, DeviceDefinition

    device = BoundDevice(
        DeviceDefinition("base", "velocity"),
        {"read": Action(lambda args, context: {})},
        implementation="fixture",
    )
    export_devices([device.describe()], tmp_path, {"endpoint": "http://127.0.0.1:8123"})
    import yaml

    profile = yaml.safe_load((tmp_path / "base/profile.yaml").read_text())
    assert "execution_group" not in profile["access"]["driver"]
    assert "kinex.velocity" in (tmp_path / "base/driver.py").read_text()


def test_default_catalog_has_complete_read_only_camera_bindings(tmp_path):
    from pathlib import Path

    from roboenv.devices.bindings.registry import BindingContext
    from roboenv.devices.model import DeviceError
    from roboenv.tasks.resolve import resolve_task
    from roboenv_runtime.bindings import default_registry

    root = Path(__file__).resolve().parents[1]
    task = resolve_task("isaaclab/g1-pick-place", project=root)
    assert {spec.name for spec in task.bindings.devices} == {
        *NAMES,
        "world",
        "camera_torso",
        "camera_scene",
    }
    assert not Path(task.task.backend_kwargs["asset_manifest"]).is_absolute()
    assert "model_path" not in task.task.backend_kwargs
    assert {path.name for path in task.robot.iterdir()} == {"workspace.yaml", "devices"}
    from roboenv.devices.bindings.spec import binding_resource

    assert binding_resource(task.robot, backend="isaaclab").is_file()
    data = json.loads((root / "runtime/src/roboenv_runtime/adapters/isaaclab/g1.json").read_text())
    assert digest(root / "src/roboenv/assets/isaaclab-g1.json") == data["asset_manifest_sha256"]
    world = Native()
    world.model = data
    world.control = Control(world, data)
    world.read_camera = lambda name, directory: {"sample_time_s": world.sample_time_s}
    from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
    from roboenv_runtime.runtime.world import WorldSession

    adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
    adapter._env = world
    adapter.run_dir = tmp_path
    session = WorldSession(adapter)
    context = BindingContext("isaaclab", "g1", adapter)
    service = default_registry().assemble(
        task.bindings,
        task.robot,
        context,
        motion_scheduler=session.schedule,
        resolve=session.resolve,
    )
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
        assert "execution" not in service.describe("base")["intrinsics"]["control"]
        import yaml
        from roboenv.devices.export import export_devices

        export_devices(
            service.inventory()["devices"],
            tmp_path / "devices",
            {"endpoint": "http://127.0.0.1:8123"},
        )
        exported = yaml.safe_load((tmp_path / "devices/base/profile.yaml").read_text())
        assert "execution" not in exported["intrinsics"]["control"]
        session.resolve(
            service.call(
                {
                    "device": "base",
                    "action": "write",
                    "args": {
                        "mode": "locomotion",
                        "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
                        "posture": {"height": 0.75},
                        "duration_s": 0.02,
                    },
                }
            )
        )
        session.resolve(session.wait(0.02))
        assert world.steps == 1
        before = world.steps
        service.call({"device": "base", "action": "stop", "args": {}})
        assert service.call({"device": "base", "action": "unload", "args": {}})["data"]["unloaded"]
        assert world.control.targets["base"] == [0, 0, 0, 0.75]
        assert world.steps == before
    finally:
        service.close()
