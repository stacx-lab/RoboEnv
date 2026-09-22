"""Public transport contracts on instrumented worlds; not native physics evidence."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.model import DeviceError
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]


def test_robodojo_public_instruction_uses_cached_native_description():
    from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter

    class Native:
        obs_manager = SimpleNamespace(instruction=["Pick up the blue cup by 10 cm."])

        def gen_instruction(self, *_args):
            raise AssertionError("Public reads must not regenerate or resample descriptions")

    adapter = RoboDojoAdapter.__new__(RoboDojoAdapter)
    adapter._env = SimpleNamespace(_env=Native())
    assert adapter.goal() == "Pick up the blue cup by 10 cm."
    assert adapter.goal() == "Pick up the blue cup by 10 cm."


def service(tmp_path, *, embodiment="piper"):
    root = ROOT / "catalog/robots/robotwin/robotwin-dual-piper"
    plan = BindingPlan.load(root, backend="robotwin")
    facts = {}
    for spec in plan.devices:
        definition = spec.definition(root)
        facts[spec.entity] = {
            "type": definition.type,
            "intrinsics": definition.intrinsics,
            "extrinsics": definition.extrinsics,
        }
    world = SimpleNamespace(embodiment=embodiment, device_facts=facts, steps=0)
    actuator_names = [name for name, fact in facts.items() if fact["type"] in {"joint", "gripper"}]
    targets = {name: [0.0] * (6 if name.endswith("arm") else 1) for name in actuator_names}

    def advance():
        world.steps += 1

    world.control = DeviceControl(
        {name: [(-1.0, 1.0)] * len(value) for name, value in targets.items()},
        targets,
        lambda target: None,
        advance,
        realtime=False,
        exhausted=lambda: world.steps >= 3,
        interval_s=1 / 30,
    )
    world.device_write = world.control.write
    world.device_stop = lambda name: world.control.stop(name, targets[name])
    world.device_read = lambda name: {
        "positions": targets[name],
        "velocities": [0.0] * len(targets[name]),
    }
    world.device_image = lambda name: (np.zeros((240, 320, 3), dtype=np.uint8), np.ones((240, 320)))
    owner = SimpleNamespace(_env=world, run_dir=tmp_path)
    context = BindingContext("robotwin", "piper", owner)
    return default_registry().assemble(plan, root, context), world, plan, root, context


def test_public_world_has_no_truth_or_mutation_shortcuts(tmp_path):
    wire, world, *_ = service(tmp_path)
    try:
        assert wire.describe("world")["actions"] == {}
        for action in ("read", "state", "invoke", "score", "reset", "restore"):
            with pytest.raises(DeviceError):
                wire.call({"device": "world", "action": action, "args": {}})
        assert world.steps == 0
    finally:
        wire.close()


def test_independent_writes_and_read_stop_invariance(tmp_path):
    wire, world, *_ = service(tmp_path)

    def call(device, action, args=None):
        return wire.call({"device": device, "action": action, "args": args or {}})

    try:
        call("left_arm", "write", {"target": [0.1] * 6})
        assert world.steps == 1
        with pytest.raises(DeviceError):
            call("right_arm", "write", {"target": [float("nan")] * 6})
        assert world.steps == 1
        call("right_arm", "write", {"target": [0.2] * 6})
        assert world.steps == 2
        for action in ("load", "read", "status", "stop", "unload"):
            call("left_arm", action)
        call("head_camera", "read")
        assert world.steps == 2
        assert len(list((tmp_path / "captures").glob("*.png"))) == 1
    finally:
        wire.close()


def test_native_aliases_are_allowed_and_rig_mismatches_fail_before_effects(tmp_path):
    with pytest.raises(ValueError, match="embodiment"):
        service(tmp_path, embodiment="aloha_agilex")
    wire, world, plan, root, context = service(tmp_path)
    try:
        alias = replace(next(s for s in plan.devices if s.entity == "left_arm"), name="alias")
        duplicate = replace(plan, devices=(*plan.devices, alias))
        aliases = default_registry().assemble(duplicate, root, context)
        try:
            assert aliases.describe("alias")["intrinsics"]["stop"]["affected_devices"] == ["alias"]
        finally:
            aliases.close()
        world.device_facts = copy.deepcopy(world.device_facts)
        world.device_facts["left_arm"]["intrinsics"]["home"]["joint1"] = 0.123
        with pytest.raises(ValueError, match="Catalog intrinsics"):
            default_registry().assemble(plan, root, context)
        assert world.steps == 0
    finally:
        wire.close()


def test_behavior_velocity_binding_accepts_named_commands_without_joint_feedback(tmp_path):
    root = ROOT / "catalog/robots/behavior/behavior-r1pro"
    plan = BindingPlan.load(root, backend="behavior")
    spec = next(s for s in plan.devices if s.name == "base")
    definition = spec.definition(root)
    world = SimpleNamespace(
        embodiment="r1pro",
        device_facts={
            "base": {
                "type": "velocity",
                "intrinsics": definition.intrinsics,
                "extrinsics": definition.extrinsics,
            }
        },
        control=SimpleNamespace(limits={"base": [(-0.5, 0.5), (-0.5, 0.5), (-1, 1)]}),
        device_read=lambda name: {"velocities": [0, 0, 0]},
    )
    writes = []

    def write(name, target, *, for_s=None, arrived=None):
        writes.append((name, target))
        return {"accepted": True, "elapsed_s": for_s or 0.0}

    world.device_write = write
    wire = default_registry().assemble(
        replace(plan, devices=(spec,)),
        root,
        BindingContext("behavior", "r1pro", SimpleNamespace(_env=world, run_dir=tmp_path)),
    )
    try:
        assert wire.call({"device": "base", "action": "read", "args": {}})["data"] == {
            "velocities": [0, 0, 0]
        }
        command = {"mode": "drive", "velocity": {"vx": 0.1, "vy": 0, "wz": -0.2}}
        wire.call({"device": "base", "action": "write", "args": command})
        assert writes == [("base", [0.1, 0, -0.2])]
        for args in (
            {"target": [0, 0, 0]},
            {**command, "posture": {"height": 0.7}},
            {**command, "step": {"id": 0}},
        ):
            with pytest.raises(DeviceError):
                wire.call({"device": "base", "action": "write", "args": args})
        assert len(writes) == 1
    finally:
        wire.close()
