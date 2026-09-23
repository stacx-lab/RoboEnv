"""Synchronous sequences exercise all seven actual native control paths on CPU."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.g1_control import Control as IsaacControl
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.backend import LiberoEnv
from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.control import Control as SimpleControl
from roboenv_runtime.bindings.motion import control_binding
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession

BACKENDS = {
    "libero": LiberoAdapter,
    "robocasa": RoboCasaAdapter,
    "robotwin": RoboTwinAdapter,
    "behavior": BehaviorAdapter,
    "robodojo": RoboDojoAdapter,
    "simple": SimpleAdapter,
    "isaaclab": IsaacLabAdapter,
}
ROOT = Path(__file__).resolve().parents[1]


class Rig:
    def __init__(self, backend):
        self.frames, self.stops = 0, 0
        self.maximum, self.failure = None, False
        self.positions = [0.0]
        self.adapter = BACKENDS[backend].__new__(BACKENDS[backend])
        self.adapter.budget = lambda: Budget(used=self.frames, max=self.maximum)
        env = SimpleNamespace(
            steps=0,
            done=False,
            control_interval=0.02,
            require_active=lambda: None,
            close=lambda: None,
        )
        self.name = "arm"
        if backend in {"libero", "robocasa"}:
            cls = LiberoEnv if backend == "libero" else RoboCasaEnv
            env = cls.__new__(cls)
            env._step_count, env._max_steps, env.terminal = 0, None, False
            env.env = SimpleNamespace(control_freq=20)
            env.command_log = []
            controller = SimpleNamespace(goal_qpos=np.array([0.0]))
            controller.set_goal = lambda value, **kw: setattr(
                controller, "goal_qpos", kw.get("set_qpos", value)
            )
            env._joint_controller = lambda: controller
            env.env.robots = [SimpleNamespace(part_controllers={"arm": controller})]
            env.tick = lambda: self.tick(controller.goal_qpos)
            self.stage = lambda q: env.step({"arm": q})
        elif backend in {"simple", "isaaclab"}:
            model = json.loads(
                (ROOT / "runtime/src/roboenv_runtime/adapters" / backend / "g1.json").read_text()
            )
            env.model = model
            env.advance = lambda targets: self.tick(targets["left_arm"])
            env.control = (SimpleControl if backend == "simple" else IsaacControl)(env, model)
            self.name = "left_arm"
            self.positions = list(env.control.targets[self.name])
            self.stage = lambda q: env.control.write(self.name, {"target": q})
            env.read_device = lambda name: self.read()
        else:
            env.control = DeviceControl(
                {"arm": [(-3, 3)]},
                {"arm": [0]},
                lambda _: None,
                lambda: self.tick(env.control.targets["arm"]),
                exhausted=lambda: self.adapter.budget().exhausted,
                interval_s=0.02,
            )
            self.stage = lambda q: env.control.write("arm", q)
        self.adapter._env, self.env = env, env
        facts = (
            env.control.profiles[self.name]["intrinsics"]
            if backend in {"simple", "isaaclab"}
            else {
                "model": {"joints": [{"type": "revolute", "limits": {"min": -3, "max": 3}}]},
                "control": {"mode": "position"},
                "feedback": {"velocities": True},
            }
        )
        facts["stop"] = {"behavior": "hold", "affected_devices": [self.name]}
        self.profile = facts

        @control_binding
        def bind(spec, root, context):
            return BoundDevice(
                DeviceDefinition(self.name, "joint", facts),
                {
                    "read": Action(lambda args, ctx: self.read()),
                    "write": Action(lambda args, ctx: self.stage(args["target"])),
                    "stop": Action(lambda args, ctx: self.stop()),
                },
                implementation="native-cpu-rig",
            )

        device = bind(
            SimpleNamespace(entity=self.name),
            None,
            BindingContext(backend, self.adapter.embodiment, self.adapter),
        )
        self.world = WorldSession(self.adapter)
        self.service = DeviceService(
            [device],
            execution_lock=self.world.lock,
            motion_scheduler=self.world.schedule,
            resolve=self.world.resolve,
            flush=self.world.poll_motions,
        )

    def tick(self, target):
        if self.failure:
            raise RuntimeError("physics failure")
        self.frames += 1
        self.env.steps = self.env._step_count = self.frames
        self.positions = [
            q + max(-0.05, min(0.05, t - q)) for q, t in zip(self.positions, target, strict=True)
        ]

    def read(self):
        return {
            "positions": list(self.positions),
            "velocities": [0.0] * len(self.positions),
            "faults": [],
            "sample_time_s": self.frames * self.adapter.control_interval_s,
        }

    def stop(self):
        self.stops += 1
        return {"stopped": True, **self.profile["stop"]}

    def call(self, action, args=None, *, deferred=False):
        request = {"device": self.name, "action": action, "args": args or {}}
        return self.service.submit(request) if deferred else self.service.call(request)["data"]

    def goal(self):
        # Move toward zero if a joint starts near its upper limit.
        return [q + (0.2 if q <= 0 else -0.2) for q in self.positions]


@pytest.fixture(params=BACKENDS)
def rig(request):
    return Rig(request.param)


def test_sequence_has_exact_duration_without_arrival_wait(rig):
    target, period = rig.goal(), rig.adapter.control_interval_s
    result = rig.call("write", {"actions": [target, target], "freq_hz": 1 / (2 * period)})
    assert result["completed_actions"] == 2
    assert result["elapsed_s"] == pytest.approx(4 * period)
    assert rig.frames == 4 and rig.stops == 1 and "reached" not in result
    rig.call("read")
    rig.world.advance()
    assert rig.frames == 4


def test_upper_layer_move_to_without_motion_limits(rig, monkeypatch):
    import copy
    import importlib.util

    from helpers import kinex_sources

    kinex, _ = kinex_sources()
    if not (kinex / "packages/core/assets/builtin/tool/move-to/tool.py").is_file():
        pytest.skip("requires Kinex move_to sources")
    monkeypatch.syspath_prepend(str(kinex / "packages/core/assets/python"))
    spec = importlib.util.spec_from_file_location(
        "move_to_tool", kinex / "packages/core/assets/builtin/tool/move-to/tool.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    joints = copy.deepcopy(rig.profile["model"]["joints"])
    for index, joint in enumerate(joints):
        joint.update(name=f"j{index}", max_velocity=None, max_acceleration=None)
    tool.profile = lambda name: {
        "type": "joint", "resolved_model": {"joints": joints}, "intrinsics": rig.profile,
    }
    calls = []

    def device(name, action, **args):
        calls.append(action)
        return {"data": json.loads(json.dumps(rig.call(action, args)))}

    tool.device = device
    tool.read = lambda name: json.loads(json.dumps(rig.call("read")))
    tool.stop_all = lambda names: rig.call("stop")
    monkeypatch.setattr("kinex.control.device", device)
    target = json.loads(json.dumps(rig.goal()))
    result = tool.control({"device": rig.name, "joints": target, "timeout_s": 5})
    assert result["reached"] and result["elapsed_s"] <= 5
    assert result["state"]["positions"] == pytest.approx(target)
    assert calls and set(calls) == {"write"}
    assert "set" not in rig.service.describe(rig.name)["actions"]




def test_buffer_timeout_stops_without_overshooting(rig):
    period = rig.adapter.control_interval_s
    result = rig.call(
        "write",
        {
            "actions": [rig.goal()] * 3,
            "freq_hz": 1 / rig.adapter.control_interval_s,
            "for_s": 1.5 * period,
        },
    )
    assert result["terminated_by"] == "timeout" and rig.frames == 1 and rig.stops == 1
    assert result["elapsed_s"] == pytest.approx(period)


def test_invalid_later_point_frequency_and_old_protocol_have_no_effects(rig):
    target = rig.goal()
    for args in (
        {"actions": [target, [100] * len(target)], "freq_hz": 50},
        {"actions": [target], "freq_hz": 0},
        {"target": target, "for_s": 1},
    ):
        with pytest.raises(DeviceError):
            rig.call("write", args)
    assert rig.frames == 0


def test_reads_stop_and_queued_cancellation(rig):
    args = {"actions": [rig.goal()] * 20, "freq_hz": 1 / rig.adapter.control_interval_s}
    futures = [rig.call("write", args, deferred=True) for _ in range(2)]
    rig.world.advance()
    assert rig.call("read")["positions"] == rig.positions
    rig.call("stop")
    rig.world.poll_motions()
    for future in futures:
        with pytest.raises(DeviceError, match="interrupt"):
            future.result()
    assert rig.frames == 1


def test_native_failure_fails_pending_motion(rig):
    future = rig.call(
        "write",
        {"actions": [rig.goal()], "freq_hz": 1 / rig.adapter.control_interval_s},
        deferred=True,
    )
    rig.failure = True
    rig.world.advance()
    with pytest.raises(DeviceError, match="physics failure"):
        future.result()
    assert rig.stops == 1


def test_owner_thread_is_enforced():
    rig, errors = Rig("libero"), []

    def wrong_thread():
        try:
            rig.world.advance()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert errors == ["native operations must run on their creating thread"]


def test_failed_timeout_stop_fences_queued_motion(rig):
    stages = []
    original = rig.stage
    rig.stage = lambda q: stages.append(q) or original(q)

    def failed_stop():
        rig.stops += 1
        raise RuntimeError("halt unavailable")

    rig.stop = failed_stop
    pending = rig.call(
        "write",
        {
            "actions": [rig.goal()] * 3,
            "freq_hz": 1 / rig.adapter.control_interval_s,
            "for_s": rig.adapter.control_interval_s,
        },
        deferred=True,
    )
    queued = rig.call(
        "write",
        {"actions": [rig.goal()], "freq_hz": 1 / rig.adapter.control_interval_s},
        deferred=True,
    )
    rig.world.advance()
    with pytest.raises(DeviceError, match="halt unavailable"):
        pending.result()
    with pytest.raises(DeviceError, match="owner recovery"):
        queued.result()
    assert rig.stops == 1 and len(stages) == 1


@pytest.mark.parametrize("frequency", [3, 123.4])
def test_native_paths_accept_arbitrary_sequence_rates(rig, frequency):
    result = rig.call("write", {"actions": [rig.goal()] * 3, "freq_hz": frequency})
    assert result["completed_actions"] == 3
    assert abs(result["elapsed_s"] - 3 / frequency) <= rig.adapter.control_interval_s
