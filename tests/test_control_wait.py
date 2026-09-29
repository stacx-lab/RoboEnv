"""Asynchronous writes and world waits exercise all seven native control paths on CPU."""

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
            env.configured_max_steps = None
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
            env.env = SimpleNamespace(step_limit_enabled=True)
            env.configure_execution_mode = lambda mode: None
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
        self.world = WorldSession(self.adapter, mode="stepped")
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


def test_write_returns_before_time_advances(rig):
    period = rig.adapter.control_interval_s
    result = rig.call("write", {"actions": [rig.goal()] * 2, "duration_s": 4 * period})
    assert result["execution_id"] and result["elapsed_s"] == 0
    assert rig.frames == 0 and rig.stops == 0
    rig.world.advance()
    assert rig.frames == 0
    waited = rig.world.resolve(rig.world.wait(4 * period))
    assert waited["elapsed_s"] == pytest.approx(4 * period)
    assert rig.frames == 4 and rig.stops == 0
    assert not rig.world.pending


def test_reads_and_stop_leave_world_time_unchanged(rig):
    rig.call("write", {"actions": [rig.goal()], "duration_s": 1})
    assert rig.call("read")["positions"] == rig.positions
    rig.call("stop")
    rig.world.poll_motions()
    assert rig.frames == 0 and rig.stops == 1
    assert not rig.world.pending


def test_native_failure_is_reported_by_wait_and_read(rig):
    rig.call("write", {"actions": [rig.goal()], "duration_s": 1})
    rig.failure = True
    with pytest.raises(DeviceError):
        rig.world.resolve(rig.world.wait(0.2))
    assert rig.call("read")["faults"]
    assert rig.stops == 1
    with pytest.raises(DeviceError):
        rig.call("write", {"actions": [rig.goal()], "duration_s": 1})


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
