"""Real control waits and dispatch over CPU state; no native physics certification."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.execution import map_result
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.g1_control import Control as IsaacControl
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.backend import LiberoEnv
from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robodojo.devices import RoboDojoDevices
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter
from roboenv_runtime.adapters.simple.adapter import SimpleAdapter
from roboenv_runtime.adapters.simple.control import Control as SimpleControl
from roboenv_runtime.bindings import default_registry
from roboenv_runtime.bindings.native_devices import bind as bind_native
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.worker import SimWorker
from roboenv_runtime.runtime.world import WorldSession

BACKENDS = {
    "libero": LiberoAdapter, "robocasa": RoboCasaAdapter, "robotwin": RoboTwinAdapter,
    "behavior": BehaviorAdapter, "robodojo": RoboDojoAdapter,
    "simple": SimpleAdapter, "isaaclab": IsaacLabAdapter,
}
ROOT = Path(__file__).resolve().parents[1]


class Rig:
    def __init__(self, backend, *, realtime=True):
        self.backend = backend
        self.ticks = 0
        self.threads = []
        self.maximum = None
        self.fail = False
        self.progress = threading.Event()
        self.entered = threading.Event()
        self.adapter = BACKENDS[backend].__new__(BACKENDS[backend])
        env = SimpleNamespace(realtime=realtime, steps=0, done=False, control_interval=0.02,
                              require_active=lambda: None, close=lambda: None)
        self.env = env
        if backend in {"libero", "robocasa"}:
            cls = LiberoEnv if backend == "libero" else RoboCasaEnv
            env = cls.__new__(cls)
            self.env = env
            env.realtime = realtime
            env._step_count, env._max_steps, env.terminal = 0, None, False
            env.env = SimpleNamespace(control_freq=20)
            env.command_log = []
            controller = SimpleNamespace(goal_qpos=np.array([0.0]))
            controller.set_goal = lambda value, **kw: setattr(
                controller, "goal_qpos", kw.get("set_qpos", value))
            env._joint_controller = lambda: controller
            env.env.robots = [SimpleNamespace(part_controllers={"arm": controller})]
            env.tick = self.tick
            self.write = lambda budget, arrive: map_result(
                env.step({"arm": [0.2]}, for_s=budget, arrived=arrive),
                lambda elapsed: {"accepted": True, "elapsed_s": elapsed})
        elif backend in {"simple", "isaaclab"}:
            model = json.loads((ROOT / "runtime/src/roboenv_runtime/adapters" / backend
                                / "g1.json").read_text())
            cls = SimpleControl if backend == "simple" else IsaacControl
            env.model = model
            env.advance = lambda targets: self.tick()
            env.control = cls(env, model, realtime=realtime)
            self.write = lambda budget, arrive: env.control.write(
                "left_arm", {"target": env.control.targets["left_arm"]},
                for_s=budget, arrived=arrive)
        else:
            env.control = DeviceControl(
                {"arm": [(-1, 1)]}, {"arm": [0]}, lambda _: None, self.tick,
                realtime=realtime, exhausted=lambda: self.budget().exhausted, interval_s=0.02)
            if backend == "robodojo":
                self.write = lambda budget, arrive: RoboDojoDevices.device_write(
                    env, "arm", [0.2], for_s=budget, arrived=arrive)
            else:
                self.write = lambda budget, arrive: env.control.write(
                    "arm", [0.2], for_s=budget, arrived=arrive)
        self.adapter._env = env
        self.adapter.budget = self.budget
        self.adapter.close = lambda: None
        self.world = WorldSession(self.adapter)
        # Exercise the same recursive acquisition as worker -> device service.
        self.wire = DeviceService([BoundDevice(
            DeviceDefinition("arm", "joint"),
            {"write": Action(self.execute), "read": Action(lambda *_: {"ticks": self.ticks})},
            implementation="instrumented native clock")],
            execution_lock=self.world.lock, segment_scheduler=self.world.schedule,
            resolve=self.world.resolve)
        self.worker = SimWorker.__new__(SimWorker)
        self.worker.world, self.worker.state = self.world, "ready"
        self.worker._dispatch = lambda op, args: self.wire.submit(args)

    def budget(self):
        return Budget(used=self.ticks, max=self.maximum)

    def tick(self):
        # State is never touched by the caller while a physics step is executing.
        self.threads.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("physics failed")
        self.ticks += 1
        if self.backend in {"libero", "robocasa"}:
            self.env._step_count += 1
        else:
            self.env.steps += 1
        self.progress.set()

    def execute(self, args, execution):
        self.entered.set()
        start = self.ticks
        def arrived():
            return self.ticks - start >= args.get("arrive_after", float("inf"))
        return execution.perform("binding.write", args,
                                 lambda: self.write(args.get("for_s"), arrived), mutating=True)

    def call(self, **args):
        return self.worker.dispatch("devices.call", {
            "device": "arm", "action": "write", "args": args})

    def close(self):
        self.world.close()
        self.wire.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("realtime", [False, True])
def test_bounded_write_advances_inside_worker_and_device_locks(backend, realtime):
    rig = Rig(backend, realtime=realtime)
    rig.world.start()
    try:
        started = time.monotonic()
        result = rig.call(for_s=0.1)["data"]
        assert result["accepted"] is True
        interval = 0.05 if backend in {"libero", "robocasa"} else 0.02
        if realtime:
            assert 0.1 <= result["elapsed_s"] <= 0.1 + interval
        else:
            assert result["elapsed_s"] == pytest.approx(interval)
        assert result["elapsed_s"] > 0
        assert time.monotonic() - started < 0.8  # formerly hit the one-second guard
        assert rig.ticks > 0
        if rig.world.main_thread_clock:
            assert set(rig.threads) == {threading.get_ident()}
    finally:
        rig.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_budget_is_immediate_and_arrival_ends_a_segment(backend):
    rig = Rig(backend)
    rig.world.start()
    try:
        with rig.world.lock:
            assert rig.call()["data"]["elapsed_s"] == 0
        result = rig.call(for_s=2, arrive_after=2)["data"]
        assert 0 < result["elapsed_s"] < 0.2
    finally:
        rig.close()


@pytest.mark.parametrize("backend", ["libero", "simple"])
@pytest.mark.parametrize("termination", ["failure", "exhaustion", "freeze"])
def test_clock_termination_wakes_the_writer(backend, termination):
    rig = Rig(backend)
    if termination == "failure":
        rig.fail = True
    if termination == "exhaustion":
        rig.maximum = 2
    if termination == "freeze":
        tick = rig.adapter.realtime_tick
        def freeze():
            tick()
            rig.world.freeze()
        rig.adapter.realtime_tick = freeze
    rig.world.start()
    try:
        started = time.monotonic()
        if termination == "failure":
            with pytest.raises(DeviceError, match="world clock failed"):
                rig.call(for_s=10)
        else:
            rig.call(for_s=10)
            assert rig.ticks == (2 if termination == "exhaustion" else 1)
        assert time.monotonic() - started < 0.8
    finally:
        rig.close()


def test_reads_finish_during_a_bounded_write():
    rig = Rig("robotwin")
    rig.world.start()
    results, failures = [], []
    def call():
        try:
            results.append(rig.call(for_s=0.2))
        except BaseException as error:
            failures.append(error)
    writer = threading.Thread(target=call)
    reader = threading.Thread(target=lambda: results.append(rig.wire.call({
        "device": "arm", "action": "read", "args": {}})))
    try:
        writer.start()
        assert rig.entered.wait(2)
        reader.start()
        writer.join(2)
        reader.join(2)
        assert not writer.is_alive() and not reader.is_alive()
        assert not failures
        assert [r["sequence"] for r in results] == [2, 1]
        assert results[0]["data"]["ticks"] < 10
    finally:
        rig.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_stalled_clock_retains_wall_bound(backend):
    rig = Rig(backend)
    rig.adapter.realtime_tick = lambda: None
    rig.world.start()
    try:
        started = time.monotonic()
        assert rig.call(for_s=0.02)["data"]["elapsed_s"] == 0
        assert 0.9 <= time.monotonic() - started < 2
    finally:
        rig.close()


def test_replaced_control_uses_the_world_scheduler():
    rig = Rig("robotwin")
    rig.adapter.reset = lambda seed: setattr(rig.env, "control", DeviceControl(
        {"arm": [(-1, 1)]}, {"arm": [0]}, lambda _: None, rig.tick,
        realtime=True, exhausted=lambda: False, interval_s=0.02))
    rig.world.reset(0)
    rig.world.start()
    try:
        rig.world.reset(1)
        assert rig.call(for_s=0.04)["data"]["elapsed_s"] > 0
    finally:
        rig.close()


@pytest.mark.parametrize("backend", ["simple", "isaaclab"])
@pytest.mark.parametrize("realtime", [False, True])
def test_component_bindings_forward_segments_and_report_state(backend, realtime):
    rig = Rig(backend, realtime=realtime)
    task = resolve_task("simple/g1-xmove-pick" if backend == "simple"
                        else "isaaclab/g1-pick-place", project=ROOT, realtime=realtime)
    control = rig.env.control
    def read(name):
        if name == "base":
            return {"velocities": control.targets[name][:3], "faults": []}
        return {"positions": list(control.targets[name]),
                "velocities": [0.0] * len(control.targets[name]), "faults": []}
    rig.env.read_device = read
    wire = default_registry().assemble(
        task.bindings, task.robot, BindingContext(backend, "g1", rig.adapter),
        execution_lock=rig.world.lock, segment_scheduler=rig.world.schedule,
        resolve=rig.world.resolve)
    rig.worker._dispatch = lambda op, args: wire.submit(args)
    rig.world.start()
    def call(name, **args):
        return rig.worker.dispatch("devices.call", {
            "device": name, "action": "write", "args": args})["data"]
    try:
        target = list(control.targets["left_arm"])
        result = call("left_arm", target=target, for_s=0.04)
        assert result["elapsed_s"] > 0 and result["accepted"] is True
        assert result["reached"] is False
        assert result["state"]["positions"] == target
        reached = call("left_arm", target=target, for_s=2, arrive={
            "joint_tolerance": [0.01] * len(target),
            "quiet_velocity": [0.01] * len(target)})
        assert reached["reached"] is True and reached["elapsed_s"] < 0.1
        mode = rig.env.model["control"]["modes"]["locomotion"]
        posture = {c["axis"]: c["default"] for c in mode["posture"]["components"]}
        result = call("base", mode="locomotion", velocity={"vx": 0, "vy": 0, "wz": 0},
                      posture=posture, for_s=0.04)
        assert result["elapsed_s"] > 0 and result["reached"] is False
        assert result["state"]["faults"] == []
        # Invalid budgets are rejected by the public schema before a command is issued.
        with rig.world.lock:
            ticks = rig.ticks
            for budget in (0, -1, "bad"):
                with pytest.raises(DeviceError):
                    call("left_arm", target=target, for_s=budget)
            assert rig.ticks == ticks
    finally:
        wire.close()
        rig.close()


def test_owner_thread_poll_rejects_foreign_thread():
    rig = Rig("simple")
    rig.world.start()
    errors = []
    def wait():
        try:
            rig.world.poll()
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=wait)
    try:
        thread.start()
        thread.join(2)
        assert errors == ["native clock must run on its creating thread"]
        assert rig.ticks == 0
    finally:
        rig.close()


@pytest.mark.parametrize("backend,cls", [
    ("robotwin", None), ("behavior", BehaviorDevices), ("robodojo", RoboDojoDevices),
])
@pytest.mark.parametrize("realtime", [False, True])
def test_shared_native_binding_forwards_segments(backend, cls, realtime):
    rig = Rig(backend, realtime=realtime)
    definition = DeviceDefinition("arm", "joint", intrinsics={
        "stop": {"behavior": "hold", "affected_devices": ["arm"]}})
    rig.env.embodiment = rig.adapter.embodiment
    rig.env.device_facts = {"arm": {"type": "joint", "intrinsics": definition.intrinsics}}
    rig.env.device_read = lambda name: {
        "positions": [0.2], "velocities": [0], "faults": []}
    rig.env.device_write = rig.env.control.write if cls is None else (
        lambda name, target, **kw: cls.device_write(rig.env, name, target, **kw))
    spec = SimpleNamespace(name="arm", entity="arm", binding=backend + ".device/1", config={},
                           definition=lambda root: definition)
    device = bind_native(spec, None, BindingContext(backend, rig.adapter.embodiment, rig.adapter))
    wire = DeviceService([device], execution_lock=rig.world.lock,
                         segment_scheduler=rig.world.schedule, resolve=rig.world.resolve)
    rig.worker._dispatch = lambda op, args: wire.submit(args)
    rig.world.start()
    try:
        result = rig.call(target=[0.2], for_s=0.04)["data"]
        assert result["accepted"] is True and result["elapsed_s"] > 0
        assert result["reached"] is False and result["state"]["positions"] == [0.2]
        result = rig.call(target=[0.2], for_s=1, arrive={"joint_tolerance": [0.01]})["data"]
        assert result["reached"] is True and result["elapsed_s"] < 0.1
    finally:
        wire.close()
        rig.close()
