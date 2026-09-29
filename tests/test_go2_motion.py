"""Go2 public asynchronous motion and world time without GPU physics."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import load_tool
from roboenv.devices.execution import request_identity
from roboenv.devices.model import DeviceError
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import DEFAULT
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env
from roboenv_runtime.bindings.assembly import assemble_devices
from roboenv_runtime.runtime.worker import SimWorker
from roboenv_runtime.runtime.world import WorldSession

ROOT = Path(__file__).resolve().parents[1]
COMMAND = {
    "mode": "locomotion",
    "velocity": {"vx": 0, "vy": -0.25, "wz": 0},
    "posture": {"height": 0.3, "roll": 0, "pitch": 0},
    "duration_s": 20,
}


@pytest.fixture
def rig():
    task = resolve_task("isaaclab/go2-d1-control", project=ROOT)
    env = Go2Env.__new__(Go2Env)
    env.profiles = {s.name: s.resolved_profile(task.robot) for s in task.bindings.devices}
    joints = env.profiles["arm"]["intrinsics"]["model"]["joints"]
    env.native = SimpleNamespace(
        steps=0,
        done=False,
        model=SimpleNamespace(
            arm_names=[j["name"] for j in joints],
            joints={
                j["name"]: {"limits": [j["limits"]["min"], j["limits"]["max"]]} for j in joints
            },
        ),
    )
    env.max_steps = 1500
    env._stepped_max_steps = 1500
    env.events = []
    env.targets = {"base": DEFAULT.copy(), "arm": np.zeros(6), "gripper": np.zeros(2)}
    samples = []

    def advance(*targets):
        env.native.steps += 1
        samples.append(copy.deepcopy(targets))

    env.native.advance = advance
    env.read_device = lambda name: {
        "faults": [],
        "sample_time_s": env.steps * env.control_interval,
        "positions": env.targets[name].tolist(),
    }
    owner = IsaacLabAdapter.__new__(IsaacLabAdapter)
    owner._env = env
    session = WorldSession(owner, "stepped")
    service = assemble_devices(task.robot, session)
    yield env, session, service, samples
    service.close()


def call(service, name, action="read", **args):
    return service.call({"device": name, "action": action, "args": args})["data"]


def wait(service, duration):
    return call(service, "world", "wait", duration_s=duration)


def test_base_submission_is_immediate_and_only_world_wait_advances_stepped_time(rig):
    env, session, service, samples = rig
    result = call(service, "base", "write", **COMMAND)
    assert result["execution_id"] and result["elapsed_s"] == 0
    assert result["state"]["sample_time_s"] == 0
    assert env.steps == 0
    session.advance()
    assert call(service, "world")["time_s"] == 0
    assert call(service, "base")["sample_time_s"] == 0
    result = wait(service, 20)
    assert result["elapsed_s"] == pytest.approx(20)
    assert result["time_s"] == pytest.approx(20)
    assert env.steps == 1000
    assert all(targets[0][1] == -0.25 for targets in samples)
    assert env.events[0]["args"] == {k: v for k, v in COMMAND.items() if k != "duration_s"}
    assert np.array_equal(env.targets["base"][:3], [0, 0, 0])
    assert np.array_equal(env.targets["base"][3:], [0.3, 0, 0])
    call(service, "base", "stop")
    assert env.steps == 1000


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "20"])
def test_bad_duration_rejected_before_effects(rig, bad):
    env, _, service, samples = rig
    with pytest.raises(DeviceError):
        call(service, "base", "write", **{**COMMAND, "duration_s": bad})
    assert not samples and not env.events
    assert np.array_equal(env.targets["base"], DEFAULT)


def test_invalid_base_fields_do_not_replace_active_motion(rig):
    env, _, service, _ = rig
    call(service, "base", "write", **COMMAND)
    for changes in ({"extra": 1}, {"posture": {}}, {"velocity": {"vx": 0, "vy": True, "wz": 0}}):
        with pytest.raises(DeviceError):
            call(service, "base", "write", **{**COMMAND, **changes})
    assert env.steps == 0 and len(env.events) == 1
    wait(service, 0.1)
    assert env.targets["base"][1] == -0.25
    call(service, "base", "stop")
    assert env.steps == 5


def test_arm_gripper_and_base_share_ticks_and_concurrent_world_waits(rig):
    env, session, service, _ = rig
    call(service, "base", "write", **{**COMMAND, "duration_s": 0.1})
    goal = [0.1] * 6
    call(service, "arm", "write", actions=[goal], duration_s=0.1)
    call(service, "gripper", "write", actions=[0.2, 0.8], duration_s=0.1)
    assert env.steps == 0
    waits = [service.submit({"device": "world", "action": "wait", "args": {"duration_s": 0.1}})
             for _ in range(2)]
    for pending in waits:
        assert session.resolve(pending)["data"]["elapsed_s"] == pytest.approx(0.1)
    assert env.steps == 5
    assert np.array_equal(env.targets["base"][:3], [0, 0, 0])
    assert np.allclose(env.targets["arm"], goal)
    assert np.allclose(env.targets["gripper"], [0.024, -0.024])
    wait(service, 0.1)
    assert np.allclose(env.targets["arm"], goal)
    assert np.allclose(env.targets["gripper"], [0.024, -0.024])


@pytest.mark.parametrize("actions", [[], [[0.5]], [-0.1], [0.2, 1.1], [0.2, float("nan")]])
def test_invalid_gripper_sequence_leaves_existing_reference_unchanged(rig, actions):
    env, _, service, _ = rig
    call(service, "gripper", "write", actions=[0.5], duration_s=0.1)
    with pytest.raises(DeviceError):
        call(service, "gripper", "write", actions=actions, duration_s=0.1)
    assert env.steps == 0 and len(env.events) == 1
    wait(service, 0.1)
    assert np.allclose(env.targets["gripper"], [0.015, -0.015])


def test_replacement_and_late_cancellation_preserve_new_motion(rig):
    env, _, service, _ = rig
    for identity, velocity in [("old", -0.25), ("new", 0.25)]:
        token = request_identity.set(identity)
        try:
            call(service, "base", "write", **{**COMMAND,
                 "velocity": {"vx": 0, "vy": velocity, "wz": 0}})
        finally:
            request_identity.reset(token)
    assert env.steps == 0
    assert [event["op"] for event in env.events] == ["write", "write"]
    assert service.cancel({"execution_id": "old"})["data"] == {"stopped": False}
    wait(service, 0.1)
    assert env.targets["base"][1] == 0.25
    assert service.cancel({"execution_id": "new"})["data"] == {"stopped": True}
    assert env.targets["base"][1] == 0
    assert env.steps == 5


def test_canceling_world_wait_leaves_peer_wait_and_motion_running(rig):
    env, session, service, _ = rig
    call(service, "base", "write", **COMMAND)
    waits = []
    for identity in ["first-wait", "second-wait"]:
        token = request_identity.set(identity)
        try:
            waits.append(service.submit({"device": "world", "action": "wait",
                                         "args": {"duration_s": 0.1}}))
        finally:
            request_identity.reset(token)
    service.cancel({"execution_id": "first-wait"})
    assert waits[0].cancelled() and not waits[1].done()
    session.resolve(waits[1])
    assert env.steps == 5 and env.targets["base"][1] == -0.25


def test_wait_rounds_to_native_tick(rig):
    env, _, service, _ = rig
    call(service, "base", "write", **{**COMMAND, "duration_s": 0.03})
    assert env.steps == 0
    assert wait(service, 0.03)["elapsed_s"] == pytest.approx(0.04)
    assert env.steps == 2 and env.targets["base"][1] == 0


def test_realtime_advances_without_wait_and_expires_velocity(rig, monkeypatch):
    env, session, service, samples = rig
    now = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    session.mode = "realtime"
    session.next_tick = now[0]
    call(service, "base", "write", **{**COMMAND, "duration_s": 1})
    assert env.steps == 0
    for _ in range(50):
        now[0] += 0.021
        session.advance()
    assert call(service, "world")["time_s"] == pytest.approx(1)
    assert env.steps == 50
    assert all(targets[0][1] == -0.25 for targets in samples)
    assert env.targets["base"][1] == 0
    now[0] += 0.021
    session.advance()
    assert env.steps == 51


def test_episode_budget_interrupts_wait_stops_base_without_device_fault(rig):
    env, _, service, _ = rig
    env.max_steps = 3
    call(service, "base", "write", **COMMAND)
    with pytest.raises(DeviceError, match="budget exhausted"):
        wait(service, 20)
    assert env.steps == 3
    assert np.array_equal(env.targets["base"][:3], [0, 0, 0])
    assert call(service, "base")["faults"] == []
    with pytest.raises(DeviceError, match="world is frozen"):
        wait(service, 0.1)


@pytest.fixture
def native_stub(rig, monkeypatch):
    from roboenv_runtime.adapters.isaaclab.go2_d1 import native, profiles

    env = rig[0]
    env.native.arm_target = np.zeros(6)
    env.native.gripper_target = np.zeros(2)
    env.native.reset = lambda seed: None
    env.native.home = {}
    monkeypatch.setattr(native, "Native", lambda *args, **kwargs: env.native)
    monkeypatch.setattr(profiles, "profiles", lambda *args: env.profiles)
    monkeypatch.setattr(Go2Env, "source_identity", staticmethod(lambda: {}))


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("task", ["control", "reach", "pick-place"])
@pytest.mark.parametrize("custom_instruction", [False, True])
def test_catalog_worker_binds_go2_devices(
    native_stub, monkeypatch, tmp_path, mode, task, custom_instruction,
):
    resolved = resolve_task(f"isaaclab/go2-d1-{task}", project=ROOT)
    owner = IsaacLabAdapter(**resolved.task.backend_kwargs)
    monkeypatch.setattr(owner._env, "close", lambda: None)
    instruction = "Custom task instruction." if custom_instruction else resolved.task.instruction
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps({"task": {"id": resolved.task.id, "instruction": instruction}}))
    monkeypatch.setenv("ROBOENV_SERVER_MANIFEST", str(manifest))
    worker = SimWorker(
        owner, task_id=resolved.task.id, seed=0, run_dir=tmp_path,
        device_root=resolved.robot, execution_mode=mode,
    )
    try:
        assert worker.state == "ready"
        assert worker.world.embodiment == resolved.bindings.embodiment == "go2_d1"
        assert call(worker.devices, "world")["time_s"] == 0
        description = worker.dispatch("environment.describe", {})
        assert description["task"]["instruction"] == instruction
    finally:
        worker.close()


def test_go2_rejects_incompatible_world_before_binding(rig, monkeypatch):
    env, session, _, _ = rig
    resolved = resolve_task("isaaclab/go2-d1-pick-place", project=ROOT)
    monkeypatch.setattr(session, "embodiment", "general")
    with pytest.raises(ValueError, match="binding embodiment mismatch"):
        assemble_devices(resolved.robot, session)
    assert env.steps == 0 and not env.events


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("task", ["control", "reach", "pick-place"])
def test_catalog_episode_limits_in_both_clocks(native_stub, monkeypatch, mode, task):
    resolved = resolve_task(f"isaaclab/go2-d1-{task}", project=ROOT)
    env = Go2Env(**resolved.task.backend_kwargs)
    owner = IsaacLabAdapter.__new__(IsaacLabAdapter)
    owner._env = env
    now = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    session = WorldSession(owner, mode)
    pending = session.wait(300.04) if mode == "stepped" else None
    for _ in range(15002):
        now[0] += 0.021
        session.advance()
    if mode == "realtime":
        assert env.max_steps is None and env.steps == 15002
        assert not env.done and session.failure is None
    else:
        assert env.max_steps == 6000 and env.steps == 6000 and env.done
        assert session.time_s == pytest.approx(120)
        assert session.failure is None
        assert session._episode_error.code == "episode_finished"
        with pytest.raises(DeviceError, match="budget exhausted"):
            pending.result()


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_default_budget_and_recorded_mode_survive_reset(native_stub, mode):
    env = Go2Env(task_name="go2-d1-pick-place", execution_mode=mode)
    assert env.max_steps == (6000 if mode == "stepped" else None)
    env.reset(1)
    assert env.execution_mode == mode
    assert env.max_steps == (6000 if mode == "stepped" else None)


def test_realtime_replay_does_not_restore_a_step_cap(native_stub, tmp_path):
    env = Go2Env(task_name="go2-d1-pick-place", execution_mode="realtime")
    native = env.native
    native.asset_sha, native.seed, native.payload_mass_kg = "fixture", 0, 0.08
    task = dict(placement_started=False, completed=False, failure=None,
                hold=0, lift_hold=0, contact_loss=0)
    native.task = SimpleNamespace(spec={})
    native.snapshot = lambda: {
        "steps": native.steps, "root": [0.0], "joints": {"joint": 0.0},
        "task": task, "done": native.done,
    }
    native.reset = lambda seed: setattr(native, "steps", 0)
    native.close = lambda: None
    for _ in range(6001):
        env.tick()
    record = env.record()
    assert record["max_steps"] is None and record["execution_mode"] == "realtime"
    result = load_tool("replay_go2_d1").replay(record, "unused", tmp_path / "replay.json")
    assert result["passed"] and result["steps"] == 6001


@pytest.mark.parametrize("kwargs, error", [
    ({"max_steps": -1}, ValueError),
    ({"execution_mode": "invalid"}, ValueError),
    ({"realtime": True}, TypeError),
    ({"command_timeout_s": 0.5}, TypeError),
    ({"task_name": "go2-d1-reach", "max_steps": 0}, ValueError),
    ({"task_name": "go2-d1-pick-place", "max_steps": True}, ValueError),
])
def test_removed_modes_and_invalid_budgets_fail_before_native_start(monkeypatch, kwargs, error):
    from roboenv_runtime.adapters.isaaclab.go2_d1 import native

    def forbidden(*args, **values):
        pytest.fail("Invalid configuration started the native simulator")

    monkeypatch.setattr(native, "Native", forbidden)
    with pytest.raises(error):
        Go2Env(**kwargs)


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("already_done", [False, True])
def test_native_termination_freezes_clock_and_settles_peer_waits(
    rig, monkeypatch, mode, already_done
):
    env, session, service, samples = rig
    now = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    session.mode = mode
    session.native.configure_execution_mode(mode)
    session.next_tick = now[0]
    call(service, "base", "write", **COMMAND)
    call(service, "arm", "write", actions=[[0.1] * 6, [0.2] * 6], duration_s=0.04)
    short, long = session.wait(0.02), session.wait(1)
    advance = env.native.advance

    def terminal_tick(*targets):
        advance(*targets)
        env.native.done = True

    env.native.advance = terminal_tick
    env.native.done = already_done
    session.advance()
    # A completed physical tick is counted; the next attempted tick reports
    # native termination through the existing no-effects protocol.
    now[0] += 0.021
    session.advance()
    expected = 0 if already_done else 0.02
    assert session.time_s == pytest.approx(expected)
    assert env.steps == len(samples) == (0 if already_done else 1)
    assert long.done() and short.done()
    with pytest.raises(DeviceError) as caught:
        long.result()
    assert caught.value.code == "episode_finished"
    assert caught.value.details["elapsed_s"] == pytest.approx(expected)
    assert caught.value.details["effects"] == ("none" if already_done else "partial")
    if already_done:
        with pytest.raises(DeviceError):
            short.result()
    else:
        assert short.result()["elapsed_s"] == pytest.approx(0.02)
    assert session.failure is None and not session.pending
    assert np.array_equal(env.targets["base"][:3], [0, 0, 0])
    assert np.allclose(env.targets["arm"], [0.1] * 6)
    snapshot = copy.deepcopy(env.events)
    for _ in range(5):
        now[0] += 1
        session.advance()
    assert env.events == snapshot
    assert call(service, "world")["time_s"] == pytest.approx(expected)
    assert call(service, "arm")["sample_time_s"] == pytest.approx(expected)
    with pytest.raises(DeviceError) as refused:
        wait(service, 0.1)
    assert refused.value.code == "episode_finished"
    with pytest.raises(DeviceError):
        call(service, "base", "write", **COMMAND)
    assert env.events == snapshot

    def reset(seed):
        env.native.steps = 0
        env.native.done = False

    env.reset = reset
    env.native.advance = advance
    session.reset(1)
    assert wait(service, 0.02)["elapsed_s"] == pytest.approx(0.02)
    assert env.steps == 1


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_wait_admission_after_native_termination_never_ticks(rig, mode):
    env, session, service, samples = rig
    session.mode = mode
    session.native.configure_execution_mode(mode)
    env.native.done = True
    with pytest.raises(DeviceError) as caught:
        wait(service, 0.1)
    assert caught.value.code == "episode_finished"
    assert session.time_s == env.steps == 0 and not samples
    assert session.failure is None


def test_base_public_write_preserves_commands_outside_former_bounds(rig):
    env, _, service, samples = rig
    command = {
        "mode": "locomotion",
        "velocity": {"vx": 2.0, "vy": -1.0, "wz": 1.0},
        "posture": {"height": 0.5, "roll": -0.3, "pitch": 0.3},
        "duration_s": 0.02,
    }
    call(service, "base", "write", **command)
    assert env.steps == 0
    expected = [2.0, -1.0, 1.0, 0.5, -0.3, 0.3]
    assert np.array_equal(env.targets["base"], expected)
    wait(service, 0.02)
    assert np.array_equal(samples[0][0], expected)
    mode = env.profiles["base"]["intrinsics"]["control"]["modes"]["locomotion"]
    assert all(c["bounds"] is None for group in mode.values() for c in group["components"])
