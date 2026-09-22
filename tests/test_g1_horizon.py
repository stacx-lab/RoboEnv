"""Exercise native timeout configuration and owner budgets without Isaac/GPU imports."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.isaaclab import g1_cameras, g1_native
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.g1 import G1Env
from roboenv_runtime.runtime import world as world_module


@pytest.fixture
def native_factory(monkeypatch):
    class Episode:
        max_episode_length = 1000  # 20 seconds at the existing 50 Hz cadence.
        device = "cpu"

        def __init__(self, cfg):
            self.cfg = cfg
            self.steps = 0
            self.terminated = self.success = False
            self.termination_manager = SimpleNamespace(get_term=lambda _: [self.success])

        def step(self, actions):
            self.steps += 1
            timed_out = self.cfg.terminations.time_out is not None and self.steps >= 1000
            return None, None, [self.terminated], [timed_out], None

        def close(self):
            pass

    def config():
        return SimpleNamespace(
            scene=SimpleNamespace(
                robot=SimpleNamespace(copy=lambda: SimpleNamespace(spawn=SimpleNamespace())),
                packing_table=SimpleNamespace(spawn=SimpleNamespace()),
                object=SimpleNamespace(spawn=SimpleNamespace()),
            ),
            sim=SimpleNamespace(dt=0.005),
            decimation=4,
            episode_length_s=20,
            actions=SimpleNamespace(lower_body_joint_pos=SimpleNamespace()),
            terminations=SimpleNamespace(
                time_out="timeout",
                success="success",
                object_dropping="dropping",
                object_too_far="distance",
            ),
        )

    task_module = (
        "isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg"
    )
    modules = {
        "torch": SimpleNamespace(inference_mode=nullcontext, tensor=lambda values, **_: values),
        "warp": SimpleNamespace(),
        "isaaclab.envs": SimpleNamespace(ManagerBasedRLEnv=Episode),
        "isaaclab.envs.mdp.actions": SimpleNamespace(
            JointPositionAction=object,
            JointPositionActionCfg=lambda **kwargs: kwargs,
        ),
        task_module: SimpleNamespace(LocomanipulationG1EnvCfg=config),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(g1_cameras, "configure", lambda *args: None)

    def build(realtime):
        model = {
            "parts": {},
            "cameras": {},
        }
        paths = {key: key for key in ("robot", "table", "object", "policy")}
        native, names, interval = g1_native.create(paths, model, "cpu", realtime=realtime)
        world = G1Env.__new__(G1Env)
        world.env, world.upper_names, world.model = native, names, model
        world.app = None
        world.realtime = realtime
        # The control period comes from the native config, not the catalog.
        world.control_interval = interval
        world.steps = 0
        world.done = world.completed = False
        return world

    return build


@pytest.mark.parametrize("realtime", [False, True])
def test_horizon_is_removed_only_for_realtime(native_factory, monkeypatch, realtime):
    env = native_factory(realtime)
    terms = env.env.cfg.terminations
    assert terms.time_out == (None if realtime else "timeout")
    assert (terms.success, terms.object_dropping, terms.object_too_far) == (
        "success",
        "dropping",
        "distance",
    )
    adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
    adapter._env = env
    env.control = SimpleNamespace(tick=lambda: env.advance({"base": [0, 0, 0, 0.75]}))
    now = [0.0]
    monkeypatch.setattr(world_module.time, "monotonic", lambda: now[0])
    clock = world_module.WorldSession(adapter)
    clock.start()
    try:
        for _ in range(1100 if realtime else 1000):
            if realtime:
                now[0] += 0.021
                clock.poll()
                continue
            env.control.tick()
        budget = adapter.budget()
        assert budget.max == (None if realtime else 1000)
        assert budget.exhausted is (not realtime)
        assert env.sample_time_s == (22 if realtime else 20)
        assert env.done is (not realtime)
        assert clock.failure is None
        if realtime:
            env.require_active()
            assert not clock._halt.is_set()
            return
        with pytest.raises(ValueError, match="terminated"):
            env.require_active()
    finally:
        clock.close()


@pytest.mark.parametrize("success", [False, True])
def test_realtime_still_honors_non_timeout_termination(native_factory, success):
    env = native_factory(True)
    env.steps = env.env.steps = 1100
    env.env.terminated = True
    env.env.success = success
    env.advance({"base": [0, 0, 0, 0.75]})
    assert env.done and env.completed is success
    with pytest.raises(ValueError, match="terminated"):
        env.advance({"base": [0, 0, 0, 0.75]})
    assert env.steps == 1101
