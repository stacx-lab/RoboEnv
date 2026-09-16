"""Native VLA bridge checks with a deterministic stub, not a model benchmark."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.adapters.libero.vla import LiberoPolicyHooks
from roboenv_runtime.vla.attach import build_vla_tool
from roboenv_runtime.vla.runner import run_policy
from roboenv_runtime.vla.spec import load_policy_spec

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    adapter = LiberoAdapter(realtime=False, max_steps=50)
    spec = load_policy_spec("pi05_libero")
    try:
        adapter.reset(0)
        native, robot = adapter._env, adapter._env._robot
        assert "vla_run" not in adapter.tools()
        assert callable(build_vla_tool(adapter, "libero"))
        with pytest.raises(ValueError, match="seven"):
            LiberoPolicyHooks(adapter, replace(spec, action={"dim": 8}))
        native.realtime = True
        with pytest.raises(ValueError, match="realtime=False"):
            LiberoPolicyHooks(adapter, spec)
        native.realtime = False
        native._max_steps = None
        with pytest.raises(ValueError, match="finite"):
            LiberoPolicyHooks(adapter, spec)
        native._max_steps = 50
        hooks = LiberoPolicyHooks(adapter, spec)
        original = robot.controller
        hooks.begin()
        assert robot.controller.name == "OSC_POSE" and robot.action_dim == 7
        before = native.sim.get_state().flatten().copy()
        used, count = adapter.budget().used, len(native.command_log)
        element = hooks.element()
        assert element["observation/state"].shape == (8,)
        for key in ("observation/image", "observation/wrist_image"):
            assert element[key].shape == (224, 224, 3) and element[key].max() > 0
            assert element[key].dtype == np.uint8
        assert adapter.budget().used == used
        for action in (
            [0] * 6,
            [0] * 8,
            [True] * 7,
            [float("nan")] * 7,
            [float("inf")] * 7,
            ["0"] * 7,
        ):
            with pytest.raises(ValueError, match="seven finite"):
                hooks.step(action)
        for operation in (
            lambda: native.step({"gripper": [1]}),
            lambda: native.stop_part("arm"),
            native.tick,
            lambda: native.reset(seed=0),
        ):
            with pytest.raises(RuntimeError, match="Finish the OSC"):
                operation()
        assert len(native.command_log) == count and adapter.budget().used == used
        assert np.array_equal(native.sim.get_state().flatten(), before)
        start = hooks.poses()["gripper"]
        for _ in range(6):
            hooks.step([0.05, 0, 0.05, 0, 0, 0, -1])
        assert adapter.budget().used == used + 6
        displacement = float(np.linalg.norm(hooks.poses()["gripper"] - start))
        assert displacement > 1e-5
        hooks.end()
        assert robot.controller is original and robot.action_dim == 8
        assert np.allclose(
            robot.controller.goal_qpos, native.sim.data.qpos[robot._ref_joint_pos_indexes]
        )
        joint_target = native.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
        joint_target[0] += 0.01
        native.step({"arm": joint_target, "gripper": [-1]})

        class StubClient:
            calls = 0

            def infer(self, observation, **kwargs):
                self.calls += 1
                assert observation["prompt"] == adapter.goal()
                actions = np.zeros((spec.replan_steps, 7))
                actions[:, -1] = -1
                return {"actions": actions}

        client = StubClient()
        used = adapter.budget().used
        report = run_policy(
            hooks,
            client,
            spec,
            n_actions=7,
            reward_fn=lambda: adapter.score().reward,
            poses_fn=hooks.poses,
            budget_fn=lambda: adapter.budget().as_dict(),
        )
        assert report["actions_taken"] == 7 and client.calls == 2
        assert adapter.budget().used == used + 7 and robot.controller is original
        report = run_policy(
            hooks,
            client,
            spec,
            n_actions=7,
            until=lambda: True,
            reward_fn=lambda: adapter.score().reward,
            poses_fn=hooks.poses,
            budget_fn=lambda: adapter.budget().as_dict(),
        )
        assert report["actions_taken"] == 1 and report["note"] == "until condition met"
        assert robot.controller is original and robot.action_dim == 8

        class Client:
            def infer(self, observation, **kwargs):
                raise RuntimeError("injected inference failure")

        with pytest.raises(RuntimeError, match="injected inference"):
            run_policy(
                hooks,
                Client(),
                spec,
                n_actions=2,
                reward_fn=lambda: adapter.score().reward,
                poses_fn=hooks.poses,
                budget_fn=lambda: adapter.budget().as_dict(),
            )
        assert robot.controller is original and robot.action_dim == 8
        native.step({"gripper": [-1]})
        native._max_steps = native._step_count + 1
        hooks.begin()
        hooks.step([0] * 7)
        used = adapter.budget().used
        with pytest.raises(RuntimeError, match="budget exhausted"):
            hooks.step([0] * 7)
        hooks.end()
        assert adapter.budget().used == used
        native._max_steps = 50
        record = adapter.observe()
        (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/replay_libero_devices.py"),
                str(out / "record.json"),
                "--out",
                str(out / "replay.json"),
            ],
            check=True,
        )
        replay = json.loads((out / "replay.json").read_text())
        assert replay["passed"] and replay["max_state_difference"] == 0
        report = {
            "passed": True,
            "model_calls": 0,
            "full_task_attempted": False,
            "displacement_m": displacement,
            "budget": adapter.budget().as_dict(),
            "control_protocol": adapter.score().detail["control_protocol"],
            "replay": replay,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    check(parser.parse_args().out)
