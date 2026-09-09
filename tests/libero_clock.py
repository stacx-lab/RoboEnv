"""Real native shared-clock, finite-budget, and deterministic frame replay checks."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from roboenv.devices.bindings.spec import BindingPlan
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError

ROOT = Path(__file__).resolve().parents[1]


def robot_subset(directory):
    import shutil

    root = ROOT / "catalog/robots/libero-panda"
    target = directory / "robot"
    shutil.copytree(root, target)
    plan = BindingPlan.load(root, backend="libero")
    plan = replace(plan, devices=tuple(s for s in plan.devices if s.entity != "camera"))
    (target / "bindings.yaml").write_text(yaml.safe_dump(plan.as_dict()))
    return target


def replay(directory):
    directory.mkdir(parents=True, exist_ok=True)
    adapter = LiberoAdapter(enable_render=False, realtime=True, max_steps=605)
    try:
        adapter.reset(0)
        world = adapter._env
        q = world.sim.data.qpos[world._robot._ref_joint_pos_indexes].copy()
        q[0] += 0.1
        # Schedule changes at simulation-frame boundaries, never wall-clock sleeps.
        events = {5: {"arm": q}, 25: {"gripper": [1]}, 40: {"gripper": [-1]}}
        for frame in range(5, 90):
            if frame in events:
                world.step(events[frame])
            if frame == 60:
                world.stop_part("arm")
                world.stop_part("gripper")
            world.tick()
        result = {
            "state": world.sim.get_state().flatten().tolist(),
            "budget": adapter.budget().as_dict(),
            "verdict": adapter.score().completed,
            "initialization": world.initialization,
        }
        (directory / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        adapter.close()


def check(directory):
    directory.mkdir(parents=True, exist_ok=True)
    robot = robot_subset(directory)
    for realtime in (True, False):
        adapter = LiberoAdapter(enable_render=False, realtime=realtime, max_steps=10)
        worker = SimWorker(
            adapter,
            task_id="libero-clock-probe",
            seed=0,
            run_dir=directory / str(realtime),
            device_root=robot,
        )
        try:

            def call(action, worker=worker, **args):
                return worker.dispatch(
                    "devices.call", {"device": "arm", "action": action, "args": args}
                )["data"]

            if realtime:
                deadline = time.monotonic() + 10
                while adapter.budget().used < 10:
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            else:
                state = adapter._env.sim.get_state().flatten().copy()
                time.sleep(0.1)
                for _ in range(4):
                    call("read")
                assert np.array_equal(state, adapter._env.sim.get_state().flatten())
                while adapter.budget().used < 10:
                    call("write", target=call("read")["positions"])
            assert adapter.budget().used == 10
            time.sleep(0.1)
            assert adapter.budget().used == 10
            try:
                call("write", target=call("read")["positions"])
            except WorkerRPCError:
                pass
            else:
                raise AssertionError("Budget allowed extra physics")
            final = worker.submit()
            assert final == worker.submit()
            time.sleep(0.1)
            assert adapter.budget().used == 10
        finally:
            worker.close()
        if realtime:
            assert not worker.world._thread.is_alive()
    for index in (0, 1):
        result = subprocess.run(
            [sys.executable, __file__, "--replay", "--out", str(directory / f"replay-{index}")],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    a, b = [json.loads((directory / f"replay-{i}/replay.json").read_text()) for i in (0, 1)]
    assert a["initialization"] == b["initialization"]
    assert a["budget"] == b["budget"] and a["verdict"] == b["verdict"]
    assert np.allclose(a["state"], b["state"], atol=1e-10, rtol=0)
    (directory / "report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "replay_max_state_difference": float(
                    np.max(np.abs(np.array(a["state"]) - b["state"]))
                ),
                "replay_native_success": a["verdict"],
                "task_success_claimed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    (replay if args.replay else check)(args.out)
