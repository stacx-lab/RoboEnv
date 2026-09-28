"""Real Panda/Omron servo execution on CPU, with synthetic camera pixels."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from helpers import kinex_sources
from robocasa_cpu import make_adapter
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError


def check(root, temporary):
    adapter = make_adapter()
    native = adapter._env
    worker = SimWorker(
        adapter,
        task_id="cpu-sync",
        seed=7,
        execution_mode="stepped",
        run_dir=temporary,
        device_root=root / "catalog/robots/robocasa/robocasa-panda-omron",
    )

    def call(device, action, **args):
        return worker.dispatch("devices.call", {"device": device, "action": action, "args": args})[
            "data"
        ]

    try:
        profile = worker.dispatch("devices.describe", {"device": "arm"})
        initial = native.sim.data.qpos.copy()
        for _ in range(3):
            call("arm", "read")
            call("camera_left", "read")
            worker.world.advance()
        assert native._step_count == 0
        np.testing.assert_array_equal(native.sim.data.qpos, initial)
        goal = list(profile["intrinsics"]["home"].values())
        goal[0] += 0.1
        result = call("arm", "write", actions=[goal] * 4, duration_s=len([goal] * 4) / 10)
        assert result["execution_id"] and result["elapsed_s"] == 0
        assert native._step_count == 0
        call("world", "wait", duration_s=0.4)
        assert native._step_count == 8
        np.testing.assert_allclose(native._robot.part_controllers["right"].goal_qpos, goal)
        assert np.linalg.norm(native.sim.data.qpos - initial) > 1e-4
        # Exercise the actual upper-layer planner against native torque dynamics,
        # whose introspected joint model has no velocity or acceleration limits.
        kinex_root, _ = kinex_sources()
        sys.path.insert(0, str(kinex_root / "packages/core/assets/python"))
        spec = importlib.util.spec_from_file_location(
            "servo_tool", kinex_root / "packages/core/assets/builtin/tool/servo/tool.py"
        )
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        tool.profile = lambda name: {
            **profile,
            "resolved_model": {"joints": profile["intrinsics"]["model"]["joints"]},
        }
        tool.device = lambda name, action, **args: {"data": call(name, action, **args)}
        tool.read = lambda name: call(name, "read")
        tool.stop_all = lambda names: [call(name, "stop") for name in names]
        import kinex.control

        kinex.control.device = tool.device
        goal = call("arm", "read")["positions"]
        goal[0] += 0.1
        before = native._step_count
        tool.control({"device": "arm", "joints": goal, "duration_s": 5})
        assert native._step_count == before
        call("world", "wait", duration_s=5)
        np.testing.assert_allclose(call("arm", "read")["positions"], goal, atol=0.01)
        before = native._step_count
        for name, args in [
            ("gripper", {"actions": [1]}),
            ("base", {"mode": "drive", "velocity": {"vx": 0.1, "vy": 0, "wz": 0}}),
        ]:
            call(name, "write", duration_s=0.05, **args)
        assert native._step_count == before
        call("world", "wait", duration_s=0.05)
        assert native._step_count == before + 1
        native._max_steps = native._step_count + 2
        try:
            call("arm", "write", actions=[goal] * 3, duration_s=0.15)
            call("world", "wait", duration_s=0.15)
        except WorkerRPCError:
            pass
        else:
            raise AssertionError("Native budget must interrupt incomplete sequences")
        assert native._step_count == native._max_steps
    finally:
        worker.close()
    print(json.dumps({"passed": True, "gpu_replay": False}))


if __name__ == "__main__":
    check(Path(sys.argv[1]), Path(sys.argv[2]))
