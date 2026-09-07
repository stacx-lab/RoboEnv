"""Autonomous PandaOmron physics, CPU only, no kitchen or rendered pixels."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from robocasa_cpu import make_adapter
from roboenv_runtime.runtime.worker import SimWorker


def wait_frames(worker, count):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with worker.world.lock:
            assert worker.world.failure is None, worker.world.failure
            if worker.adapter.budget().used >= count:
                return
        time.sleep(0.01)
    raise AssertionError("Background simulation did not advance")


def check(root, temporary):
    adapter = make_adapter()
    native = adapter._env
    native.realtime = True
    native.horizon = native.env.horizon = 2
    native._max_steps = 1  # Legacy overrides must not cap realtime testing.
    worker = SimWorker(adapter, task_id="cpu-realtime", seed=7, run_dir=temporary,
                       device_root=root / "catalog/robots/robocasa-panda-omron")
    try:
        def call(device, action, **args):
            return worker.dispatch("devices.call", {"device": device, "action": action,
                                                    "args": args})["data"]

        with worker.world.lock:
            # Drop the free cube, then issue no request while physics runs.
            address = native.sim.model.get_joint_qpos_addr(native.env.cube.joints[0])[0]
            native.sim.data.qpos[address + 2] += 0.4
            native.sim.forward()
            initial_z = native.sim.data.qpos[address + 2]
            start = native._step_count
        wait_frames(worker, start + 4)
        with worker.world.lock:
            assert native.sim.data.qpos[address + 2] < initial_z - 0.03
            assert native._step_count > native.horizon
            assert adapter.budget().max is None
            assert not adapter.budget().exhausted
            before = np.array(call("arm", "read")["positions"])
            result = call("arm", "write", target=[0, 0, 0.5, 0, 0, 0])
            assert result == {"completed": False, "accepted": True}
            arm = native._robot.part_controllers["right"]
            goal = arm.goal_pos.copy()
            start = native._step_count
            # An unrelated device write must preserve the arm's target.
            call("body", "write", target=[0.1, 0, 0])
        wait_frames(worker, start + 6)
        with worker.world.lock:
            np.testing.assert_allclose(arm.goal_pos, goal, atol=1e-12)
            after = np.array(call("arm", "read")["positions"])
            assert np.linalg.norm(after - before) > 1e-3
            call("body", "stop")
            call("arm", "stop")
            call("gripper", "write", target=[1])
            start = native._step_count
        wait_frames(worker, start + 3)
        with worker.world.lock:
            call("gripper", "stop")
            held = native._robot.gripper["right"].current_action.copy()
            start = native._step_count
        wait_frames(worker, start + 2)
        with worker.world.lock:
            np.testing.assert_allclose(native._robot.gripper["right"].current_action, held)
            np.testing.assert_allclose(native._robot.part_controllers["base"].goal_qvel, 0)
            # Invalid command has no side effect on a previously accepted goal.
            from roboenv_runtime.runtime.worker import WorkerRPCError
            try:
                call("arm", "write", target=[2] * 6)
            except WorkerRPCError:
                pass
            else:
                raise AssertionError("Out-of-range input accepted")
        final = worker.submit()
        frames = native._step_count
        time.sleep(0.15)
        assert native._step_count == frames
        assert worker.submit() == final
    finally:
        worker.close()
    assert not worker.world._thread.is_alive()
    print(json.dumps({"passed": True, "gpu_replay": False}))


def check_synchronous(root, temporary):
    adapter = make_adapter(realtime=False)
    native = adapter._env
    native.horizon = native.env.horizon = 2
    native._max_steps = 1
    worker = SimWorker(adapter, task_id="cpu-synchronous", seed=7, run_dir=temporary,
                       device_root=root / "catalog/robots/robocasa-panda-omron")
    try:
        def call(device, action, **args):
            return worker.dispatch("devices.call", {"device": device, "action": action,
                                                    "args": args})["data"]

        assert worker.world._thread is None
        profile = worker.dispatch("devices.describe", {"device": "arm"})
        assert profile["intrinsics"]["clock_mode"] == "stepped"
        initial = native.sim.data.qpos.copy()
        time.sleep(0.15)
        call("arm", "read")
        call("camera_left", "read")
        call("arm", "stop")
        np.testing.assert_array_equal(native.sim.data.qpos, initial)
        assert native._step_count == 0
        for frame, (device, target) in enumerate([
            ("arm", [0, 0, 0.5, 0, 0, 0]),
            ("gripper", [1]),
            ("body", [0.1, 0, 0]),
            ("torso", [0.1]),
        ], start=1):
            result = call(device, "write", target=target)
            assert result == {"accepted": True, "completed": True}
            assert native._step_count == frame
            position = native.sim.data.qpos.copy()
            sim_time = native.sim.data.time
            time.sleep(0.075)
            np.testing.assert_array_equal(native.sim.data.qpos, position)
            assert native.sim.data.time == sim_time
        assert native._step_count > native.horizon
        assert np.linalg.norm(native.sim.data.qpos - initial) > 1e-4
        assert adapter.budget().max is None
        result = worker.submit()
        assert result["budget"]["max"] is None
    finally:
        worker.close()
    print(json.dumps({"passed": True, "gpu_replay": False}))


if __name__ == "__main__":
    check_fn = check_synchronous if sys.argv[3:] == ["synchronous"] else check
    check_fn(Path(sys.argv[1]), Path(sys.argv[2]))
