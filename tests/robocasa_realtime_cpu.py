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
    native._max_steps = None  # Unbounded service mode.
    worker = SimWorker(
        adapter,
        task_id="cpu-realtime",
        seed=7,
        run_dir=temporary,
        device_root=root / "catalog/robots/robocasa/robocasa-panda-omron",
    )
    try:

        def call(device, action, **args):
            return worker.dispatch(
                "devices.call", {"device": device, "action": action, "args": args}
            )["data"]

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
            target = before.copy()
            target[0] += 0.1
            result = call("arm", "write", target=target.tolist())
            assert result == {"accepted": True}
            arm = native._robot.part_controllers["right"]
            goal = arm.goal_qpos.copy()
            start = native._step_count
            # An unrelated device write must preserve the arm's target.
            call("base", "write", mode="drive", velocity={"vx": 0.1, "vy": 0, "wz": 0})
        wait_frames(worker, start + 6)
        with worker.world.lock:
            np.testing.assert_allclose(arm.goal_qpos, goal, atol=1e-12)
            after = np.array(call("arm", "read")["positions"])
            assert np.linalg.norm(after - before) > 1e-3
            call("base", "stop")
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
                call("arm", "write", target=[100] * 7)
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
    native._max_steps = None
    worker = SimWorker(
        adapter,
        task_id="cpu-synchronous",
        seed=7,
        run_dir=temporary,
        device_root=root / "catalog/robots/robocasa/robocasa-panda-omron",
    )
    try:

        def call(device, action, **args):
            return worker.dispatch(
                "devices.call", {"device": device, "action": action, "args": args}
            )["data"]

        assert worker.world._thread is None
        profile = worker.dispatch("devices.describe", {"device": "arm"})
        assert "control_interval_s" not in profile["intrinsics"]
        initial = native.sim.data.qpos.copy()
        time.sleep(0.15)
        call("arm", "read")
        call("camera_left", "read")
        call("arm", "stop")
        np.testing.assert_array_equal(native.sim.data.qpos, initial)
        assert native._step_count == 0
        goal = list(profile["intrinsics"]["home"].values())
        goal[0] += 0.1
        for frame, (device, target) in enumerate(
            [
                ("arm", goal),
                ("gripper", [1]),
                ("base", [0.1, 0, 0]),
                ("torso", [0.1]),
            ],
            start=1,
        ):
            args = {"target": target}
            if device == "base":
                args = {
                    "mode": "drive",
                    "velocity": dict(zip(("vx", "vy", "wz"), target, strict=True)),
                }
            result = call(device, "write", **args)
            assert result == {"accepted": True}
            assert native._step_count == frame
            np.testing.assert_allclose(native._robot.part_controllers["right"].goal_qpos, goal)
            assert abs(native.sim.data.time - frame * 0.05) < 1e-10
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


def check_bounded(root, temporary, realtime):
    from roboenv_runtime.runtime.worker import WorkerRPCError

    adapter = make_adapter(realtime=realtime)
    native = adapter._env
    native._max_steps = 3
    native.stop_on_success = True
    worker = SimWorker(
        adapter,
        task_id="cpu-bounded",
        seed=7,
        run_dir=temporary,
        device_root=root / "catalog/robots/robocasa/robocasa-panda-omron",
    )
    try:
        if realtime:
            wait_frames(worker, 3)
        else:
            for _ in range(3):
                worker.dispatch(
                    "devices.call",
                    {
                        "device": "gripper",
                        "action": "write",
                        "args": {"target": [1]},
                    },
                )
        with worker.world.lock:
            status = worker.dispatch("simulation.status", {})
            assert status["terminal"] and "reason" not in status and "success" not in status, status
            assert status["budget"]["used"] == status["budget"]["max"] == 3
            position = native.sim.data.qpos.copy()
            sim_time = native.sim.data.time
        time.sleep(0.15)
        with worker.world.lock:
            try:
                worker.dispatch(
                    "devices.call",
                    {
                        "device": "gripper",
                        "action": "write",
                        "args": {"target": [-1]},
                    },
                )
            except WorkerRPCError as exc:
                assert exc.code == "episode_finished"
            else:
                raise AssertionError("Motion accepted after terminal feedback")
            assert native._step_count == 3
            assert native.sim.data.time == sim_time
            np.testing.assert_array_equal(native.sim.data.qpos, position)
            assert worker.dispatch("simulation.status", {}) == status
        assert (temporary / "terminal.json").is_file()
    finally:
        worker.close()
    print(json.dumps({"passed": True, "gpu_replay": False}))


if __name__ == "__main__":
    mode = sys.argv[3] if len(sys.argv) > 3 else "realtime"
    if mode.startswith("bounded-"):
        check_bounded(Path(sys.argv[1]), Path(sys.argv[2]), mode == "bounded-realtime")
    else:
        check_fn = check_synchronous if mode == "synchronous" else check
        check_fn(Path(sys.argv[1]), Path(sys.argv[2]))
