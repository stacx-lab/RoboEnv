"""One official LIBERO world with persistent absolute Panda joint goals."""

import hashlib
import os
import random
from pathlib import Path

import numpy as np

from .config import ensure_libero_config

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")
INITIALIZATION_FRAMES = 5


class LiberoEnv:
    def __init__(
        self,
        suite_name="libero_spatial",
        task_id=0,
        *,
        init_state_index=0,
        realtime=True,
        max_steps=None,
        enable_render=True,
        camera_names=DEFAULT_CAMERAS,
        camera_size=(128, 128),
    ):
        if suite_name not in SUITES:
            raise ValueError("Unsupported official LIBERO suite")
        if type(task_id) is not int or task_id < 0:
            raise ValueError("task_id must be a nonnegative integer")
        if type(init_state_index) is not int or init_state_index < 0:
            raise ValueError("init_state_index must be a nonnegative integer")
        if type(realtime) is not bool or type(enable_render) is not bool:
            raise ValueError("realtime and enable_render must be booleans")
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 5):
            raise ValueError("max_steps must exceed five initialization frames, or be null")
        if not camera_names or set(camera_names) - set(DEFAULT_CAMERAS):
            raise ValueError("Only native LIBERO RGB sensors are supported")
        if tuple(camera_size) != (128, 128):
            raise ValueError("Native LIBERO catalogs require 128 by 128 RGB sensors")
        ensure_libero_config()
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark_dict
        from libero.libero.envs.env_wrapper import ControlEnv

        benchmark = get_benchmark_dict()[suite_name](task_order_index=0)
        if task_id >= benchmark.n_tasks:
            raise ValueError("task_id outside selected suite")
        task = benchmark.get_task(task_id)
        self.bddl = Path(benchmark.get_task_bddl_file_path(task_id))
        self.state_file = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        if not self.bddl.is_file() or not self.state_file.is_file():
            raise ValueError("Task is missing its official BDDL or init-state file")
        # Files are trusted, pinned upstream assets; never load agent-supplied pickle data.
        self._states = benchmark.get_task_init_states(task_id)
        if init_state_index >= len(self._states):
            raise ValueError("init_state_index outside the official state collection")
        self.suite_name, self.task_id = suite_name, task_id
        self.init_state_index = init_state_index
        self.realtime, self._render = realtime, enable_render
        self._camera_names, self._camera_size = tuple(camera_names), tuple(camera_size)
        self._max_steps, self._step_count = max_steps, 0
        self._success = False
        self.initialization = {}
        self._gripper_command = 0.0
        self.command_log = []
        self.wrapper = ControlEnv.__new__(ControlEnv)
        try:
            self.wrapper.__init__(
                bddl_file_name=str(self.bddl),
                controller="JOINT_POSITION",
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=enable_render,
                camera_names=list(camera_names),
                camera_heights=128,
                camera_widths=128,
                camera_depths=False,
                ignore_done=True,
                control_freq=20,
            )
        except BaseException:
            if hasattr(self.wrapper, "env"):
                self.wrapper.close()
            raise
        self.env = self.wrapper.env

    @property
    def _robot(self):
        return self.env.robots[0]

    @property
    def sim(self):
        return self.env.sim

    def reset(self, *, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.wrapper.seed(seed)
        # Use native reset directly: ControlEnv.reset retries randomization forever.
        self.env.reset()
        self.wrapper.set_init_state(self._states[self.init_state_index])
        self._robot.controller.update(force=True)
        self._step_count, self._success = 0, False
        self.stop_part("arm")
        self.stop_part("gripper")
        initial_state_hash = hashlib.sha256(self.sim.get_state().flatten().tobytes()).hexdigest()
        for _ in range(INITIALIZATION_FRAMES):
            self.tick()
        self.command_log = []
        self.initialization = {
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_order_index": 0,
            "seed": seed,
            "init_state_index": self.init_state_index,
            "init_state_sha256": hashlib.sha256(self.state_file.read_bytes()).hexdigest(),
            "bddl_sha256": hashlib.sha256(self.bddl.read_bytes()).hexdigest(),
            "loaded_state_sha256": initial_state_hash,
            "initialization_frames": INITIALIZATION_FRAMES,
            "controller": "JOINT_POSITION/set_qpos",
            "control_frequency_hz": 20,
            "realtime": self.realtime,
            "max_steps": self._max_steps,
        }

    def goal_text(self):
        return str(self.wrapper.language_instruction)

    def get_step_budget(self):
        return {"used": self._step_count, "max": self._max_steps}

    def task_completed(self):
        return self._success

    def step(self, commands):
        if self._max_steps is not None and self._step_count >= self._max_steps:
            raise RuntimeError("LIBERO action frame budget exhausted")
        controller = self._robot.controller
        for part, value in commands.items():
            if part == "arm":
                target = np.array(value, dtype=float, copy=True)
                if not np.array_equal(target, controller.goal_qpos):
                    controller.set_goal(np.zeros(7), set_qpos=target)
            elif part == "gripper":
                self._gripper_command = float(value[0])
            else:
                raise ValueError("Unknown native actuator")
        self.command_log.append(
            {
                "frame": self._step_count,
                "op": "write",
                "commands": {part: np.asarray(value).tolist() for part, value in commands.items()},
            }
        )
        if not self.realtime:
            self.tick()

    def stop_part(self, part):
        if part not in {"arm", "gripper"}:
            raise ValueError("Unknown native actuator")
        self.command_log.append({"frame": self._step_count, "op": "stop", "part": part})
        if part == "arm":
            controller = self._robot.controller
            controller.update(force=True)
            controller.reset_goal()
            return
        self._gripper_command = 0.0
        robot = self._robot
        actuators = robot._ref_joint_gripper_actuator_indexes
        low, high = self.sim.model.actuator_ctrlrange[actuators].T
        positions = self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
        robot.gripper.current_action = np.clip(2 * (positions - low) / (high - low) - 1, -1, 1)

    def tick(self):
        if self._max_steps is not None and self._step_count >= self._max_steps:
            raise RuntimeError("LIBERO action frame budget exhausted")
        controller = self._robot.controller
        previous = controller.__dict__.get("set_goal")
        # robosuite 1.4 updates policy goals at the first substep. Suppress only
        # that update; its native torque, gripper and task hooks still run.
        controller.set_goal = lambda _action: None
        try:
            action = np.zeros(self._robot.action_dim)
            action[-1] = self._gripper_command
            self.env.step(action)
            self.sim.forward()
            self._step_count += 1
            self._success = bool(self.wrapper.check_success()) or self._success
        finally:
            if previous is None:
                del controller.set_goal
            else:
                controller.set_goal = previous

    def render(self, camera=None):
        if not self._render:
            return None
        camera = camera or self._camera_names[0]
        if camera not in self._camera_names:
            raise ValueError("Camera is not enabled")
        height, width = self._camera_size
        return np.asarray(self.sim.render(camera_name=camera, height=height, width=width))[
            ::-1
        ].copy()

    def close(self):
        if hasattr(self.wrapper, "env"):
            self.wrapper.close()
