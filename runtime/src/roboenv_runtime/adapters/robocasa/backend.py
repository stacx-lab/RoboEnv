"""One RoboCasa world, native control intervals, and permitted measurements."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

DEFAULT_ROBOT = "PandaOmron"
DEFAULT_TASK = "PickPlaceCounterToCabinet"
DEFAULT_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def _ensure_offscreen_rendering():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def joint_controller_config(robot):
    """Use physical absolute joint goals for every manipulator part."""
    from robosuite.controllers import load_composite_controller_config, load_part_controller_config

    config = load_composite_controller_config(robot=robot)
    parts = config["body_parts"]
    for arm in (name for name in ("right", "left") if name in parts):
        gripper = parts[arm].get("gripper")
        parts[arm] = {
            **load_part_controller_config(default_controller="JOINT_POSITION"),
            "input_type": "absolute",
        }
        if gripper is not None:
            parts[arm]["gripper"] = gripper
    if "torso" in parts:
        parts["torso"]["input_type"] = "absolute"
    return config


class RoboCasaEnv:
    def __init__(
        self,
        task_name: str = DEFAULT_TASK,
        *,
        robot: str = DEFAULT_ROBOT,
        layout_ids: Any = None,
        style_ids: Any = None,
        seed: int | None = None,
        max_steps: int | None = None,
        stop_on_success: bool = False,
        enable_render: bool = True,
        camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
        camera_size: tuple[int, int] = (256, 256),
        obj_instance_split: str | None = None,
    ) -> None:
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer or null")
        if type(stop_on_success) is not bool:
            raise ValueError("stop_on_success must be a boolean")
        self.stop_on_success = stop_on_success
        _ensure_offscreen_rendering()
        import robocasa  # noqa: F401  (importing it registers the kitchen environments)
        import robosuite

        self._stream_commands = {}
        self.task_name = task_name
        self.robot = robot
        self.seed = seed
        self._render = bool(enable_render)
        self._camera_names = tuple(camera_names)
        self._camera_size = tuple(camera_size)
        self.configured_max_steps = max_steps
        self._max_steps = max_steps
        self._step_count = 0
        self.owner_recording = None
        self.terminal = False
        self.on_step = None
        controller_config = joint_controller_config(robot)
        env_kwargs: dict[str, Any] = {
            "robots": robot,
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": bool(enable_render),
            "use_camera_obs": False,  # Camera devices render on demand, never on the physics thread.
            "camera_names": list(camera_names) if enable_render else None,
            "camera_heights": camera_size[0],
            "camera_widths": camera_size[1],
            "camera_depths": bool(enable_render),
            "ignore_done": True,
            "translucent_robot": False,
            "seed": seed,
        }
        if layout_ids is not None:
            env_kwargs["layout_ids"] = layout_ids
        if style_ids is not None:
            env_kwargs["style_ids"] = style_ids
        if obj_instance_split is not None:
            env_kwargs["obj_instance_split"] = obj_instance_split
        self.env = robosuite.make(task_name, **env_kwargs)
        self.horizon = int(getattr(self.env, "horizon", 1000) or 1000)
        self._obs: dict[str, Any] | None = None

    def _ensure_built(self) -> None:
        """Robosuite builds the kitchen (fixtures, objects, language) on the first reset;
        prompts read those before the trial's own reset, so build once on demand.
        """
        if not hasattr(self.env, "fixtures") or not hasattr(self.env, "obj_body_id"):
            self.reset(seed=self.seed)

    @property
    def _robot(self) -> Any:
        # robosuite rebuilds its Robot objects on every reset, so never cache one.
        return self.env.robots[0]

    @property
    def sim(self) -> Any:
        """The MuJoCo simulation wrapper."""
        return self.env.sim

    def close(self) -> None:
        """Close the simulation."""
        self.env.close()

    def episode_meta(self) -> dict[str, Any]:
        """RoboCasa's episode metadata (layout, style, language, fixture refs ...)."""
        self._ensure_built()
        try:
            return dict(self.env.get_ep_meta())
        except Exception:
            return {}

    def goal_text(self) -> str:
        """The task's language instruction for this episode."""
        return str(self.episode_meta().get("lang", ""))

    def task_completed(self) -> bool:
        """The benchmark's own success check."""
        return bool(self.env._check_success())

    def list_objects(self) -> list[str]:
        """Names of the task's movable objects."""
        self._ensure_built()
        return sorted(self.env.objects)

    def get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose ``(position, quaternion_wxyz)`` of an object or fixture."""
        body_id = self.env.obj_body_id.get(name)
        if body_id is None:
            fixture = self.env.fixtures.get(name)
            if fixture is None:
                raise KeyError(f"unknown object or fixture {name!r}")
            root = getattr(fixture, "root_body", None)
            if root and root in self.sim.model.body_names:
                body_id = self.sim.model.body_name2id(root)
            else:  # static fixture: placement pose from the model
                pos = np.asarray(getattr(fixture, "pos", (0.0, 0.0, 0.0)), dtype=float)
                quat = np.asarray(getattr(fixture, "quat", (1.0, 0.0, 0.0, 0.0)), dtype=float)
                return pos.copy(), quat.copy()
        pos = np.array(self.sim.data.body_xpos[body_id], dtype=float)
        quat = np.array(self.sim.data.body_xquat[body_id], dtype=float)  # MuJoCo: wxyz
        return pos, quat

    def get_step_budget(self) -> dict[str, int | None]:
        """Native control frames executed by pending commands."""
        return {"used": self._step_count, "max": self._max_steps}

    def render(self, camera: str | None = None) -> np.ndarray | None:
        """An RGB frame from ``camera`` (default: the first configured camera)."""
        if not self._render:
            return None
        self._pixel_center_rendering()
        name = camera or self._camera_names[0]
        h, w = self._camera_size
        frame = self.sim.render(camera_name=name, height=h, width=w)
        return np.asarray(frame)[::-1].copy()

    def render_rgbd(self, camera: str):
        """Aligned upright RGB and optical-axis depth in metres, one render."""
        from robosuite.utils.camera_utils import get_real_depth_map

        if not self._render or camera not in self._camera_names:
            raise ValueError("Camera must be enabled in the task's simulator configuration")
        self._pixel_center_rendering()
        h, w = self._camera_size
        rgb, depth_buffer = self.sim.render(camera_name=camera, height=h, width=w, depth=True)
        raw = np.asarray(depth_buffer, dtype=float)
        # Far-plane/background pixels and invalid renderer values have no depth.
        valid = np.isfinite(raw) & (raw >= 0) & (raw < 1)
        metres = get_real_depth_map(self.sim, np.where(valid, raw, 0))
        metres = np.where(valid, metres, 0)
        return np.asarray(rgb)[::-1].copy(), metres[::-1].copy()

    def _pixel_center_rendering(self):
        # Multisample depth resolves a subpixel sample, unlike the catalog's center ray.
        # Recreate only the GPU buffers, on the rendering thread, after each reset.
        model = self.sim.model._model
        if model.vis.quality.offsamples:
            model.vis.quality.offsamples = 0
            context = self.sim._render_context_offscreen
            if context is not None:
                context.gl_ctx.make_current()
                context.con.free()
                context._set_mujoco_context_and_buffers()

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed = seed
            self.env.rng = np.random.default_rng(seed)
        self._obs = self.env.reset()
        self._step_count = 0
        self._stream_commands = {}
        for part in self._robot.part_controllers:
            self.stop_part(part)
        return self._obs

    @property
    def control_interval_s(self):
        return 1.0 / self.env.control_freq

    def step(self, commands):
        """Stage validated native references without advancing physics."""
        if self._max_steps is not None and self._step_count >= self._max_steps:
            raise RuntimeError("episode_finished: no further motion is permitted")
        if self.owner_recording is not None:
            self.owner_recording.command("write", {k: np.asarray(v).tolist() for k, v in commands.items()})
        for part, command in commands.items():
            controller = self._robot.part_controllers[part]
            value = np.array(command, dtype=float, copy=True)
            if part in {"base", "right_gripper"}:
                self._stream_commands[part] = value
                continue
            if not np.array_equal(value, controller.goal_qpos):
                controller.set_goal(value)
        return 0.0

    def stop_part(self, part):
        if self.owner_recording is not None:
            self.owner_recording.command("stop", part)
        controller = self._robot.part_controllers[part]
        if part == "base":
            self._stream_commands[part] = np.zeros(controller.control_dim)
            controller.set_goal(np.zeros(controller.control_dim))
        elif part == "right_gripper":
            self._stream_commands[part] = np.zeros(1)
            # Hold the current finger positions as native normalized targets.
            low, high = controller.actuator_min, controller.actuator_max
            positions = self.sim.data.qpos[controller.qpos_index]
            target = np.clip(2 * (positions - low) / (high - low) - 1, -1, 1)
            self._robot.gripper["right"].current_action = target.copy()
            controller.set_goal(target.copy())
        else:
            # reset_goal() uses the controller's cached pre-substep state. Hold
            # the measured terminal configuration, not that stale sample.
            controller.set_goal(self.sim.data.qpos[controller.qpos_index].copy())

    def tick(self):
        """Run one native interval while retaining position targets.

        Robosuite's normal step retains all physics, observables and task hooks.
        Only its policy goal update is replaced during this locked call: arm and
        torso goals persist, while velocity / gripper inputs run continuously.
        """
        if self._max_steps is not None and self._step_count >= self._max_steps:
            return
        composite = self._robot.composite_controller

        def update_streams(_action):
            for part, command in self._stream_commands.items():
                value = command.copy()
                if part == "base":
                    from .base import native_command

                    value = native_command(self._robot.part_controllers[part], value)
                if part == "right_gripper":
                    value = self._robot.gripper["right"].format_action(value)
                self._robot.part_controllers[part].set_goal(value)

        # Restore even on native failure, and preserve instance/class lookup.
        previous = composite.__dict__.get("set_goal")
        composite.set_goal = update_streams
        try:
            self._obs, _, _, _ = self.env.step(np.zeros(self._robot.action_dim))
            self.sim.forward()  # Publish transforms consistent with the integrated joints.
            self._step_count += 1
            if self.on_step is not None:
                self.on_step()
        finally:
            if previous is None:
                del composite.set_goal
            else:
                composite.set_goal = previous
