"""Typed absolute targets through RoboDojo's native EvalEnv control cadence."""

import hashlib
import json
import time

import numpy as np

from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose
from roboenv_runtime.paths import robodojo_root

from .backend import RoboDojoEnv
from .rig import describe


def values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def pose_matrix(pose):
    pose = values(pose)
    return matrix([*pose[:3], *pose[4:7], pose[3]])


class RoboDojoDevices(RoboDojoEnv):
    unbounded = False

    def __init__(
        self,
        task_name,
        env_cfg_type="arx_x5",
        *,
        max_steps=None,
        layout_config=None,
    ):
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive action-frame count")
        self.embodiment = env_cfg_type
        self.layout_config = layout_config or env_cfg_type
        if self.layout_config != "arx_x5":
            raise ValueError("layout_config must name a supported native layout collection")
        self.device_facts, self.rig_frames, self.provenance = describe(
            robodojo_root(), env_cfg_type
        )
        self.control = None
        self.physics_steps = 0
        self.requested_frame_limit = max_steps
        self.asset_roots = {"robodojo": robodojo_root() / "Assets"}
        super().__init__(task_name=task_name, env_cfg_type=env_cfg_type)
        self.native_step_limit = int(self._env.step_lim)
        self.native_interval = int(self._env.obs_manager.collect_interval)
        # EvalEnv creates its native simulation on the first reset. Its configured
        # timestep is already available and defines the advertised control clock.
        self.physics_dt = float(self._env.dt)
        self.control_interval = self.native_interval * self.physics_dt
        if self.native_interval < 1 or self.physics_dt <= 0:
            raise ValueError("Invalid native RoboDojo control interval")
        native_step = self._env.sim_step

        def counted_step(*args, **kwargs):
            result = native_step(*args, **kwargs)
            self.physics_steps += 1
            return result

        self._env.sim_step = counted_step
        native_interpolate = self._env.process_control_info

        def interpolate_with_holds(control_info, env_idx):
            sequence = native_interpolate(control_info, env_idx)
            for name, positions in self.gripper_holds.items():
                robot = self.actuators[name][0]
                key = self._env.robot_manager.process_name(robot.gripper_name)
                for frame in sequence:
                    frame[key] = {"position": list(positions), "velocity": 0}
            return sequence

        self._env.process_control_info = interpolate_with_holds
        self.gripper_holds = {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        started, before = time.monotonic(), self.physics_steps
        # Always use native saved-layout reset. Legacy soft reset cannot restore
        # every deformable, fluid, trigger or support-arm state.
        layout = self._layout_id_for(self.seed)
        self._env.env_seeds = [layout]
        self._env.reset(seed=[layout])
        if not np.isclose(self._env.sim.physics_dt, self.physics_dt, rtol=0, atol=1e-12):
            raise ValueError("Native physics timestep differs from the declared control clock")
        for name in ("run_reward", "get_score"):
            hook = getattr(self._env, name, None)
            if callable(hook):
                hook()
        self.initialization = {
            "requested_seed": self.seed,
            "effective_layout": layout,
            "layout_config": self.layout_config,
            "embodiment_adaptation": self.layout_config != self.embodiment,
            "mode": "native_saved_layout",
            "physics_steps": self.physics_steps - before,
            "wall_time_s": time.monotonic() - started,
        }
        self.episode_start_steps = self.physics_steps
        self._configure_step_limit()
        self.actuators = {}
        self.gripper_holds = {}
        limits, targets = {}, {}
        manager = self._env.robot_manager
        for side in ("left", "right"):
            robot = manager.get_robot_by_arm_name(side + "_arm")
            if robot.type != "target":
                raise ValueError("This task requires a separate rig for its native support arm")
            key = manager.robot_key[manager.robot_list.index(robot)]
            names = self.device_facts[side + "_arm"]["intrinsics"]["model"]["joints"]
            if list(robot.arm_joints_name) != [joint["name"] for joint in names]:
                raise ValueError("Native arm joint order differs from source rig")
            native_limits = np.asarray(values(key.data.joint_pos_limits[0]))[
                robot.arm_joint_indices
            ]
            expected = np.array(
                [[joint["limits"]["min"], joint["limits"]["max"]] for joint in names]
            )
            if not np.allclose(native_limits, expected, atol=1e-5, rtol=0):
                raise ValueError("Native USD limits differ from URDF limits")
            for field, native in (
                ("max_velocity", "joint_vel_limits"),
                ("max_effort", "joint_effort_limits"),
            ):
                actual = np.asarray(values(getattr(key.data, native)[0]))[robot.arm_joint_indices]
                if not np.allclose(actual, [joint[field] for joint in names], atol=1e-5, rtol=0):
                    raise ValueError(
                        "Native actuator limits differ from source controller configuration"
                    )
            for suffix, indices in (
                ("arm", robot.arm_joint_indices),
                ("gripper", robot.gripper_joint_indices),
            ):
                name = side + "_" + suffix
                self.actuators[name] = (robot, key, list(indices))
                limits[name] = expected.tolist() if suffix == "arm" else [(0.0, 1.0)]
                targets[name] = self._measured_target(name)
            base = pose_matrix(manager.get_link_pose(robot, self.rig_frames[side]["base"], [0])[0])
            declared = self.device_facts[side + "_arm"]["extrinsics"]["mount"]
            if not np.allclose(
                base, matrix(declared["translation_m"] + declared["rotation_xyzw"]), atol=1e-5
            ):
                raise ValueError(
                    f"Task robot mount differs from source rig: {side}; "
                    f"native={matrix_pose(base)}, "
                    f"declared={declared['translation_m'] + declared['rotation_xyzw']}"
                )
        self.control = DeviceControl(
            limits,
            targets,
            self._apply,
            self._advance,
            exhausted=self.exhausted,
            on_write=self._release_holds,
            interval_s=self.control_interval,
        )
        self._verify_cameras()
        self.initialization["state_sha256"] = self.owner_state()["sha256"]
        from roboenv_runtime.adapters.native_record import initialize_record

        initialize_record(
            self,
            {
                "task_name": self.task_name,
                "env_cfg_type": self.embodiment,
                "layout_config": self.layout_config,
                "max_steps": self.frame_limit,
            },
            "robodojo",
        )

    def _assemble_config(self, seed):
        cfg = super()._assemble_config(seed)
        cfg.eval_cfg.config_name = self.layout_config
        for robot in cfg.robot.robots:
            quaternion = np.asarray(robot["default_root_rot"], dtype=float)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
                raise ValueError("Native robot mount requires a finite wxyz quaternion")
            length = np.linalg.norm(quaternion)
            if length == 0:
                raise ValueError("Native robot mount quaternion cannot be zero")
            # Rounded source values such as [0.707, 0, 0, 0.707] are not unit
            # quaternions. PhysX and the URDF reader normalize them differently.
            robot["default_root_rot"] = (quaternion / length).tolist()
            if robot.get("type", "target") == "target":
                robot["need_planner"] = False
        return cfg

    def _configure_step_limit(self):
        if self.requested_frame_limit is not None and (
            type(self.requested_frame_limit) is not int or self.requested_frame_limit < 1
        ):
            raise ValueError("max_steps must be a positive action-frame count")
        # EvalEnv checks this counter inside take_action/is_episode_end too.
        # Success/failure flags remain native when only the timeout is disabled.
        if self.unbounded:
            self.frame_limit = None
            self._env.step_lim = float("inf")
        else:
            native_limit = self.native_step_limit
            self._env.step_lim = native_limit
            self.frame_limit = min(native_limit, self.requested_frame_limit or native_limit)

    def exhausted(self):
        return bool(self._env.end_flag[0]) or (
            self.frame_limit is not None and self.control.frame >= self.frame_limit
        )

    def _fraction(self, robot, primary):
        low, high = robot.gripper_scale
        fraction = (primary - low) / (high - low)
        return fraction if robot.gripper_move["sign"] == 1 else 1 - fraction

    def _measured_target(self, name):
        robot, key, indices = self.actuators[name]
        positions = np.asarray(values(key.data.joint_pos[0]))[indices].tolist()
        return positions if name.endswith("_arm") else [self._fraction(robot, positions[0])]

    def _apply(self, targets):
        import torch

        self.native_action = {}
        manager = self._env.robot_manager
        for name, target in targets.items():
            robot, key, indices = self.actuators[name]
            arm = name.endswith("_arm")
            command = manager.process_name(robot.arm_name if arm else robot.gripper_name)
            self.native_action[command] = list(target)
            positions = list(target)
            if not arm:
                low, high = robot.gripper_scale
                fraction = target[0] if robot.gripper_move["sign"] == 1 else 1 - target[0]
                primary = low + fraction * (high - low)
                positions = [
                    primary,
                    primary * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
                positions = self.gripper_holds.get(name, positions)
            tensor = torch.tensor([positions], device=key.device, dtype=key.data.joint_pos.dtype)
            key.set_joint_position_target(tensor, joint_ids=indices)
            key.set_joint_velocity_target(torch.zeros_like(tensor), joint_ids=indices)
            # Target setters and cache updates do not advance the world.
            manager.control_manager.prev_control[0][command] = {
                "position": positions,
                "velocity": 0,
            }

    def _advance(self):
        before = self.physics_steps
        self._env.take_action(self.native_action)
        if self.physics_steps - before != self.native_interval:
            raise RuntimeError("Native control cadence changed; acceptance must be revalidated")

    def device_write(self, name, target):
        return self.control.write(name, target)

    def device_stop(self, name):
        if name.endswith("_gripper"):
            self.gripper_holds[name] = self.device_read(name)["positions"]
        return self.control.stop(name, self._measured_target(name))

    def _release_holds(self, devices):
        for name in devices:
            self.gripper_holds.pop(name, None)

    @property
    def sample_time_s(self):
        return (self.physics_steps - self.episode_start_steps) * self.physics_dt

    def device_read(self, name):
        robot, key, indices = self.actuators[name]
        result = {
            "positions": np.asarray(values(key.data.joint_pos[0]))[indices].tolist(),
            "velocities": np.asarray(values(key.data.joint_vel[0]))[indices].tolist(),
            "faults": [],
            "sample_time_s": self.sample_time_s,
        }
        if name.endswith("_arm"):
            frames = self.rig_frames[name.removesuffix("_arm")]
            manager = self._env.robot_manager
            base = pose_matrix(manager.get_link_pose(robot, frames["base"], [0])[0])
            flange = pose_matrix(manager.get_link_pose(robot, frames["flange"], [0])[0])
            result["frame_poses"] = {"flange": matrix_pose(np.linalg.inv(base) @ flange)}
        if name.endswith("_gripper"):
            result.update(
                self.rig_frames[name.removesuffix("_gripper")]["gripper"].read(result["positions"])
            )
        return result

    def _camera(self, name):
        native = {
            "head_camera": "cam_head",
            "left_camera": "cam_left_wrist",
            "right_camera": "cam_right_wrist",
        }[name]
        manager = self._env.camera_manager
        return manager.cameras[0][manager.camera_names[0].index(native)]

    def _verify_cameras(self):
        for name in ("head_camera", "left_camera", "right_camera"):
            camera = self._camera(name)
            definition = self.device_facts[name]
            color = definition["intrinsics"]["color"]
            p = color["projection"]
            expected = [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]]
            actual_k = self._render_intrinsics(name)
            if tuple(camera.get_resolution()) != (
                color["width"],
                color["height"],
            ) or not np.allclose(
                actual_k,
                expected,
                atol=1e-3,
            ):
                raise ValueError(
                    f"Native optical intrinsics differ from source rig: {name}; "
                    f"resolution={camera.get_resolution()}, "
                    f"native={values(actual_k)}, expected={expected}"
                )
            position, quaternion = camera.get_world_pose(camera_axes="ros")
            actual = matrix([*values(position), *values(quaternion)[1:], values(quaternion)[0]])
            mount = definition["extrinsics"]["mount"]
            if mount["parent"] != "world":
                robot = self.actuators[mount["parent"]][0]
                flange = self.rig_frames[mount["parent"].removesuffix("_arm")]["flange"]
                parent = pose_matrix(self._env.robot_manager.get_link_pose(robot, flange, [0])[0])
                actual = np.linalg.inv(parent) @ actual
            if not np.allclose(
                actual, matrix(mount["translation_m"] + mount["rotation_xyzw"]), atol=1e-5
            ):
                raise ValueError(
                    f"Native optical mounting differs from source rig: {name}; "
                    f"native={matrix_pose(actual)}, "
                    f"declared={mount['translation_m'] + mount['rotation_xyzw']}"
                )

    def _render_intrinsics(self, name):
        """Read the effective projection of the native tiled render product."""
        import omni.replicator.core as rep

        camera = self._camera(name)
        index = self._env.camera_manager.cameras[0].index(camera)
        product = self._env.capture_manager.tiled_render_products[index]
        annotator = rep.AnnotatorRegistry.get_annotator("camera_params", device="cpu")
        annotator.attach(product)
        try:
            parameters = None
            for frame in range(8):
                self._env.render()
                if frame < 2:
                    continue
                parameters = annotator.get_data()
                resolution = parameters.get("renderProductResolution", ())
                if len(resolution) == 2 and all(value > 0 for value in resolution):
                    break
            else:
                raise RuntimeError("Native camera parameters are not ready after renderer warmup")
            width, height = parameters["renderProductResolution"]
            if tuple(camera.get_resolution()) != (width, height):
                raise ValueError(
                    f"Native render product resolution differs from its camera: {name}; "
                    f"render_product={(width, height)}, camera={camera.get_resolution()}"
                )
            if parameters["cameraModel"] != "pinhole":
                raise ValueError("Native render product must use pinhole projection")
            projection = np.asarray(parameters["cameraProjection"]).reshape(4, 4)
            if not np.isfinite(projection).all() or not np.allclose(
                [projection[0, 1], projection[1, 0], projection[2, 0], projection[2, 1]], 0
            ):
                raise ValueError("Native render product has undeclared skew or optical offset")
            return np.array(
                [
                    [projection[0, 0] * width / 2, 0, width / 2],
                    [0, projection[1, 1] * height / 2, height / 2],
                    [0, 0, 1],
                ]
            )
        finally:
            annotator.detach(product)

    def device_image(self, name):
        self._env.render()
        # RoboDojo owns tiled render products and their annotators. Its Camera
        # instances describe optics but do not attach individual RGB annotators.
        index = self._env.camera_manager.cameras[0].index(self._camera(name))
        captured = self._env.capture_manager.step(env_ids=[0], cam_ids=[index])
        frame = np.asarray(captured[0]["rgb"][0]["data"])
        color = self.device_facts[name]["intrinsics"]["color"]
        if (
            frame.ndim != 3
            or frame.shape[:2] != (color["height"], color["width"])
            or frame.shape[2] not in (3, 4)
            or frame.dtype != np.uint8
        ):
            raise RuntimeError("Native tiled camera has no valid RGB frame")
        return frame[:, :, :3].copy(), None

    def task_completed(self):
        return bool(self._env.end_flag[0] and self._env.success[0])

    def owner_state(self):
        # Isaac Lab serializes rigid, articulated and deformable scene state.
        # Native reward/support state is included separately and never exported.
        def encode(value):
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if hasattr(value, "shape"):
                return values(value)
            return value

        state = encode(self._env.sim.scene.get_state())
        # RoboDojo creates task objects outside Isaac Lab's scene registry.
        # Read their live native views, including particles and articulated props.
        scene = self._env.scene_manager
        objects = {}
        for kind, attribute in (
            ("rigid", "_rigid_and_dynamic_objects"),
            ("articulation", "_articulation_objects"),
            ("geometry", "_geometry_objects"),
            ("garment", "_garment_objects"),
            ("fluid", "_fluid_objects"),
        ):
            objects[kind] = {}
            for name, obj in sorted(getattr(scene, attribute)[0].items()):
                if kind == "fluid":
                    objects[kind][name] = {
                        "positions": encode(obj.get_particle_positions()),
                        "velocities": encode(
                            np.asarray(obj.point_instancer.GetVelocitiesAttr().Get())
                        ),
                    }
                    continue
                entry = {"pose": encode(obj.get_world_pose())}
                if kind in {"rigid", "articulation"}:
                    entry.update(
                        linear_velocity=encode(obj.get_linear_velocity()),
                        angular_velocity=encode(obj.get_angular_velocity()),
                    )
                if kind == "articulation":
                    entry.update(
                        joint_positions=encode(obj.get_joint_positions()),
                        joint_velocities=encode(obj.get_joint_velocities()),
                    )
                if kind == "garment":
                    entry.update(
                        points=encode(obj._cloth_prim_view.get_world_positions()),
                        velocities=encode(obj._cloth_prim_view.get_velocities()),
                    )
                objects[kind][name] = entry
        reward = self._env.reward_manager
        state["task_objects"] = objects
        state["reward_state"] = {
            key: encode(getattr(reward, key))
            for key in (
                "score_achieved",
                "score_completed_count",
                "final_score_achieved",
                "final_score_completed_count",
                "score_meta",
                "score_trigger_meta",
                "final_score_meta",
            )
        }
        state["predicate_state"] = {
            key: encode(getattr(reward.func_parser, key))
            for key in ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")
        }
        state["control_state"] = encode(self._env.robot_manager.control_manager.prev_control)
        state["support_state"] = {
            key: encode(getattr(self._env, key))
            for key in ("support_arm_action", "query_support_times", "pending_support_check")
            if hasattr(self._env, key)
        }
        state.update(
            native_action_frames=int(self._env.take_action_cnt[0]),
            ended=bool(self._env.end_flag[0]),
            success=self.task_completed(),
            score_completed=encode(self._env.reward_manager.score_completed_count),
            support_actions=encode(getattr(self._env, "support_arm_action", [])),
        )
        serialized = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"state": state, "sha256": hashlib.sha256(serialized.encode()).hexdigest()}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._app is not None:
            self._app.close()
        self.control = None
