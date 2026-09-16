"""Continuous native motor control with BDDL evaluation owned by RoboEnv."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

from .backend import BehaviorBackend, fix_cache_env
from .instances import restore_instance
from .rig import array, describe


def pose(prim):
    position, quaternion = prim.get_position_orientation()
    return matrix([*array(position), *array(quaternion)])


def robot_config():
    # Controller groups are model facts, not assumptions about an old R1Pro class.
    from omnigibson.utils.asset_utils import get_dataset_path

    definition = yaml.safe_load(
        (Path(get_dataset_path("omnigibson-robot-assets")) / "models/r1pro/r1pro.yaml").read_text()
    )
    controllers = {}
    for group in definition["raw_controller_order"]:
        controllers[group] = {
            "name": "HolonomicBaseJointController" if group == "base" else "JointController",
            "motor_type": "velocity" if group == "base" else "position",
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "subsume_controllers": [],
        }
        if group != "base":
            controllers[group]["use_delta_commands"] = False
    return {
        "name": "robot",
        "model": "r1pro",
        "obs_modalities": ["rgb", "depth_linear"],
        "scale": 1.0,
        "self_collisions": True,
        "action_normalize": False,
        "action_type": "continuous",
        "grasping_mode": "physical",
        "controller_config": controllers,
        "sensor_config": {
            "VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 640}}
        },
    }


class BehaviorDevices(BehaviorBackend):
    embodiment = "r1pro"
    control_interval = 1 / 30

    def __init__(
        self,
        activity_name,
        scene_model=None,
        activity_instance_id=0,
        max_steps=6000,
        *,
        realtime=False,
    ):
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive action-frame count")
        super().__init__(activity_name, scene_model, activity_instance_id, max_steps)
        self.realtime = realtime
        self.frame_limit = max_steps
        self.control = None
        self._native_success = False
        self._native_done = False

    def build_config(self):
        cfg = super().build_config()
        from omnigibson.tasks.behavior_task import BehaviorTask

        # The released full scene is template zero. Other initial states are
        # task-relevant state files, restored after the base environment resets.
        cfg["scene"]["scene_instance"] = BehaviorTask.get_cached_activity_scene_filename(
            self.scene_model, self.activity_name, 0, 0
        )
        cfg["robots"] = [robot_config()]
        cfg["env"]["action_frequency"] = 30
        cfg["env"]["physics_frequency"] = 120
        cfg["env"]["external_sensors"] = None
        return cfg

    def reset(self, seed=0):
        import torch

        random.seed(seed)
        np.random.seed(int(seed) % 2**32)
        torch.manual_seed(seed)
        started = time.monotonic()
        self.boot()
        import omnigibson as og
        from omnigibson.sensors import VisionSensor
        from omnigibson.utils.asset_utils import get_dataset_path

        self.env.reset()
        # Attach the native camera-parameter annotators while the initial render
        # graph is live, before the instance restore refreshes physics handles.
        for sensor in self.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.camera_parameters
        instance = restore_instance(self.env, self.robot, self.activity_instance_id)
        if not np.isclose(og.sim.get_physics_dt(), 1 / 120) or not np.isclose(
            og.sim.get_sim_step_dt(), self.control_interval
        ):
            raise ValueError("Native BEHAVIOR clock differs from the declared device protocol")
        self._native_start_time = float(og.sim.current_time)
        self._native_start_index = int(og.sim.current_time_step_index)
        self.asset_roots = {
            name: Path(get_dataset_path(name))
            for name in (
                "omnigibson-robot-assets",
                "behavior-1k-assets",
                "2026-challenge-task-instances",
            )
        }
        self.device_facts, self.groups, self.rig_frames, self.cameras, provenance = describe(
            self.robot, self.control_interval, realtime=self.realtime
        )
        limits, targets = {}, {}
        lower, upper = array(self.robot.joint_lower_limits), array(self.robot.joint_upper_limits)
        indices = self.robot.controller_action_idx
        for name, group in self.groups.items():
            group["action_indices"] = array(indices[group["controller"]]).astype(int).tolist()
            if len(group["action_indices"]) != len(group["indices"]):
                raise ValueError("Native controller must accept one command per controlled joint")
            if name == "base":
                limits[name] = [(-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0)]
                targets[name] = [0.0, 0.0, 0.0]
            elif name.endswith("_gripper"):
                group["lower"] = lower[group["indices"]]
                group["upper"] = upper[group["indices"]]
                if np.any(group["upper"] <= group["lower"]):
                    raise ValueError("Gripper has no finite physical joint range")
                limits[name], targets[name] = [(0.0, 1.0)], self._measured_target(name)
            else:
                joints = self.device_facts[name]["intrinsics"]["model"]["joints"]
                limits[name] = [
                    (joint["limits"]["min"], joint["limits"]["max"]) for joint in joints
                ]
                actual = np.column_stack([lower[group["indices"]], upper[group["indices"]]])
                if not np.allclose(actual, limits[name], atol=1e-5):
                    raise ValueError("Native USD joint limits differ from the R1Pro URDF")
                targets[name] = self._measured_target(name)
        self._native_done = False
        self._native_success = bool(self.goal_report()["success"])
        self.gripper_holds = {}
        self.control = DeviceControl(
            limits,
            targets,
            self._apply,
            self._advance,
            realtime=self.realtime,
            exhausted=self.exhausted,
            on_write=self._release_holds,
        )
        self.initialization = {
            "requested_seed": seed,
            "activity": self.activity_name,
            "scene_model": self.scene_model,
            "activity_instance_id": self.activity_instance_id,
            "instance": instance,
            "seed_selects_instance": False,
            "wall_time_s": time.monotonic() - started,
            "rig": provenance,
            "grasping_mode": "physical",
            "control_frequency_hz": 30,
            "physics_frequency_hz": 120,
            "native_clock_start_s": self._native_start_time,
            "native_context_start_index": self._native_start_index,
        }
        self.initialization["loaded_state_sha256"] = self.owner_state()["sha256"]
        from roboenv_runtime.adapters.native_record import initialize_record

        initialize_record(
            self,
            {
                "activity_name": self.activity_name,
                "scene_model": self.scene_model,
                "activity_instance_id": self.activity_instance_id,
                "max_steps": self.frame_limit,
                "realtime": self.realtime,
            },
            "behavior",
        )
        # The base environment reads this packaged YAML in addition to Python.
        # Keep its identity task-independent so different activity instances can
        # share native device conformance while configuration edits require it again.
        self.implementation_identity["configuration_sha256"] = hashlib.sha256(
            (Path(__file__).parent / "r1pro.yaml").read_bytes()
        ).hexdigest()

    def exhausted(self):
        return self._native_done or (
            self.control is not None and self.control.frame >= self.frame_limit
        )

    def _measured_target(self, name):
        if name == "base":
            return [0.0, 0.0, 0.0]
        group = self.groups[name]
        q = array(self.robot.get_joint_positions())[group["indices"]]
        if name.endswith("_gripper"):
            return [float(np.mean((q - group["lower"]) / (group["upper"] - group["lower"])))]
        return q.tolist()

    def _apply(self, targets):
        import torch

        action = np.zeros(self.robot.action_dim, dtype=np.float32)
        for name, target in targets.items():
            group = self.groups[name]
            value = np.asarray(target)
            if name.endswith("_gripper"):
                value = group["lower"] + value[0] * (group["upper"] - group["lower"])
                value = self.gripper_holds.get(name, value)
            action[group["action_indices"]] = value
        self.action = torch.tensor(action)
        self.robot.apply_action(self.action)

    def _advance(self):
        _, _, terminated, truncated, _ = self.env.step(self.action)
        self._native_success = self._native_success or bool(self.goal_report()["success"])
        self._native_done = bool(terminated or truncated)

    def device_write(self, name, target):
        return self.control.write(name, target)

    def _release_holds(self, devices):
        for name in devices:
            self.gripper_holds.pop(name, None)

    def device_stop(self, name):
        if name.endswith("_gripper"):
            self.gripper_holds[name] = self.device_read(name)["positions"]
        result = self.control.stop(name, self._measured_target(name))
        return {**result, "behavior": "zero_velocity" if name == "base" else "hold"}

    def device_read(self, name):
        elapsed = self.control.frame * self.control_interval
        if name == "base":
            transform = pose(self.robot)
            linear = transform[:3, :3].T @ array(self.robot.get_linear_velocity())
            angular = transform[:3, :3].T @ array(self.robot.get_angular_velocity())
            value = matrix_pose(transform)
            return {
                "pose": {"translation_m": value[:3], "rotation_xyzw": value[3:]},
                "velocities": [float(linear[0]), float(linear[1]), float(angular[2])],
                "faults": [],
                "sample_time_s": elapsed,
            }
        group = self.groups[name]
        result = {
            "positions": array(self.robot.get_joint_positions())[group["indices"]].tolist(),
            "velocities": array(self.robot.get_joint_velocities())[group["indices"]].tolist(),
            "faults": [],
            "sample_time_s": elapsed,
        }
        if name in self.rig_frames:
            frame = self.rig_frames[name]
            result["frame_poses"] = {
                "flange": matrix_pose(
                    np.linalg.inv(pose(self.robot.links[frame["base"]]))
                    @ pose(self.robot.links[frame["flange"]])
                )
            }
        if name.endswith("_gripper"):
            result.update(group["gripper"].read(result["positions"]))
        return result

    def device_image(self, name):
        import omnigibson as og

        og.sim.render()
        observation, _ = self.robot.sensors[self.cameras[name]].get_obs()
        color = array(observation["rgb"])[..., :3]
        depth = array(observation["depth_linear"])
        if color.dtype != np.uint8 or color.shape[:2] != depth.shape:
            raise RuntimeError("Native RGB/depth sensor formats disagree")
        return color.copy(), depth.copy()

    def task_completed(self):
        return self._native_success

    def owner_state(self):
        import omnigibson as og

        def encode(value):
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if hasattr(value, "shape"):
                return array(value).tolist()
            return value

        state = encode(og.sim.dump_state(serialized=False))
        state["roboenv_goal"] = self.goal_report()
        state["roboenv_clock"] = {
            "elapsed_native_time_s": float(og.sim.current_time) - self._native_start_time,
            "native_context_steps": int(og.sim.current_time_step_index) - self._native_start_index,
        }
        raw = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"state": state, "sha256": hashlib.sha256(raw.encode()).hexdigest()}

    def close(self):
        if self.env is not None:
            import atexit

            import omnigibson as og

            # clear() releases the world but leaves Kit alive. Shut Kit down
            # before Python unloads native plugins; otherwise a completed replay
            # can segfault during interpreter teardown. Defer this because the
            # caller still owns result serialization and its process exit code.
            atexit.register(og.shutdown)
            og.clear()
            self.env = None
            self.robot = None
        self.control = None


def compile_rig():
    """Used only by the owner generator, with the same controller/sensor settings."""
    fix_cache_env()
    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "robots": [robot_config()],
            "env": {"action_frequency": 30, "physics_frequency": 120},
        }
    )
    try:
        og.sim.render()
        return describe(env.robots[0], 1 / 30)[0]
    finally:
        og.clear()
