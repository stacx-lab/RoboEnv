"""Native SAPIEN device control sharing the existing RoboTwin task lifecycle."""

import hashlib
import json
from pathlib import Path

import numpy as np

from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.urdf_geometry import matrix

from .backend import ROBOTWIN_ROOT, RoboTwinEnv
from .rig import EMBODIMENTS, OPTICAL, describe


def pose(value):
    return [*value.p.tolist(), *value.q[1:].tolist(), float(value.q[0])]


class RoboTwinDevices(RoboTwinEnv):
    unbounded = False

    def __init__(
        self,
        task_name="handover_block",
        embodiment="piper",
        *,
        max_steps=None,
        task_config="demo_clean",
        allow_seed_substitution=False,
        max_settling_checks=1,
    ):
        if embodiment not in EMBODIMENTS:
            raise ValueError("Unsupported typed RoboTwin embodiment")
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("max_steps must be positive or null for the task horizon")
        if task_config != "demo_clean":
            raise ValueError("The exported rigs require the stable demo_clean camera mounts")
        self.embodiment = embodiment
        self.evaluation_mode = True
        self.requested_frame_limit = max_steps
        self.frame_limit = max_steps
        self.asset_roots = {
            name: Path(ROBOTWIN_ROOT) / "assets" / name for name in ("embodiments", "objects")
        }
        self.device_facts, self.rig_frames, self.rig_identity = describe(ROBOTWIN_ROOT, embodiment)
        self.control = None
        super().__init__(
            task_name=task_name,
            task_config=task_config,
            embodiment=EMBODIMENTS[embodiment],
            allow_seed_substitution=allow_seed_substitution,
            max_settling_checks=max_settling_checks,
            collect_wrist_camera=True,
        )

    def reset(self, *, seed=None):
        self._depth_cameras = {}
        super().reset(seed=seed)
        self._native_success = bool(self.task.check_success())
        self._configure_step_limit()
        dt = float(self.task.scene.get_timestep())
        if not np.isclose(dt, 1 / 250, rtol=0, atol=1e-8):
            raise ValueError("RoboTwin practice control requires native 250 Hz physics")
        self.control_interval = 10 * dt
        self.gripper_holds = {}
        limits, targets = {}, {}
        for side in ("left", "right"):
            entity = getattr(self.task.robot, side + "_entity")
            arm = side + "_arm"
            description = self.device_facts[arm]["intrinsics"]["model"]
            native_joints = getattr(self.task.robot, side + "_arm_joints")
            if [joint.get_name() for joint in native_joints] != [
                j["name"] for j in description["joints"]
            ]:
                raise ValueError("Native arm joint order differs from the static rig")
            limits[arm] = [(j["limits"]["min"], j["limits"]["max"]) for j in description["joints"]]
            for joint, bounds in zip(native_joints, limits[arm], strict=True):
                if not np.allclose(joint.get_limits()[0], bounds, atol=1e-6, rtol=0):
                    raise ValueError("Native joint limits differ from the pinned URDF")
            targets[arm] = self.device_read(arm)["positions"]
            gripper = side + "_gripper"
            limits[gripper] = [(0, 1)]
            targets[gripper] = self._gripper_target(side)
            base = entity.find_link_by_name(self.rig_frames[side]["base"])
            expected = self.device_facts[arm]["extrinsics"]["mount"]
            target_pose = [*expected["translation_m"], *expected["rotation_xyzw"]]
            if not np.allclose(
                matrix(pose(base.get_pose())), matrix(target_pose), atol=1e-6, rtol=0
            ):
                raise ValueError("Native arm mount differs from the static rig")
        self.control = DeviceControl(
            limits,
            targets,
            self._apply_targets,
            self._advance,
            exhausted=self._exhausted,
            on_write=self._release_holds,
            interval_s=self.control_interval,
        )
        self.initialization.update(
            rig=self.rig_identity,
            control_frequency_hz=25,
            physics_frequency_hz=250,
            native_timestep_s=dt,
            control_decimation=10,
            max_steps=self.frame_limit,
            task_config="demo_clean",
            protocol="absolute_joint_position_25hz_practice",
            loaded_state_sha256=self._state_hash(),
        )
        self._verify_cameras()
        from roboenv_runtime.adapters.native_record import initialize_record

        initialize_record(
            self,
            {
                "task_name": self.task_name,
                "embodiment": self.embodiment,
                "max_steps": self.frame_limit,
                "task_config": "demo_clean",
                "allow_seed_substitution": False,
                "max_settling_checks": self.max_settling_checks,
            },
            "robotwin",
        )

    def _configure_step_limit(self):
        self.frame_limit = None if self.unbounded else self.requested_frame_limit or int(self.task.step_lim)

    def _exhausted(self):
        return (self.frame_limit is not None and self.control is not None
                and self.control.frame >= self.frame_limit)

    def _gripper_target(self, side):
        robot = self.task.robot
        joint = getattr(robot, side + "_gripper")[0][0]
        entity = getattr(robot, side + "_entity")
        index = entity.get_active_joints().index(joint)
        lower, upper = getattr(robot, side + "_gripper_scale")
        return [(float(entity.get_qpos()[index]) - lower) / (upper - lower)]

    def _apply_targets(self, goals):
        robot = self.task.robot
        for side in ("left", "right"):
            target = goals[side + "_arm"]
            robot.set_arm_joints(target, np.zeros(len(target)), side)
            # Stops hold measured finger positions without advancing other grippers.
            hold = self.gripper_holds.get(side + "_gripper")
            if hold is not None:
                for (joint, _, _), value in zip(
                    getattr(robot, side + "_gripper"), hold, strict=True
                ):
                    joint.set_drive_target(value)
                    joint.set_drive_velocity_target(0.0)

    def _advance(self):
        for _ in range(10):
            for side in ("left", "right"):
                name = side + "_gripper"
                if name not in self.gripper_holds:
                    # Preserve native saturation, incremental targets and velocity
                    # feedforward at the same cadence as upstream take_action.
                    self.task.robot.set_gripper(self.control.targets[name][0], side)
            self.task.scene.step()
            self._sim_step_count += 1
            self._native_success = bool(self.task.check_success()) or self._native_success
        self.task.take_action_cnt = self.control.frame + 1

    def device_write(self, name, target):
        return self.control.write(name, target)

    def _release_holds(self, devices):
        for name in devices:
            self.gripper_holds.pop(name, None)

    def device_stop(self, name):
        side = name.split("_", 1)[0]
        if name.endswith("gripper"):
            self.gripper_holds[name] = self.device_read(name)["positions"]
        target = (
            self._gripper_target(side)
            if name.endswith("gripper")
            else self.device_read(name)["positions"]
        )
        return self.control.stop(name, target)

    def device_read(self, name):
        if name not in self.device_facts or self.device_facts[name]["type"] not in {
            "joint",
            "gripper",
        }:
            raise ValueError("Unknown native actuator")
        side = name.split("_", 1)[0]
        entity = getattr(self.task.robot, side + "_entity")
        gripper = name.endswith("gripper")
        joints = (
            [entry[0] for entry in getattr(self.task.robot, side + "_gripper")]
            if gripper
            else getattr(self.task.robot, side + "_arm_joints")
        )
        active = entity.get_active_joints()
        indexes = [active.index(joint) for joint in joints]
        result = {
            "positions": entity.get_qpos()[indexes].astype(float).tolist(),
            "velocities": entity.get_qvel()[indexes].astype(float).tolist(),
            "faults": [],
            "sample_time_s": self.sample_time_s,
        }
        if not gripper:
            frames = self.rig_frames[side]
            base = entity.find_link_by_name(frames["base"]).get_pose()
            flange = entity.find_link_by_name(frames["flange"]).get_pose()
            result.update(efforts=[], frame_poses={"flange": pose(base.inv() * flange)})
        if gripper:
            result.update(self.rig_frames[side]["gripper"].read(result["positions"]))
        return result

    def _camera(self, name):
        cameras = self.task.cameras
        if name == "head_camera":
            return cameras.static_camera_list[cameras.static_camera_name.index(name)]
        if name not in {"left_camera", "right_camera"}:
            raise ValueError("Unknown native camera")
        return getattr(cameras, name)

    @property
    def sample_time_s(self):
        return self._sim_step_count / 250

    def _verify_cameras(self):
        self.task._update_render()
        for name in ("head_camera", "left_camera", "right_camera"):
            camera = self._camera(name)
            facts = self.device_facts[name]
            projection = facts["intrinsics"]["color"]["projection"]
            k = camera.get_intrinsic_matrix()
            if not np.allclose(
                [k[0, 0], k[1, 1], k[0, 2] - 0.5, k[1, 2] - 0.5],
                [projection[key] for key in ("fx", "fy", "cx", "cy")],
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError("Native camera projection differs from the static rig")
            transform = matrix(pose(camera.entity.get_pose())) @ OPTICAL
            if name != "head_camera":
                side = name.split("_", 1)[0]
                link = getattr(self.task.robot, side + "_entity").find_link_by_name(
                    self.rig_frames[side]["flange"]
                )
                transform = np.linalg.inv(matrix(pose(link.get_pose()))) @ transform
            expected = facts["extrinsics"]["mount"]
            if not np.allclose(
                transform,
                matrix(expected["translation_m"] + expected["rotation_xyzw"]),
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("Native camera mount differs from the static rig")

    def device_image(self, name):
        from .cameras import pixel_center_depth

        self.task._update_render()
        camera = self._camera(name)
        depth = pixel_center_depth(self, name, camera)
        camera.take_picture()
        color = camera.get_picture("Color")
        return (np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8), depth

    def owner_state(self):
        scene = self.task.scene
        dynamics = scene.get_physx_system().rigid_dynamic_components
        statics = scene.get_physx_system().rigid_static_components
        return {
            "actors": [
                {
                    "name": body.entity.get_name(),
                    "pose": pose(body.entity.get_pose()),
                    "velocity": body.linear_velocity.tolist() if body in dynamics else [0, 0, 0],
                    "angular_velocity": body.angular_velocity.tolist()
                    if body in dynamics
                    else [0, 0, 0],
                }
                for body in list(dynamics) + list(statics)
            ],
            "articulations": [
                {
                    "name": entity.get_name(),
                    "pose": pose(entity.get_root_pose()),
                    "qpos": entity.get_qpos().tolist(),
                    "qvel": entity.get_qvel().tolist(),
                    "root_velocity": entity.get_root_linear_velocity().tolist(),
                    "root_angular_velocity": entity.get_root_angular_velocity().tolist(),
                }
                for entity in scene.get_all_articulations()
            ],
        }

    def _state_hash(self):
        return hashlib.sha256(json.dumps(self.owner_state(), sort_keys=True).encode()).hexdigest()

    def task_completed(self):
        return self._native_success

    def close(self):
        self._close_task()
        self._depth_cameras = {}
        self.control = None
