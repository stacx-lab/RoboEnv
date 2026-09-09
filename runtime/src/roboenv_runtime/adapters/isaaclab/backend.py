"""One synchronous Go2 scene, owned and accessed by the worker's main thread."""

import os

import numpy as np

from .observations import body_state, camera_state, contact_state, imu_state
from .task import JOINT_TOLERANCE_RAD, MAX_INTERVALS
from .rig import CONTROL_DT, DECIMATION, FEET, JOINTS, PHYSICS_DT, facts, scene_config


class IsaacLabEnv:
    def __init__(self, *, realtime=False, device="cuda:0", task_name="go2-posture"):
        if realtime:
            raise ValueError("Isaac Lab currently supports synchronous execution only")
        if task_name != "go2-posture":
            raise ValueError("Unknown Isaac Lab task")
        self.app = None
        self.sim = None
        self.steps = self.held = 0
        self.images = None
        self.image_step = None
        self.model = facts()
        # Asset acquisition and checksum verification happen before Kit starts.
        from roboenv_runtime.paths import repo_root
        from roboenv.provisioning.isaaclab_assets import verify_assets

        asset_root = verify_assets(repo_root())
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        try:
            from isaaclab.app import AppLauncher

            self.app = AppLauncher(headless=True, enable_cameras=True, device=device).app
            import torch
            from isaaclab.scene import InteractiveScene
            from isaaclab.sim import SimulationCfg, SimulationContext

            self.torch = torch
            self.sim = SimulationContext(
                SimulationCfg(dt=PHYSICS_DT, device=device, render_interval=DECIMATION)
            )
            self.scene = InteractiveScene(scene_config(asset_root / "go2.usd"))
            self.sim.reset()
            self.robot = self.scene["robot"]
            self.joint_ids, names = self.robot.find_joints(list(JOINTS), preserve_order=True)
            if tuple(names) != JOINTS or self.robot.num_joints != len(JOINTS):
                raise ValueError("Go2 joint layout differs from the pinned rig")
            self.foot_ids = [self.robot.body_names.index(foot) for foot in FEET]
            limits = self.robot.data.joint_pos_limits.torch[0, self.joint_ids].cpu().numpy()
            if not np.allclose(limits, self.model["limits"], atol=1e-5):
                raise ValueError("Go2 joint limits differ from the catalog")
            home = self.robot.data.default_joint_pos.torch[0, self.joint_ids].cpu().numpy()
            if not np.allclose(home, self.model["home"], atol=1e-6):
                raise ValueError("Go2 initial joint configuration differs from the catalog")
            self.target = self.robot.data.default_joint_pos.torch.clone()
        except BaseException:
            # Kit's fast shutdown can terminate the process before the exception propagates.
            import traceback

            traceback.print_exc()
            self.close()
            raise

    def reset(self, seed):
        self.torch.manual_seed(seed)
        data = self.robot.data
        self.robot.write_root_pose_to_sim_index(root_pose=data.default_root_pose.torch.clone())
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=data.default_root_vel.torch.clone()
        )
        self.robot.write_joint_position_to_sim_index(position=data.default_joint_pos.torch.clone())
        self.robot.write_joint_velocity_to_sim_index(velocity=data.default_joint_vel.torch.clone())
        self.scene.reset()
        self.target = data.default_joint_pos.torch.clone()
        # One initialization interval primes finite-difference sensors; episode time starts afterward.
        self._interval()
        self.steps = self.held = 0
        self.images = self.image_step = None

    def _interval(self):
        for _ in range(DECIMATION):
            self.robot.set_joint_position_target_index(target=self.target)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            # IMU differentiation must observe every physics step, independent of reads.
            _ = self.scene["imu"].data
            for foot in FEET:
                _ = self.scene[foot].data

    def write(self, target):
        if self.steps >= MAX_INTERVALS:
            raise ValueError("Go2 control-interval budget exhausted")
        self.target[:, self.joint_ids] = self.torch.as_tensor(
            target, device=self.sim.device, dtype=self.target.dtype
        )
        self._interval()
        self.steps += 1
        q = self.robot.data.joint_pos.torch[0, self.joint_ids].cpu().numpy()
        self.held = self.held + 1 if np.max(np.abs(q - self.model["posture"])) < JOINT_TOLERANCE_RAD else 0
        return {"accepted": True}

    def stop(self):
        self.target[:, self.joint_ids] = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self.robot.set_joint_position_target_index(target=self.target)
        return {"stopped": True}

    @property
    def sample_time_s(self):
        return self.steps * CONTROL_DT

    def read_body(self):
        return body_state(self.robot, self.joint_ids, self.foot_ids, self.sample_time_s)

    def read_sensor(self, entity):
        if entity == "imu":
            return imu_state(self.scene[entity], self.sample_time_s)
        if entity not in FEET:
            raise ValueError(f"Unknown Go2 sensor: {entity}")
        return contact_state(self.scene[entity], self.robot,
                             self.foot_ids[FEET.index(entity)], self.sample_time_s)

    def read_camera(self, directory):
        if self.image_step == self.steps:
            return dict(self.images)
        self.sim.render()
        self.images = camera_state(self.scene["camera_front"], directory, self.sample_time_s)
        self.image_step = self.steps
        return dict(self.images)

    def close(self):
        try:
            if self.sim is not None:
                self.sim.stop()
                self.sim.clear_instance()
        finally:
            if self.app is not None:
                self.app.close()
                self.app = None
            self.sim = None
