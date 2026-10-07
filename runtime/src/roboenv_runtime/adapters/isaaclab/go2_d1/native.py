"""Native Isaac Lab world; only this owner advances physics."""

import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .adapter import DEFAULT, LEG_NAMES, RobotModel, Trot, matrix_quat, quat_matrix
from .limits import ARM_EFFORT_LIMITS, ARM_VELOCITY_LIMIT_SIM
from .tasks import PICK_PLACE, definition


class Native:
    dt = 0.005
    interval = 0.02

    def __init__(
        self,
        asset_manifest,
        cameras=True,
        task_name="go2-d1-control",
        task_spec=None,
        device_profiles=None,
    ):
        self.task_spec = definition(task_name) if task_spec is None else task_spec
        manifest = Path(asset_manifest).resolve()
        facts = json.loads(manifest.read_text())
        if facts.get("schema") != "roboenv/go2-d1-assets/1":
            raise ValueError("Unsupported Go2–D1 asset manifest")
        for name, sha in facts["files"].items():
            path = (manifest.parent / name).resolve()
            if (
                not path.is_relative_to(manifest.parent)
                or hashlib.sha256(path.read_bytes()).hexdigest() != sha
            ):
                raise ValueError(f"Changed or missing Go2–D1 asset: {name}")
        self.task_name = task_name
        self.payload_mass_kg = self.task_spec.get("object_mass_kg")
        self.asset_manifest = manifest
        self.asset_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.model = RobotModel(manifest.parent / "robot.urdf")
        self.home = json.loads((manifest.parent / "home.json").read_text())
        if device_profiles is None:
            from .environment import load_profiles

            device_profiles = load_profiles()
        validate_robot_profiles(device_profiles, self.model, self.home)
        self.device_profiles = device_profiles
        self.app = self.sim = None
        self.cameras = {}
        from roboenv_runtime.adapters.gpu import launch_isaaclab

        self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=cameras)
        try:
            self._create(manifest.parent, "cuda:0", cameras)
        except BaseException:
            self.close()
            raise

    def _create(self, directory, device, cameras):
        import isaaclab.sim as sim
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self.torch = torch
        self.sim = sim.SimulationContext(sim.SimulationCfg(dt=self.dt, device=device))
        ground = sim.GroundPlaneCfg(
            physics_material=sim.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            )
        )
        ground.func("/World/Ground", ground)
        light = sim.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)
        # Align the neutral support polygon with the actual folded assembly COM.
        # Recompute when a regenerated URDF changes the arm, mounts or masses.
        self.support_center = np.zeros(2)
        for _ in range(4):
            q = Trot(self.support_center).targets(DEFAULT, 0)
            self.home.update(dict(zip(LEG_NAMES, q.tolist(), strict=True)))
            self.support_center = self.model.center_of_mass(self.home)[:2]

        cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim.UrdfFileCfg(
                asset_path=str(directory / "robot.urdf"),
                usd_dir=str(directory / "usd"),
                fix_base=False,
                merge_fixed_joints=True,
                make_instanceable=False,
                activate_contact_sensors=self.task_name == PICK_PLACE,
                joint_drive=sim.UrdfConverterCfg.JointDriveCfg(
                    gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                articulation_props=sim.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0, 0.30), joint_pos=self.home, joint_vel={".*": 0.0}
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=LEG_NAMES,
                    stiffness=400.0,
                    damping=8.0,
                    effort_limit_sim={
                        ".*_hip_joint": 23.7,
                        ".*_thigh_joint": 23.7,
                        ".*_calf_joint": 35.55,
                    },
                    velocity_limit_sim=20.0,
                ),
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=self.model.arm_names,
                    stiffness=120.0,
                    damping=5.0,
                    effort_limit_sim=dict(ARM_EFFORT_LIMITS),
                    velocity_limit_sim=ARM_VELOCITY_LIMIT_SIM,
                ),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=["d1_Joint7_.*"],
                    stiffness=1000.0,
                    damping=10.0,
                    effort_limit_sim=40.0,
                    velocity_limit_sim=0.1,
                ),
            },
        )
        self.robot = Articulation(cfg)
        from .tasks import TaskScene

        self.scene = TaskScene(self, self.task_name)
        if cameras:
            from isaaclab.sensors import Camera, CameraCfg

            for name in ("camera_head", "camera_wrist"):
                intrinsics = self.device_profiles[name]["intrinsics"]
                color = intrinsics["color"]
                projection = color["projection"]
                width, height = color["width"], color["height"]
                aperture = width * 24.0 / projection["fx"]
                depth_range = intrinsics["depth"]["reliable_range_m"]
                camera = Camera(
                    CameraCfg(
                        prim_path="/World/" + name,
                        width=width,
                        height=height,
                        data_types=["rgb", "distance_to_image_plane"],
                        spawn=sim.PinholeCameraCfg(
                            focal_length=24.0,
                            horizontal_aperture=aperture,
                            vertical_aperture=height * 24.0 / projection["fy"],
                            clipping_range=(depth_range["min"], depth_range["max"]),
                        ),
                    )
                )
                self.cameras[name] = camera
        self.sim.reset()
        self.robot.update(self.dt)
        self.names = self.robot.joint_names
        self.ids = {
            name: self.names.index(name)
            for name in LEG_NAMES + self.model.arm_names + ["d1_Joint7_1", "d1_Joint7_2"]
        }
        self.reset(0)

    @property
    def task(self):
        return self.scene.state

    def array(self, value):
        if hasattr(value, "torch"):
            value = value.torch
        return value[0].detach().cpu().numpy().copy()

    def reset(self, seed):
        self.seed = seed
        self.gait = Trot(self.support_center)
        self.posture_feedback = True
        self.steps = 0
        self.done = False
        self.arm_target = np.array([self.home[n] for n in self.model.arm_names])
        self.gripper_target = np.array([self.home[n] for n in ["d1_Joint7_1", "d1_Joint7_2"]])
        root = self.robot.data.default_root_pose.torch.clone()
        velocity = self.robot.data.default_root_vel.torch.clone()
        self.robot.write_root_pose_to_sim_index(root_pose=root)
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity)
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch.clone()
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch.clone()
        )
        self.robot.reset()
        self.robot.update(self.dt)
        self.scene.reset()

    def advance(self, base, arm, gripper):
        if self.done:
            raise ValueError("Go2 episode has terminated")
        for _ in range(4):
            root = self.root_pose()
            r = quat_matrix(root[3:])
            measured = [root[2], np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1))]
            velocity = self.array(self.robot.data.root_lin_vel_b)
            angular = self.array(self.robot.data.root_ang_vel_b)
            legs = self.gait.targets(
                np.array(base),
                self.dt,
                measured if self.posture_feedback else None,
                [velocity[0], velocity[1], angular[2]],
                [(r @ velocity)[2], angular[0], angular[1]] if self.posture_feedback else None,
            )
            values = dict(zip(LEG_NAMES, legs, strict=True))
            values.update(zip(self.model.arm_names, arm, strict=True))
            values.update(zip(["d1_Joint7_1", "d1_Joint7_2"], gripper, strict=True))
            q = self.robot.data.joint_pos.torch.clone()
            for name, value in values.items():
                bounds = self.model.joints[name]["limits"]
                if name in LEG_NAMES:
                    value = np.clip(value, *bounds)
                if not bounds[0] - 1e-5 <= value <= bounds[1] + 1e-5:
                    raise ValueError(f"Controller exceeded joint limit: {name}")
                q[0, self.ids[name]] = float(value)
            self.robot.set_joint_position_target_index(target=q)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.dt)
            self.scene.physics_update()
        self.steps += 1
        root = self.array(self.robot.data.root_link_pose_w)
        self.done = bool(
            root[2] < 0.16 or not np.isfinite(root).all() or quat_matrix(root[3:])[2, 2] < 0.5
        )

        self.scene.update(self.done)
        self.done = self.done or self.task.done

    def root_pose(self):
        p = self.array(self.robot.data.root_link_pose_w)
        return p

    def joints(self):
        q = self.array(self.robot.data.joint_pos)
        return dict(zip(self.names, q.tolist(), strict=True))

    def capture(self, name, directory):
        from ..observations import camera_state

        camera = self.cameras[name]
        frames = self.model.forward(self.joints())
        optical = (
            "wrist_d405_color_optical_frame"
            if name == "camera_wrist"
            else "head_d435_color_optical_frame"
        )
        base = self.root_pose()
        t = np.eye(4)
        t[:3, :3] = quat_matrix(base[3:])
        t[:3, 3] = base[:3]
        t = t @ frames[optical]
        kwargs = {"device": self.robot.device, "dtype": self.torch.float32}
        camera.set_world_poses(
            positions=self.torch.tensor(t[None, :3, 3], **kwargs),
            orientations=self.torch.tensor(matrix_quat(t[:3, :3])[None], **kwargs),
            convention="ros",
        )
        for _ in range(3):
            self.sim.render()
        return camera_state(camera, directory, self.steps * self.interval)

    def snapshot(self):
        return {
            "root": self.root_pose().tolist(),
            "joints": self.joints(),
            "velocities": self.array(self.robot.data.joint_vel).tolist(),
            "gait": self.gait.state(),
            "posture_feedback": self.posture_feedback,
            "steps": self.steps,
            "done": self.done,
            **self.scene.snapshot(),
        }

    def close(self):
        if self.app is not None:
            error = sys.exc_info()[1]
            if error is not None:
                traceback.print_exc()
            self.app.close(exit_code=1 if error is not None else 0)
            self.app = None


def validate_robot_profiles(profiles, model, home):
    """Check YAML geometry against the prepared URDF before Isaac is started."""
    from .adapter import transform

    def same(actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=1e-7):
            raise ValueError(f"Go2 profile differs from native model: {label}")

    expected_types = {
        "base": "velocity",
        "arm": "joint",
        "gripper": "gripper",
        "camera_head": "rgbd",
        "camera_wrist": "rgbd",
        "world": "world",
    }
    if {n: p["type"] for n, p in profiles.items()} != expected_types:
        raise ValueError("Go2 device inventory differs from the native model")
    gripper_mount = profiles["gripper"]["extrinsics"]["mount"]
    if (gripper_mount["parent"], gripper_mount["parent_frame"]) != ("arm", "flange"):
        raise ValueError("Go2 gripper must be mounted on the arm flange")
    same(gripper_mount["rotation_xyzw"], [0, 0, 0, 1], "gripper orientation")
    for name in ("camera_head", "camera_wrist"):
        intrinsic = profiles[name]["intrinsics"]
        color, depth = intrinsic["color"], intrinsic["depth"]
        width, height = color["width"], color["height"]
        projection = color["projection"]
        if any(type(v) is not int or v <= 0 for v in (width, height)) or (
            depth["width"],
            depth["height"],
        ) != (width, height):
            raise ValueError("Go2 cameras require matching positive RGB-D dimensions")
        if (
            not np.isfinite([projection["fx"], projection["fy"]]).all()
            or min(projection["fx"], projection["fy"]) <= 0
        ):
            raise ValueError("Go2 camera focal lengths must be positive and finite")
        same(
            [projection["cx"], projection["cy"], projection["skew"]],
            [width / 2, height / 2, 0],
            name + " centered projection",
        )
    arm = profiles["arm"]["intrinsics"]
    joints, chain = arm["model"]["joints"], arm["model"]["chain"]
    if [j["name"] for j in joints] != model.arm_names or len(chain) != len(joints):
        raise ValueError("Go2 arm joint order differs from the native model")
    for index, (joint, link) in enumerate(zip(joints, chain, strict=True)):
        name = joint["name"]
        native = model.joints[name]
        same([joint["limits"]["min"], joint["limits"]["max"]], native["limits"], name)
        same(joint["max_effort"], ARM_EFFORT_LIMITS[name], name + " effort")
        if joint["max_velocity"] is not None or joint["type"] != "revolute":
            raise ValueError("Go2 arm requires revolute joints with native velocity limits")
        if link["joint"] != index or link["reference"] != 0:
            raise ValueError("Go2 arm chain order/reference differs from native")
        same(link["anchor"], [0, 0, 0], name + " anchor")
        same(link["axis"], native["axis"], name + " axis")
        same(
            transform({"translation_m": link["origin"][:3], "rotation_xyzw": link["origin"][3:]}),
            native["origin"],
            name + " origin",
        )
        same(arm["home"][name], home[name], name + " home")
    frames = model.forward({})
    for device, parent, parent_frame, frame in (
        ("arm", "base", "base", frames["d1_base_link"]),
        ("camera_head", "base", "base", frames["head_d435_color_optical_frame"]),
        (
            "camera_wrist",
            "arm",
            "flange",
            np.linalg.inv(frames["d1_Link6"]) @ frames["wrist_d405_color_optical_frame"],
        ),
    ):
        mount = profiles[device]["extrinsics"]["mount"]
        if (mount["parent"], mount["parent_frame"]) != (parent, parent_frame):
            raise ValueError(f"Go2 {device} mount parent differs from native")
        same(transform(mount), frame, device + " mount")
