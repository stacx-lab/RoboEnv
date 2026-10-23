"""The simulated world in Isaac Lab (PhysX): flat ground, a task's scene, boxes and items, the T2
and its head camera."""

from __future__ import annotations

import math
import hashlib

from .assets import generated_directory
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.actuator import booster_pd_torque
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.controller import RobotState
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.kinematics import RobotKinematics
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import RobotConfig
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.rotations import matrix_to_quat, quat_to_matrix, rpy_matrix
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.scenes import Scene, graspnet_usd, articulated_model
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.articulation import t2_articulation_cfg
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.task import Box, Item, SceneTask, Articulated

RENDER_PASSES = 3
"""Renders before reading a camera, so that it shows the robot where it is now: a single pass
still draws the robot at an older pose."""

SCENE_PRIM = "/World/Scene"
"""Prim of the task's house."""

SCENE_SOLIDS = ("walls", "openings", "furniture")
"""Groups of the house that collide; its ceilings are hidden."""

TASK_PRIM = "/World/Task"
"""Parent prim of the task's boxes and items."""

ITEM_CLEARANCE = 0.002
"""Gap left under an item at the start, m, so that it starts out of contact."""


def _np(proxy) -> np.ndarray:
    return proxy.torch[0].detach().cpu().numpy().astype(float)


def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    return np.r_[xyzw[3], xyzw[:3]]


def _house_usd(scene: Scene) -> Path:
    """The house's USD with its material and texture paths pointing into its own folder, matching
    file names regardless of case; written to the process-local conversion directory."""
    folder = generated_directory() / hashlib.sha256(str(scene.directory).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scene.usd"
    if path.is_file():
        return path
    files = {
        f.relative_to(scene.directory).as_posix().lower(): f for f in scene.directory.rglob("*")
    }

    def repair(asset: str) -> str:
        for folder in ("props/", "textures/"):
            if folder in asset:
                local = asset[asset.index(folder) :]
                match = files.get(local.lower())
                return str(match or (scene.directory / local))
        return asset

    layer = Sdf.Layer.FindOrOpen(str(scene.source))
    UsdUtils.ModifyAssetPaths(layer, repair)
    layer.Export(str(path))
    return path


def _spawn_scene(scene: Scene, material: sim_utils.RigidBodyMaterialCfg) -> None:
    """Put the house at `SCENE_PRIM`, turned Z up, ceilings hidden, its instances expanded, and
    make the meshes of `SCENE_SOLIDS` static triangle-mesh colliders with `material`."""
    if scene.name == "household":
        from .household_usd import room

        room(sim_utils.get_current_stage())
        return
    cfg = sim_utils.UsdFileCfg(usd_path=str(_house_usd(scene)))
    half = math.sqrt(0.5)
    cfg.func(SCENE_PRIM, cfg, translation=scene.translation, orientation=(half, 0.0, 0.0, half))
    stage = sim_utils.get_current_stage()
    # Meshes inside instances cannot take collision APIs; expand them, nested ones included.
    while instances := [
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(SCENE_PRIM), Usd.TraverseInstanceProxies())
        if p.IsInstance()
    ]:
        for prim in instances:
            prim.SetInstanceable(False)
    sim_utils.set_prim_visibility(stage.GetPrimAtPath(f"{SCENE_PRIM}/ceilings"), False)
    material_path = f"{SCENE_PRIM}/physics_material"
    material.func(material_path, material)
    for group in SCENE_SOLIDS:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{SCENE_PRIM}/{group}")):
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).GetPointsAttr().Get():
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
                sim_utils.bind_physics_material(prim.GetPath(), material_path)


def _item_usd(model: int, asset_directory: Path) -> Path:
    """SIMPLE's USD of GraspNet-1B object `model` as one rigid body, written to the process-local conversion directory: metres and Z up, as its points already are, with the rigid body and the
    model's mass moved to its root."""
    source = graspnet_usd(model, asset_directory)
    folder = generated_directory() / hashlib.sha256(str(asset_directory).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model:03d}_rigid.usd"
    if path.is_file():
        return path
    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = stage.GetDefaultPrim()
    mass = 0.0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass += UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            prim.RemoveAPI(UsdPhysics.MassAPI)
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    UsdUtils.ModifyAssetPaths(stage.GetRootLayer(),
        lambda asset: str((source.parent / asset).resolve()) if asset.startswith("./") else asset)
    stage.GetRootLayer().Export(str(path))
    return path


def _model_bounds(usd: Path) -> np.ndarray:
    """Bounding box of a model in its own frame: rows min, max. Shape (2, 3), m."""
    stage = Usd.Stage.Open(str(usd))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeUntransformedBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return np.array([box.GetMin(), box.GetMax()], dtype=float)


def _spawn_box(box: Box, material: sim_utils.RigidBodyMaterialCfg) -> None:
    cfg = sim_utils.CuboidCfg(
        size=box.size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=box.color),
    )
    cfg.func(f"{TASK_PRIM}/{box.name}", cfg, translation=box.pos)


class IsaacCamera:
    """The head camera: an Isaac camera whose world pose is set from the head link before each
    capture. Call from the simulation's thread only.

    Args:
        robot: Robot config; its `camera` gives the mount, image size, field of view and range.
    """

    def __init__(self, robot: RobotConfig) -> None:
        c = robot.camera
        self._config = c
        self._mount = np.eye(4)
        self._mount[:3, :3] = rpy_matrix(c.rpy)
        self._mount[:3, 3] = c.pos
        self._camera = Camera(
            CameraCfg(
                prim_path=f"/World/{c.name}",
                width=c.width,
                height=c.height,
                update_period=0.0,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=c.intrinsics().ravel().tolist(),
                    width=c.width,
                    height=c.height,
                    clipping_range=(c.near, c.far),
                ),
            )
        )

    def capture(
        self, sim: sim_utils.SimulationContext, world_head: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render with the camera's body at `world_head` (body-to-world, shape (4, 4)).

        Returns:
            RGB, shape (H, W, 3) uint8; depth along the optical axis, shape (H, W) float32, m,
            NaN where nothing is within `far`.
        """
        pose = world_head @ self._mount
        quat = matrix_to_quat(pose[:3, :3])
        device = sim.device
        self._camera.set_world_poses(
            positions=torch.tensor(pose[None, :3, 3], device=device, dtype=torch.float32),
            orientations=torch.tensor(
                np.r_[quat[1:], quat[0]][None], device=device, dtype=torch.float32
            ),
            convention="ros",
        )
        for _ in range(RENDER_PASSES):
            sim.render()
        self._camera.update(0.0, force_recompute=True)
        out = self._camera.data.output
        rgb = out["rgb"].torch[0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
        depth = out["distance_to_image_plane"].torch[0, :, :, 0].detach().cpu().numpy()
        depth = depth.astype(np.float32)
        depth[~(depth < self._config.far)] = np.nan
        return rgb, depth






class IsaacWorld:
    """Flat ground, a task's scene, boxes and items, and the T2 in Isaac Lab: the world the owner
    runs.

    The Isaac app must be running (headless, cameras enabled when `camera` is True) before this is
    built, and every method must be called from the app's thread. Each `step` drives the policy's
    joints toward the given targets with Booster's PD, applied as joint torque; the finger joints
    follow `set_finger_targets` through PhysX position drives.

    Args:
        robot: Robot config.
        joint_names: The policy's joints in the order `read` and `step` use: all but the
            fingers.
        task: The task; its scene, boxes and items are added to the ground, and its start is the
            default start.
        camera: Whether to simulate the head camera.
        device: Torch device of the physics, matching the app's; CPU PhysX steps a single robot
            several times faster than GPU PhysX. Cameras render on the GPU either way.

    Attributes:
        joint_names: The robot's joints, in the order given.
        task: The task.
        gains: Joint drive gains and motor limits, `joint_names` order.
        camera: The head camera, or None.
    """

    def __init__(
        self,
        robot: RobotConfig,
        joint_names: Sequence[str],
        task: SceneTask,
        camera: bool = True,
        device: str = "cpu",
    ) -> None:
        self.robot = robot
        self.joint_names: tuple[str, ...] = tuple(joint_names)
        self.task = task
        self.gains = robot.pd_gains(self.joint_names)
        self.default_joint_pos = robot.keyframe_joint_pos(self.joint_names)
        """Keyframe joint positions, `joint_names` order. Shape (num_joints,), rad."""
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=robot.simulation.friction,
            dynamic_friction=robot.simulation.friction,
            restitution=0.0,
        )
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=robot.timestep, device=device, physics_material=material,
                                    enable_scene_query_support=task.spec["version"] == 3)
        )
        """The Isaac simulation context."""
        # Author the same static Z-up collision plane locally: the default grid USD
        # otherwise requires a live NVIDIA asset-server connection during replay.
        ground = UsdGeom.Plane.Define(sim_utils.get_current_stage(), "/World/Ground")
        ground.CreateAxisAttr("Z")
        ground.CreateWidthAttr(100.0)
        ground.CreateLengthAttr(100.0)
        ground.CreateDisplayColorAttr([(0.4, 0.4, 0.4)])
        self.household = task.spec["version"] == 3
        if not self.household:
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material.func("/World/GroundMaterial", material)
        if not self.household:
            sim_utils.bind_physics_material("/World/Ground", "/World/GroundMaterial")
        light = sim_utils.DomeLightCfg(intensity=450.0 if self.household else 2000.0)
        light.func("/World/Light", light)
        if task.scene:
            _spawn_scene(task.scene, material)
        for box in task.boxes:
            _spawn_box(box, material)
        self._items: dict[str, RigidObject] = {}
        self._item_start: dict[str, list[float]] = {}
        self._item_bounds: dict[str, np.ndarray] = {}
        if task.items:
            material.func(f"{TASK_PRIM}/physics_material", material)
        for item in task.items:
            self._spawn_item(item)
        self._articulated = {}
        self._articulated_start = {}
        for obj in task.articulated:
            self._spawn_articulated(obj)
        self._robot = Articulation(t2_articulation_cfg(robot, self.joint_names))
        self.camera = IsaacCamera(robot) if camera else None
        self.measurements = None
        if self.household:
            from .household_physics import Measurements

            self.measurements = Measurements(self)
        self.sim.reset()
        for obj in task.articulated:
            art = self._articulated[obj.name]
            if self.household:
                from .household_models import model

                floating = not model(obj.model).fixed
            else:
                floating = articulated_model(obj.model, task.asset_directory).floating
            if floating:
                self._articulated_start[obj.name] = art.data.root_link_pose_w.torch.clone()
        isaac_names = list(self._robot.joint_names)
        fingers = robot.fingers.joints
        if sorted(isaac_names) != sorted(self.joint_names + fingers):
            raise ValueError("the articulation's joints differ from joint_names and the fingers")
        self._to_isaac = np.array([isaac_names.index(n) for n in self.joint_names])
        self._finger_idx = [isaac_names.index(n) for n in fingers]
        self._head = self._robot.body_names.index(robot.camera.body)
        self._kinematics: dict[tuple[str, ...], RobotKinematics] = {}
        self._device = device
        self._time = 0.0

    def _spawn_item(self, item: Item) -> None:
        if self.household:
            from .household_usd import make_asset

            usd = make_asset(item.model)
        else:
            usd = _item_usd(item.model, self.task.asset_directory)
        bounds = _model_bounds(usd)
        pos = (*item.xy, item.support + (0. if item.fixed else ITEM_CLEARANCE) if self.household else item.support - bounds[0, 2] + ITEM_CLEARANCE)
        quat = (0.0, 0.0, math.sin(item.yaw / 2), math.cos(item.yaw / 2))
        path = f"{TASK_PRIM}/{item.name}"
        self._items[item.name] = RigidObject(
            RigidObjectCfg(
                prim_path=path,
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
            )
        )
        for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath(path)):
            if self.household and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if item.fixed:
                    PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr(False)
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(item.fixed)
            if not self.household and prim.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.bind_physics_material(prim.GetPath(), f"{TASK_PRIM}/physics_material")
        self._item_start[item.name] = [*pos, *quat]
        self._item_bounds[item.name] = bounds

    def _spawn_articulated(self, obj: Articulated) -> None:
        if self.household:
            from .household_models import model as definition
            from .household_usd import make_asset

            definition = definition(obj.model)
            self._articulated[obj.name] = Articulation(ArticulationCfg(
                prim_path=f"{TASK_PRIM}/{obj.name}",
                spawn=sim_utils.UsdFileCfg(usd_path=str(make_asset(obj.model))),
                init_state=ArticulationCfg.InitialStateCfg(pos=(*obj.xy,obj.support),
                    rot=(0.,0.,math.sin(obj.yaw/2),math.cos(obj.yaw/2)), joint_pos=dict(obj.joints)),
                actuators={"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.,
                    damping={j.name:j.damping for j in definition.joints},
                    friction={j.name:j.friction for j in definition.joints},
                    dynamic_friction={j.name:j.friction for j in definition.joints},
                    armature=.0001)}))
            return
        model = articulated_model(obj.model, self.task.asset_directory)
        # Places the model's frame, not its base link.
        pos = (*obj.xy, obj.support - model.bounds[0, 2] + ITEM_CLEARANCE)
        quat = (0.0, 0.0, math.sin(obj.yaw / 2), math.cos(obj.yaw / 2))
        dynamics = model.joints
        self._articulated[obj.name] = Articulation(
            ArticulationCfg(
                prim_path=f"{TASK_PRIM}/{obj.name}",
                spawn=sim_utils.MjcfFileCfg(
                    asset_path=str(model.mjcf),
                    usd_dir=str(generated_directory() / f"articulated-{obj.model:03d}"),
                    fix_base=not model.floating,
                    self_collision=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=quat, joint_pos=obj.joints),
                actuators={
                    "joints": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=0.0,
                        damping={n: d.damping for n, d in dynamics.items()},
                        friction={n: d.friction for n, d in dynamics.items()},
                        dynamic_friction={n: d.friction for n, d in dynamics.items()},
                        armature={
                            n: max(d.armature, 1e-4)
                            for n, d in dynamics.items()
                        },
                    )
                },
            )
        )
        # MJCF conversion does not retain SIMPLE's USD materials. Give the fixture
        # a visible matte finish so its silhouette does not disappear into the ground.
        path = f"{TASK_PRIM}/{obj.name}"
        material_path = f"{path}/task_material"
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.38, 0.52), roughness=0.6)
        material.func(material_path, material)
        sim_utils.bind_visual_material(path, material_path, stronger_than_descendants=True)

    @property
    def physics_dt(self) -> float:
        """Physics timestep, s."""
        return self.robot.timestep

    @property
    def time(self) -> float:
        """Simulated time since reset, in seconds."""
        return self._time

    def step(self, target: np.ndarray) -> None:
        """One physics step: Booster PD torque toward `target` (joint positions, `joint_names`
        order, rad), then a PhysX step without rendering."""
        q = _np(self._robot.data.joint_pos)[self._to_isaac]
        qd = _np(self._robot.data.joint_vel)[self._to_isaac]
        torque = np.zeros(self._robot.num_joints)
        torque[self._to_isaac] = booster_pd_torque(target, q, qd, self.gains)
        self._robot.set_joint_effort_target_index(
            target=torch.tensor(torque[None], device=self._device, dtype=torch.float32),
        )
        self._robot.write_data_to_sim()
        self.sim.step(render=False)
        self._robot.update(self.physics_dt)
        for item in (*self._items.values(), *self._articulated.values()):
            item.update(self.physics_dt)
        self._time += self.physics_dt
        if self.measurements is not None:
            self.measurements.refresh()
            for name, obj in self.task.spec["articulated"].items():
                if obj["model"] == "lamp":
                    self.measurements.lamp_light(name)
            self.task.evaluate(self)

    def set_finger_targets(self, positions: np.ndarray) -> None:
        """Position targets of the finger joints, `robot.fingers.joints` order, rad; they hold
        until changed or reset."""
        self._robot.set_joint_position_target_index(
            target=torch.tensor(positions[None], device=self._device, dtype=torch.float32),
            joint_ids=self._finger_idx,
        )

    def read_fingers(self) -> tuple[np.ndarray, np.ndarray]:
        """Finger joint positions (rad) and velocities (rad/s), `robot.fingers.joints` order."""
        d = self._robot.data
        return _np(d.joint_pos)[self._finger_idx], _np(d.joint_vel)[self._finger_idx]

    def read(self) -> RobotState:
        """The robot's proprioception now, as its encoders and IMU report it."""
        d = self._robot.data
        return RobotState(
            joint_pos=_np(d.joint_pos)[self._to_isaac],
            joint_vel=_np(d.joint_vel)[self._to_isaac],
            base_quat=_wxyz(_np(d.root_link_quat_w)),
            base_ang_vel=_np(d.root_link_ang_vel_b),
        )

    def root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Root link origin position (3,) m and orientation wxyz (4,), world frame."""
        d = self._robot.data
        return _np(d.root_link_pos_w), _wxyz(_np(d.root_link_quat_w))

    def root_velocity(self) -> np.ndarray:
        """Root link velocity: linear (3,) m/s in the world frame, then angular (3,) rad/s in the
        root body frame. Shape (6,)."""
        d = self._robot.data
        return np.r_[_np(d.root_link_lin_vel_w), _np(d.root_link_ang_vel_b)]



    def link_poses(self, links: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """World poses of URDF links, simulator truth: positions (n, 3) m and orientations wxyz
        (n, 4), `links` order. Links merged away in Isaac, such as the palms and the fingertip
        frames, are included."""
        key = tuple(links)
        if key not in self._kinematics:
            joints = self.joint_names + self.robot.fingers.joints
            self._kinematics[key] = RobotKinematics(self.robot, joints, key)
        pos, quat = self.root_pose()
        q = np.r_[self.read().joint_pos, self.read_fingers()[0]]
        positions, quats = self._kinematics[key].frame_poses(q, quat)
        return pos + positions, quats


    def head_pose(self) -> np.ndarray:
        """Native head link world transform, without rendering or stepping."""
        link = self._robot.data.body_link_pose_w.torch[0, self._head].detach().cpu().numpy()
        head = np.eye(4)
        x, y, z, w = link[3:].astype(float)
        head[:3, :3] = quat_to_matrix(np.array([w, x, y, z]))
        head[:3, 3] = link[:3]
        return head

    def head_camera_pose(self) -> np.ndarray:
        """Head optical frame in world coordinates, including the fixed mount."""
        config = self.robot.camera
        mount = np.eye(4)
        mount[:3, :3] = rpy_matrix(config.rpy)
        mount[:3, 3] = config.pos
        return self.head_pose() @ mount

    def capture_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """The head camera now: RGB and depth, as `IsaacCamera.capture`.

        Raises:
            RuntimeError: The world has no camera.
        """
        if self.camera is None:
            raise RuntimeError("this world simulates no camera")
        return self.camera.capture(self.sim, self.head_pose())

    def item_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose of item `name`'s model origin, simulator truth: position (3,) m and
        orientation wxyz (4,)."""
        d = self._items[name].data
        return _np(d.root_link_pos_w), _wxyz(_np(d.root_link_quat_w))

    def articulated_pose(self, name):
        data = self._articulated[name].data
        return _np(data.root_link_pos_w), _wxyz(_np(data.root_link_quat_w))

    def articulated_joints(self, name):
        obj = self._articulated[name]
        return dict(zip(obj.joint_names, _np(obj.data.joint_pos)))

    def articulated_state(self, name):
        obj = self._articulated[name]
        data = obj.data
        return {"pose": np.concatenate(self.articulated_pose(name)).tolist(),
                "velocity": np.r_[_np(data.root_link_lin_vel_w), _np(data.root_link_ang_vel_w)].tolist(),
                "joints": self.articulated_joints(name),
                "joint_velocities": dict(zip(obj.joint_names, _np(data.joint_vel)))}

    def item_velocity(self, name: str) -> np.ndarray:
        data = self._items[name].data
        return np.r_[_np(data.root_link_lin_vel_w), _np(data.root_link_ang_vel_w)]

    def item_speed(self, name: str) -> float:
        """Linear speed of item `name`'s model origin, simulator truth, m/s."""
        return float(np.linalg.norm(_np(self._items[name].data.root_link_lin_vel_w)))

    def item_corners(self, name: str) -> np.ndarray:
        """Corners of item `name`'s bounding box, world frame, simulator truth. Shape (8, 3), m."""
        lo, hi = self._item_bounds[name]
        local = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        pos, quat = self.item_pose(name)
        return pos + local @ quat_to_matrix(quat).T

    def item_gap(self, name: str, points: np.ndarray) -> float:
        """Smallest distance from world `points` (shape (n, 3), m) to item `name`'s bounding box,
        simulator truth, m; 0 for a point inside."""
        pos, quat = self.item_pose(name)
        local = (np.asarray(points) - pos) @ quat_to_matrix(quat)
        lo, hi = self._item_bounds[name]
        excess = np.maximum(np.maximum(lo - local, local - hi), 0.0)
        return float(np.linalg.norm(excess, axis=1).min())

    def reset(self) -> None:
        """Put the robot in its keyframe pose at the task's start, at rest with the fingers open
        and targeted open, put the items back where they start, and reset the task. The root
        height is the keyframe's."""
        x, y, yaw = self.task.start
        root = self.robot.keyframe_root_pos()
        f = {"device": self._device, "dtype": torch.float32}
        pose = [x + root[0], y + root[1], root[2], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        self._robot.write_root_pose_to_sim_index(root_pose=torch.tensor([pose], **f))
        self._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, **f))
        q = np.zeros(self._robot.num_joints)
        q[self._to_isaac] = self.default_joint_pos
        self._robot.write_joint_position_to_sim_index(position=torch.tensor(q[None], **f))
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(1, self._robot.num_joints, **f))
        self._robot.reset()
        self.set_finger_targets(np.zeros(len(self._finger_idx)))
        self._robot.update(0.0)
        for name, item in self._items.items():
            item.write_root_pose_to_sim_index(root_pose=torch.tensor([self._item_start[name]], **f))
            if not self.household or not self.task.spec["items"][name]["fixed"]:
                item.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, **f))
            item.reset()
            item.update(0.0)
        for obj in self.task.articulated:
            art = self._articulated[obj.name]
            # A fixed base's pose must remain at the world joint's authored transform.
            if obj.name in self._articulated_start:
                art.write_root_pose_to_sim_index(root_pose=self._articulated_start[obj.name])
                art.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, **f))
            q = [obj.joints.get(n, 0.0) for n in art.joint_names]
            art.write_joint_state_to_sim_index(
                position=torch.tensor([q], **f), velocity=torch.zeros(1, len(q), **f))
            art.reset()
            art.update(0.0)
        self._time = 0.0
        if self.measurements is not None:
            self.measurements.reset()
        self.task.reset()
