"""Isaac Lab / PhysX owner of the ABC-derived dual YAM scene.

Only physical_profiles from the prepared owner bundle determine geometry. Public
profiles are validated for hardware identity but never used to position anything.
"""

import copy
import sys
import traceback
from pathlib import Path

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

from .assets import load_manifest, sha256


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def array(value):
    return tensor(value).detach().cpu().numpy().copy()


RGB_SUPERSAMPLING = 2


def calibration_render_cfg(sim):
    """Keep marker detail without imposing offline path tracing on live views."""
    return sim.RenderCfg(
        antialiasing_mode="DLAA",
        enable_dlssg=False,
        enable_dl_denoiser=True,
        carb_settings={
            "/rtx/rendermode": "RealTimePathTracing",
            "/rtx/post/motionblur/enabled": False,
        },
    )


def render_intrinsics(projection):
    scale = RGB_SUPERSAMPLING
    return [
        [scale * projection["fx"], 0, scale * (projection["cx"] + 0.5) - 0.5],
        [0, scale * projection["fy"], scale * (projection["cy"] + 0.5) - 0.5],
        [0, 0, 1],
    ]


def pose_matrix(pose):
    return matrix(pose)


class Native:
    dt = 0.002
    interval = 0.034

    def __init__(self, manifest, task, profiles):
        manifest = Path(manifest).resolve()
        facts = load_manifest(manifest)
        task.validate_profiles(profiles, facts["physical_profiles"])
        self.physical = copy.deepcopy(facts["physical_profiles"])
        self.asset_sha = sha256(manifest)
        self.task = task
        self.app = self.sim = None
        self.robots, self.cameras, self.contacts = {}, {}, {}
        self.home = {
            side: [
                self.physical[side + "_arm"]["intrinsics"]["home"][f"{side}_joint{i}"]
                for i in range(1, 7)
            ]
            for side in ("left", "right")
        }
        from roboenv_runtime.adapters.gpu import launch_isaaclab

        self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=True)
        try:
            self._create(manifest.parent)
        except BaseException:
            self.close()
            raise

    def _create(self, directory):
        import isaaclab.sim as sim
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import Camera, CameraCfg
        from pxr import Usd, UsdPhysics

        self.torch = torch
        self.sim = sim.SimulationContext(
            sim.SimulationCfg(dt=self.dt, device="cuda:0", render=calibration_render_cfg(sim))
        )
        ground = sim.GroundPlaneCfg()
        ground.func("/World/Ground", ground)
        light = sim.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)
        converted = {}
        for name in ("station", "left", "right"):
            cfg = sim.MjcfConverterCfg(
                asset_path=str(directory / (name + ".xml")),
                usd_dir=str(directory / "usd" / self.asset_sha / name),
                fix_base=True,
                make_instanceable=False,
                self_collision=True,
            )
            converted[name] = sim.MjcfConverter(cfg).usd_path
        station = sim.UsdFileCfg(usd_path=converted["station"])
        station.func("/World/Station", station)
        # MJCF conversion may author a rigid root even for static-only geometry.
        stage = self.sim.stage
        prims = list(Usd.PrimRange(stage.GetPrimAtPath("/World/Station")))
        for prim in reversed(prims):
            if prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
                continue
            for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
        self.flange_paths = {}
        for side in ("left", "right"):
            mount = self.physical[side + "_arm"]["extrinsics"]["mount"]
            home = {f"{side}_joint{i + 1}": q for i, q in enumerate(self.home[side])}
            home.update({side + "_left_finger": 0.0475, side + "_right_finger": -0.0475})
            root = "/World/" + side.capitalize()
            cfg = ArticulationCfg(
                prim_path=root,
                spawn=sim.UsdFileCfg(
                    usd_path=converted[side],
                    activate_contact_sensors=True,
                    articulation_props=sim.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=4,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(mount["translation_m"]),
                    rot=tuple(mount["rotation_xyzw"]),
                    joint_pos=home,
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "large": ImplicitActuatorCfg(
                        joint_names_expr=[side + "_joint[1-3]"],
                        stiffness=40.0,
                        damping=2.5,
                        effort_limit_sim=28.0,
                    ),
                    "small": ImplicitActuatorCfg(
                        joint_names_expr=[side + "_joint[4-6]"],
                        stiffness={side + "_joint4": 20.0, side + "_joint[56]": 10.0},
                        damping={side + "_joint4": 0.5, side + "_joint[56]": 1.0},
                        effort_limit_sim=10.0,
                    ),
                    "finger": ImplicitActuatorCfg(
                        joint_names_expr=[side + "_left_finger"], stiffness=400.0, damping=14.0
                    ),
                    "passive": ImplicitActuatorCfg(
                        joint_names_expr=[side + "_right_finger"], stiffness=0.0, damping=0.0
                    ),
                },
            )
            self.robots[side] = Articulation(cfg)
            self.flange_paths[side] = unique_prim(stage, root, side + "_link_6")
            couple_fingers(stage, root, side)
            apply_body_dynamics(stage, root, directory / (side + ".xml"))
        self.scene = self.task.scene_type(self)
        self.scene.build()
        for name in ("camera_top", "camera_left", "camera_right"):
            color = self.physical[name]["intrinsics"]["color"]
            p = color["projection"]
            self.cameras[name] = Camera(
                CameraCfg(
                    prim_path="/World/" + name,
                    width=color["width"] * RGB_SUPERSAMPLING,
                    height=color["height"] * RGB_SUPERSAMPLING,
                    data_types=["rgb"],
                    spawn=sim.PinholeCameraCfg(
                        focal_length=24.0,
                        horizontal_aperture=color["width"] * 24 / p["fx"],
                        vertical_aperture=color["height"] * 24 / p["fy"],
                        clipping_range=(0.01, 10.0),
                    ),
                )
            )
        self.sim.reset()
        self.ids, self.flange_ids = {}, {}
        for side, robot in self.robots.items():
            robot.update(self.dt)
            self.ids[side] = {name: i for i, name in enumerate(robot.joint_names)}
            self.flange_ids[side] = robot.body_names.index(side + "_link_6")
        # Set exact OpenCV principal point (ABC uses (w-1)/2, (h-1)/2).
        for name, camera in self.cameras.items():
            p = self.physical[name]["intrinsics"]["color"]["projection"]
            camera.set_intrinsic_matrices(
                torch.tensor(
                    [render_intrinsics(p)],
                    device="cuda:0",
                    dtype=torch.float32,
                )
            )

    def reset(self, seed):
        self.seed, self.steps = seed, 0
        for robot in self.robots.values():
            robot.write_root_pose_to_sim_index(
                root_pose=tensor(robot.data.default_root_pose).clone()
            )
            robot.write_root_velocity_to_sim_index(
                root_velocity=tensor(robot.data.default_root_vel).clone()
            )
            robot.write_joint_position_to_sim_index(
                position=tensor(robot.data.default_joint_pos).clone()
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=tensor(robot.data.default_joint_vel).clone()
            )
            robot.reset()
            robot.update(self.dt)
        for contact in self.contacts.values():
            contact.reset()

    def advance(self, targets):
        # Commands stay constant over the control interval. Stage them once;
        # keep actuator writes, physics and sensor updates at every substep.
        for side, robot in self.robots.items():
            values = {f"{side}_joint{i + 1}": q for i, q in enumerate(targets[side + "_arm"])}
            opening = targets[side + "_gripper"][0] * 0.0475
            values.update({side + "_left_finger": opening, side + "_right_finger": -opening})
            q = tensor(robot.data.joint_pos).clone()
            limits = array(robot.data.joint_pos_limits)[0]
            for name, value in values.items():
                index = self.ids[side][name]
                q[0, index] = float(np.clip(value, *limits[index]))
            robot.set_joint_position_target_index(target=q)
        for _ in range(17):
            for robot in self.robots.values():
                robot.write_data_to_sim()
            self.sim.step(render=False)
            for robot in self.robots.values():
                robot.update(self.dt)
            for contact in self.contacts.values():
                contact.update(self.dt)
        self.steps += 1

    def flange_transform(self, side):
        robot = self.robots[side]
        return pose_matrix(array(robot.data.body_link_pose_w)[0, self.flange_ids[side]])

    def read_device(self, name):
        side = name.split("_")[0]
        robot = self.robots[side]
        state = {"timestamp": self.steps * self.interval, "faults": []}
        q = array(robot.data.joint_pos)[0]
        if not np.isfinite(q).all():
            state["faults"] = [{"code": "nonfinite_joint_state"}]
        if name.endswith("_arm"):
            ids = [self.ids[side][f"{side}_joint{i}"] for i in range(1, 7)]
            return {
                **state,
                "positions": q[ids].tolist(),
                "velocities": array(robot.data.joint_vel)[0, ids].tolist(),
                "efforts": [],
            }
        if name.endswith("_gripper"):
            left, right = [
                self.ids[side][side + suffix] for suffix in ("_left_finger", "_right_finger")
            ]
            return {**state, "aperture_m": max(0.0, 0.0008 + float(q[left] - q[right]))}
        if name.endswith("_contact"):
            force = array(self.contacts[side].data.force_matrix_w)[0].reshape(-1, 3).sum(axis=0)
            tcp = self.physical[side + "_gripper"]["intrinsics"]["frames"]["tcp"]
            rotation = (
                self.flange_transform(side)[:3, :3]
                @ matrix([*tcp["translation_m"], *tcp["rotation_xyzw"]])[:3, :3]
            )
            return {
                **state,
                "force": dict(zip("xyz", (rotation.T @ force).tolist(), strict=True)),
                "torque": None,
            }
        raise ValueError("Unknown YAM device")

    def capture(self, name, directory):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        from ...common.observations import camera_state

        camera = self.cameras[name]
        mount = self.physical[name]["extrinsics"]["mount"]
        transform = matrix([*mount["translation_m"], *mount["rotation_xyzw"]])
        if name != "camera_top":
            transform = self.flange_transform(name.removeprefix("camera_")) @ transform
        pose = matrix_pose(transform)
        camera.set_world_poses(
            positions=self.torch.tensor([pose[:3]], device="cuda:0", dtype=self.torch.float32),
            orientations=self.torch.tensor(
                [pose[3:]], device="cuda:0", dtype=self.torch.float32
            ),
            convention="ros",
        )
        for _ in range(3):
            self.sim.render()
            ensure_isaac_rtx_render_update()
        color = self.physical[name]["intrinsics"]["color"]
        return camera_state(
            camera, directory, self.steps * self.interval,
            rgb_size=(color["width"], color["height"]),
        )

    def close(self):
        if self.app is not None:
            if sys.exc_info()[1] is not None:
                traceback.print_exc()
            self.app.close(exit_code=1 if sys.exc_info()[1] else 0)
            self.app = None


def unique_prim(stage, root, name):
    from pxr import Usd

    matches = [
        str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.GetName() == name
    ]
    if len(matches) != 1:
        raise ValueError(f"MJCF import must preserve one {name}; got {matches}")
    return matches[0]


def body_dynamics(path):
    """Resolve MJCF inference and compiler bounds in the pinned source compiler."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    result = {}
    for index in range(1, model.nbody):  # Exclude the nonphysical world body.
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        compensation = float(model.body_gravcomp[index])
        if not name or compensation not in (0.0, 1.0):
            raise ValueError("YAM bodies require names and zero or full gravity compensation")
        result[name] = {
            "mass": float(model.body_mass[index]),
            "center": model.body_ipos[index].tolist(),
            "axes_wxyz": model.body_iquat[index].tolist(),
            "inertia": model.body_inertia[index].tolist(),
            "disable_gravity": compensation == 1.0,
        }
    return result


def apply_body_dynamics(stage, root, path):
    """Preserve compiled MJCF inertials before PhysX initializes the articulation.

    The importer omits compiler boundmass/boundinertia, leaving empty fixed
    camera frames with a 1 kg PhysX fallback instead of their source 1 g mass.
    """
    from pxr import Gf, PhysxSchema, UsdPhysics

    dynamics = body_dynamics(path)
    bodies = {name: stage.GetPrimAtPath(unique_prim(stage, root, name)) for name in dynamics}
    if any(not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in bodies.values()):
        raise ValueError("MJCF import must preserve the YAM rigid-body mapping")
    for name, facts in dynamics.items():
        prim = bodies[name]
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(facts["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*facts["center"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*facts["inertia"]))
        w, x, y, z = facts["axes_wxyz"]
        mass.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(facts["disable_gravity"])


def couple_fingers(stage, root, side):
    """Explicit q_right = -q_left constraint, independent of MJCF equality support."""
    from pxr import PhysxSchema, UsdPhysics

    master = unique_prim(stage, root, side + "_left_finger")
    slave = stage.GetPrimAtPath(unique_prim(stage, root, side + "_right_finger"))
    if not slave.IsA(UsdPhysics.PrismaticJoint):
        raise ValueError("YAM finger must import as a prismatic joint")
    # For a single-DOF joint, PhysX ignores the schema's rotX axis token.
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(slave, "rotX")
    mimic.CreateReferenceJointRel().SetTargets([master])
    mimic.CreateReferenceJointAxisAttr("rotX")
    mimic.CreateGearingAttr(1.0)
    mimic.CreateOffsetAttr(0.0)
