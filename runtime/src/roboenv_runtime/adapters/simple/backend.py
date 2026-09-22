"""SIMPLE native world: fixed physics cadence, retained goals and private evidence."""

import copy
import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
from roboenv.provisioning.simple_assets import digest, verify_assets
from roboenv.provisioning.sources import status
from roboenv.tasks.simple import entries

from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose
from roboenv_runtime.paths import repo_root

from .control import Control
from .profiles import CONTROL_INTERVAL_S


class SimpleEnv:
    control_interval = CONTROL_INTERVAL_S

    def __init__(
        self,
        *,
        env_id="simple/G1WholebodyXMovePickTeleop-v0",
        realtime=True,
        arm_gravity_compensation=True,
        sim_mode="mujoco_isaac",
        dr_level=0,
        initialization=None,
        stabilize_steps=300,
    ):
        if type(arm_gravity_compensation) is not bool:
            raise ValueError("arm_gravity_compensation must be a boolean")
        self.arm_gravity_compensation = arm_gravity_compensation
        root = repo_root()
        self.root = root
        self.env = None
        self.realtime = realtime
        self.model = json.loads(Path(__file__).with_name("g1.json").read_text())
        self.control = Control(
            self, self.model, realtime=realtime
        )
        if sim_mode not in {"mujoco", "mujoco_isaac"}:
            raise ValueError("SIMPLE supports mujoco or mujoco_isaac")
        if type(dr_level) is not int or dr_level not in {0, 1, 2}:
            raise ValueError("SIMPLE DR level must be 0, 1 or 2")
        if type(stabilize_steps) is not int or not 100 <= stabilize_steps <= 1000:
            raise ValueError("stabilize_steps must be between 100 and 1000")
        tasks = {entry["env_id"]: entry for entry in entries(root / "third_party/SIMPLE")}
        if env_id not in tasks:
            raise ValueError("Only registered SIMPLE decoupled G1 teleop tasks are supported")
        if not status(root, ["simple"])["ready"]:
            raise ValueError("SIMPLE source pins/patches changed; run roboenv sources status")
        data = verify_assets(root)
        if digest(data / "robots/g1_sonic/g1_29dof_with_hand.xml") != self.model["robot_sha256"]:
            raise ValueError("SIMPLE robot differs from generated catalog")
        self.env_id, self.sim_mode, self.dr_level = env_id, sim_mode, dr_level
        self.initial_state = copy.deepcopy(initialization)
        self.stabilize_steps = stabilize_steps
        self.max_steps = None if realtime else tasks[env_id]["metadata"]["max_episode_steps"]
        os.environ["SIMPLE_DATA_DIR"] = str(data)
        os.environ["SIMPLE_OFFLINE"] = "1"
        os.environ.setdefault("MUJOCO_GL", "egl")
        if sim_mode == "mujoco_isaac":
            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        self.steps, self.reward, self.completed, self.done = 0, 0.0, False, False
        self.frames, self.events, self.images = [], [], {}

    @property
    def sample_time_s(self):
        return self.steps * self.control_interval

    def reset(self, seed):
        if self.env is not None:
            raise ValueError("SIMPLE reset requires a fresh worker")
        import gymnasium as gym
        import simple.envs  # noqa: F401
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        from .wbc import WholeBodyControl

        random.seed(seed)
        np.random.seed(seed)
        config = SimLoopConfig(enable_waist=True).load_wbc_yaml()
        config.update(ENV_NAME="simple", IN_PROCESS_WBC=True, ENABLE_ELASTIC_BAND=False)
        config["ARM_GRAVITY_COMPENSATION"] = self.arm_gravity_compensation
        try:
            self.env = gym.make(
                self.env_id,
                sim_mode=self.sim_mode,
                headless=True,
                sonic_config=config,
                render_hz=50,
                dr_level=self.dr_level,
            ).unwrapped
            self.task = self.env.task
            options = (
                {"state_dict": copy.deepcopy(self.initial_state)} if self.initial_state else None
            )
            self.env.reset(seed=seed, options=options)
            self.robot = self.task.robot
            self.mjmodel, self.mjdata = self.env.mjModel, self.env.mjData
            self.wbc = WholeBodyControl(self.robot)
            if self.env.control_decimal != 4 or not np.isclose(self.mjmodel.opt.timestep, 0.005):
                raise ValueError("SIMPLE requires four 5ms physics substeps")
            self._check_geometry()
            self.initial_state = json.loads(
                json.dumps(
                    self.task.state_dict(),
                    default=lambda value: (
                        value.tolist() if isinstance(value, np.ndarray) else value.item()
                    ),
                )
            )
            references = self.wbc.initial_references()
            for name in self.control.names[1:]:
                part = self.model["parts"][name]
                self.control.targets[name] = [float(references[j["name"]]) for j in part["joints"]]
            self.initialization_frames = 0
            stabilized = False
            for _ in range(self.stabilize_steps):
                self._step(self.control.targets)
                self.initialization_frames += 1
                if self.robot.stabilized:
                    stabilized = True
                    break
            if not stabilized:
                raise RuntimeError(
                    "SIMPLE standing initialization did not stabilize within its bound"
                )
            from .cameras import validate_isaac

            validate_isaac(self)
            # Initialization is outside the episode budget. Preserve native task state.
            self.steps, self.reward, self.completed, self.done = 0, 0.0, False, False
            self.images = {}
            from roboenv_runtime.adapters.native_record import implementation_identity

            self.implementation = implementation_identity("simple")
            self.initialization = {
                "env_id": self.env_id,
                "seed": seed,
                "dr_level": self.dr_level,
                "sim_mode": self.sim_mode,
                "realtime": self.realtime,
                "arm_gravity_compensation": self.arm_gravity_compensation,
                # Actor insertion order affects native model/solver ordering. An
                # encoded JSON value survives canonical sorting by owner transport.
                "state_dict_json": json.dumps(self.initial_state),
                "stabilize_steps": self.stabilize_steps,
                "initialization_frames": self.initialization_frames,
                "source_lock_sha256": digest(self.root / "upstreams.lock.json"),
                "facts_sha256": digest(Path(__file__).with_name("g1.json")),
                "asset_pin_sha256": digest(self.root / "src/roboenv/assets/simple-g1.json"),
                "loaded_state_sha256": hashlib.sha256(self.state().tobytes()).hexdigest(),
                "control_protocol": "simple-g1-metric-wbc/4",
            }
        except BaseException:
            self.close()
            raise

    def _check_geometry(self):
        import mujoco

        self.ids = {}
        for name, part in self.model["parts"].items():
            ids = [self.mjmodel.joint(j["name"]).id for j in part["joints"]]
            limits = [[j["limits"]["min"], j["limits"]["max"]] for j in part["joints"]]
            if not np.allclose(self.mjmodel.jnt_range[ids], limits, atol=1e-6, rtol=0):
                raise ValueError(f"SIMPLE joint limits differ: {name}")
            self.ids[name] = ids
        # Validate native camera intrinsics and fixed torso mounts at launch.
        for name, spec in self.model["cameras"].items():
            camera = self.task.layout.cameras[spec["native"]]
            if tuple(camera.resolution) != (spec["width"], spec["height"]):
                raise ValueError(f"SIMPLE camera dimensions differ: {name}")
            if not np.allclose(
                [camera.fx, camera.fy, camera.cx, camera.cy],
                [spec["projection"][key] for key in ("fx", "fy", "cx", "cy")],
            ):
                raise ValueError(f"SIMPLE camera calibration differs: {name}")
            cid = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, spec["native"])
            native = matrix([*self.mjmodel.cam_pos[cid], *self.mjmodel.cam_quat[cid][[1, 2, 3, 0]]])
            native[:3, :3] = native[:3, :3] @ np.diag([1, -1, -1])
            expected = matrix(spec["mount"]["translation_m"] + spec["mount"]["rotation_xyzw"])
            if not np.allclose(native, expected, atol=1e-5, rtol=0):
                raise ValueError(f"SIMPLE camera mount differs: {name}")

    def _pose(self, body):
        index = self.mjmodel.body(body).id
        return [*self.mjdata.xpos[index], *self.mjdata.xquat[index][[1, 2, 3, 0]]]

    def read_device(self, name):
        import mujoco

        state = {"faults": [], "sample_time_s": self.sample_time_s}
        if name == "base":
            pose = self._pose("pelvis")
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.mjmodel,
                self.mjdata,
                mujoco.mjtObj.mjOBJ_XBODY,
                self.mjmodel.body("pelvis").id,
                velocity,
                1,
            )
            return {
                **state,
                "pose": {"translation_m": pose[:3], "rotation_xyzw": pose[3:]},
                "height_m": pose[2],
                "velocities": velocity[[3, 4, 5, 0, 1, 2]].tolist(),
                "frame_poses": {
                    "torso": matrix_pose(
                        np.linalg.inv(matrix(pose)) @ matrix(self._pose("torso_link"))
                    )
                },
                "waist_positions": {
                    joint["name"]: float(self.mjdata.qpos[self.mjmodel.jnt_qposadr[index]])
                    for joint, index in zip(
                        self.model["parts"]["waist"]["joints"], self.ids["waist"], strict=True
                    )
                },
            }
        ids = self.ids[name]
        state.update(
            positions=self.mjdata.qpos[self.mjmodel.jnt_qposadr[ids]].tolist(),
            velocities=self.mjdata.qvel[self.mjmodel.jnt_dofadr[ids]].tolist(),
        )
        if name.endswith("hand"):
            from .hands import frame_poses

            return {**state, "efforts": [], "frame_poses": frame_poses(self, name)}
        part = self.model["parts"][name]
        pose = matrix_pose(
            np.linalg.inv(matrix(self._pose(part["base"]))) @ matrix(self._pose(part["tip"]))
        )
        return {**state, "efforts": [], "frame_poses": {"flange": pose}}

    def read_camera(self, name, directory):
        from PIL import Image

        if name not in self.model["cameras"]:
            raise ValueError("Unknown SIMPLE camera")
        if name not in self.images:
            frames = self.env.render()
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            for camera_name, spec in self.model["cameras"].items():
                data = np.asarray(frames[spec["native"]])
                if data.dtype != np.uint8 or data.shape != (spec["height"], spec["width"], 3):
                    raise ValueError("SIMPLE renderer returned invalid RGB")
                path = directory / f"{uuid.uuid4().hex}.png"
                Image.fromarray(data).save(path)
                self.images[camera_name] = {
                    "file": str(path),
                    "width": spec["width"],
                    "height": spec["height"],
                    "sample_time_s": self.sample_time_s,
                }
        return dict(self.images[name])

    def require_active(self):
        if self.done:
            raise ValueError("SIMPLE episode has terminated")

    def _step(self, targets):
        start = float(self.mjdata.time)
        result = self.env.step(self.wbc.action(targets, self.model["parts"]))
        if not np.isclose(self.mjdata.time - start, 0.02, atol=1e-9, rtol=0):
            raise RuntimeError("SIMPLE physics cadence changed")
        # mj_step integrates qpos after populating body transforms. Refresh only
        # derived kinematics so measured joints, poses and twists share a timestamp;
        # leave dynamics, native reward evaluation, and warm-start state untouched.
        import mujoco

        mujoco.mj_kinematics(self.mjmodel, self.mjdata)
        mujoco.mj_comPos(self.mjmodel, self.mjdata)
        mujoco.mj_comVel(self.mjmodel, self.mjdata)
        return result

    def advance(self, targets):
        self.require_active()
        _, reward, terminated, truncated, _ = self._step(targets)
        self.frames.append(copy.deepcopy(targets))
        self.steps += 1
        self.reward = float(reward)
        self.completed = bool(terminated)
        self.done = bool(
            terminated or truncated or (self.max_steps is not None and self.steps >= self.max_steps)
        )
        self.images = {}

    def state(self):
        import mujoco

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        result = np.empty(mujoco.mj_stateSize(self.mjmodel, spec))
        mujoco.mj_getState(self.mjmodel, self.mjdata, result, spec)
        return result

    def record(self):
        return {
            "schema": "roboenv/simple-record/3",
            "initialization": copy.deepcopy(self.initialization),
            "frames": copy.deepcopy(self.frames),
            "state": self.state().tolist(),
            "commands": copy.deepcopy(self.control.events),
            "implementation": dict(self.implementation),
            "pending_targets": copy.deepcopy(self.control.targets),
            "controller_state": self.wbc.state(),
            "budget": {"used": self.steps, "max": self.max_steps, "unit": "action_frames"},
            "native_success": self.completed,
            "reward": self.reward,
            "done": self.done,
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
