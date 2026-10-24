"""One T2 world, advanced exclusively by RoboEnv's shared owner scheduler."""

import copy
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

from roboenv.devices.model import DeviceError
from roboenv.provisioning.booster_t2_assets import asset_root, verify_assets
from roboenv_runtime.core.types import Budget, ScoreReport
from roboenv_runtime.paths import repo_root

from .interface import Pose
from .task import SceneTask


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def profiles():
    root = repo_root() / "catalog/robots/isaaclab/isaaclab-booster-t2/devices"
    return {p.parent.name: {k: v for k, v in yaml.safe_load(p.read_text()).items()
                           if k in ("type", "intrinsics", "extrinsics")}
            for p in sorted(root.glob("*/profile.yaml"))}


class T2Env:
    control_interval = 0.02

    def __init__(self, *, task_name, task_spec, asset_directory=None,
                 max_steps=None, execution_mode="stepped", stop_on_success=True,
                 cameras=True, device_profiles=None):
        from ...registry import get_task
        from .controller.policy import PolicyConfig, OnnxPolicy
        from .controller.robot import RobotConfig
        from .controller.controller import LocoManipController
        from .controller.kinematics import RobotKinematics

        if type(cameras) is not bool or type(stop_on_success) is not bool:
            raise ValueError("Camera and stop-on-success flags must be boolean")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be positive or null")
        task = get_task(task_name)
        if task.embodiment != "booster_t2":
            raise ValueError("T2 requires a T2 task")
        self.definition = task.definition(task_spec)
        self.task_name = task_name
        self.asset_directory = Path(asset_directory or asset_root(repo_root())).resolve()
        authored = self.definition["version"] in (3, 4, 5)
        self.asset_identity = verify_assets(self.asset_directory,
            scenes=bool(self.definition["scene"]) and self.definition["version"] != 3,
            articulated=bool(self.definition["articulated"]) and not authored)
        self.policy_config = PolicyConfig(asset_directory=self.asset_directory)
        self.robot = RobotConfig(asset_directory=self.asset_directory)
        self.policy = LocoManipController(self.policy_config, self.robot,
                                         OnnxPolicy(self.policy_config.onnx))
        if self.policy.policy_dt != self.control_interval or self.robot.timestep != 0.005:
            raise ValueError("T2 requires the pinned 50 Hz controller and 200 Hz physics")
        self.arm_kinematics = {
            side: RobotKinematics(self.robot, self.robot.joint_groups[side + "_arm"],
                                  [side + "_palm"]) for side in ("left", "right")}
        self.profiles = copy.deepcopy(profiles() if device_profiles is None else device_profiles)
        expected = profiles()
        if self.profiles != expected:
            raise ValueError("T2 device facts differ from the selected catalog")
        self._stepped_max_steps = max_steps
        self.stop_on_success = stop_on_success
        self.configure_execution_mode(execution_mode)
        from ...common.recording import source_identity

        self.sources = source_identity(task_name, Path(__file__).parent)
        for path in sorted(Path(__file__).parent.rglob("*.yaml")):
            self.sources[str(path.relative_to(Path(__file__).parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.app = self.native = None
        try:
            from roboenv_runtime.adapters.gpu import launch_isaaclab

            self.app = launch_isaaclab("isaaclab", headless=True, enable_cameras=cameras)
            from .world import IsaacWorld

            self.task = SceneTask(task_name, self.definition, self.asset_directory)
            self.native = IsaacWorld(self.robot, self.policy.joint_names, self.task,
                                      camera=cameras, device="cpu")
            self._item_start = copy.deepcopy(self.native._item_start)
            self.reset(0)
        except BaseException:
            self.close()
            raise

    def configure_execution_mode(self, mode):
        if mode not in ("realtime", "stepped"):
            raise ValueError("Unknown T2 execution mode")
        self.execution_mode = mode
        self.max_steps = None if mode == "realtime" else self._stepped_max_steps

    @property
    def completed(self):
        return self.task.state.completed

    @property
    def done(self):
        return (self.fallen or self.stop_on_success and self.completed
                or self.max_steps is not None and self.steps >= self.max_steps)

    def reset(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Seed zero retains the source layout. Other seeds make small physical variations.
        start = np.asarray(self.definition["start"], dtype=float)
        if seed:
            start += rng.uniform([-0.015, -0.015, -0.02], [0.015, 0.015, 0.02])
        self.task.start = tuple(start)
        self.native._item_start = copy.deepcopy(self._item_start)
        if seed and self.definition["version"] not in (3, 4, 5):
            for pose in self.native._item_start.values():
                pose[:2] = (np.asarray(pose[:2]) + rng.uniform(-0.015, 0.015, 2)).tolist()
        self.native.reset()
        self.policy.reset()
        self.steps, self.fallen, self.events = 0, False, []
        self.velocity = np.zeros(3)
        self.joint_targets = {name: np.array([self.robot.keyframe_joint_pos(joints)[i]
                                             if name == "head" else 0 for i in range(len(joints))])
                              for name, joints in self.robot.joint_groups.items()
                              if not name.endswith("_arm")}
        self.palm_targets = dict(self.policy_config.rest_palm_targets)
        self.palm_offset = {s: np.zeros(3) for s in ("left", "right")}
        self.last_palm_target = {}
        self.images = {}
        # Settling is owner initialization, excluded from the episode's 50 Hz budget.
        for _ in range(100):
            self._advance()
            if self._fallen():
                raise ValueError("T2 fell during initialization; do not substitute the seed")
        self.native._time = 0.0
        self.task.reset()
        self.task.state.initialize(self.native)
        self.initial_state = self.snapshot()
        self.initialization = {
            "requested_seed": seed, "effective_seed": seed,
            "task_name": self.task_name, "control_protocol": "booster-t2-palm-wbc/1",
            "control_interval_s": self.control_interval, "physics_interval_s": 0.005,
            "initialization_frames": 100, "loaded_state_sha256": checksum(self.initial_state),
            "asset_sha256": self.asset_identity["asset_sha256"],
            "sources_sha256": checksum(self.sources), "task_sha256": checksum(self.definition),
            "policy_sha256": hashlib.sha256(self.policy_config.onnx.read_bytes()).hexdigest(),
        }

    def _fallen(self):
        pos, quat = self.native.root_pose()
        q = self.native.read().joint_pos
        if not np.isfinite(np.r_[pos, quat, q]).all():
            return True
        return bool(pos[2] < 0.5 or 1 - 2 * (quat[1] ** 2 + quat[2] ** 2) < math.cos(1.0))

    def _validate(self, name, args):
        if name == "base":
            if (set(args) != {"mode", "velocity"} or args["mode"] != "locomotion"
                    or not isinstance(args["velocity"], dict)
                    or set(args["velocity"]) != {"vx", "vy", "wz"}):
                raise ValueError("T2 base requires locomotion mode and vx, vy, wz")
            value = [args["velocity"][k] for k in ("vx", "vy", "wz")]
            limits = self.policy_config.command_range
        else:
            if name not in self.robot.joint_groups or set(args) != {"target"}:
                raise ValueError("Joint writes require target only")
            value = args["target"]
            limits = self.robot.limits(self.robot.joint_groups[name])
        if (not isinstance(value, list) or len(value) != len(limits)
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in value)):
            raise ValueError("Command has wrong dimensions or nonfinite values")
        q = np.asarray(value, dtype=float)
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("Command exceeds trained or joint limits")
        if name.endswith("_arm"):
            side = name.split("_", 1)[0]
            p, r = self.arm_kinematics[side].frame_poses(q, np.array([1., 0, 0, 0]))
            box = self.policy_config.palm_workspace[side]
            if np.any(p[0] < box[0] - 0.05) or np.any(p[0] > box[1] + 0.05):
                raise ValueError("Palm target exceeds trained workspace")
            return Pose.of(p[0], r[0])
        return q

    def set_target(self, name, args):
        value = self._validate(name, args)
        if self.done:
            raise DeviceError("episode_finished", "T2 episode terminated", details={"effects": "none"})
        if name == "base":
            self.velocity = value
        elif name.endswith("_arm"):
            self.palm_targets[name.split("_", 1)[0]] = value
        else:
            self.joint_targets[name] = value
        self.events.append({"frame": self.steps, "op": "write", "device": name,
                            "args": copy.deepcopy(args)})
        return {"accepted": True}

    def stop(self, name, reason="requested"):
        if name == "base":
            self.velocity[:] = 0
        elif name.endswith("_arm") and name in self.robot.joint_groups:
            side = name.split("_", 1)[0]
            self.palm_targets[side] = self.policy.palm_poses(self.native.read())[side]
            self.palm_offset[side][:] = 0
        elif name == "head":
            self.joint_targets[name] = np.asarray(self.read_device(name)["positions"])
        elif name not in ("left_hand", "right_hand"):
            raise ValueError("Unknown actuator")
        # Finger stop retains applied targets and grip preload.
        self.events.append({"frame": self.steps, "op": "stop", "device": name, "reason": reason})
        return {"stopped": True, "behavior": "stand" if name == "base" else "hold",
                "affected_devices": [name]}

    def _advance(self):
        from .controller.controller import PolicyCommand

        state = self.native.read()
        measured = self.policy.palm_poses(state)
        targets = {}
        for side, target in self.palm_targets.items():
            last = self.last_palm_target.get(side)
            if last is not None and np.linalg.norm(target.pos - last) < 0.002:
                self.palm_offset[side] = np.clip(self.palm_offset[side]
                    + self.control_interval / 0.5 * (target.pos - measured[side].pos), -0.05, 0.05)
            self.last_palm_target[side] = target.pos.copy()
            box = self.policy_config.palm_workspace[side]
            targets[side] = Pose.of(np.clip(target.pos + self.palm_offset[side], box[0], box[1]),
                                    target.quat)
        target = self.policy.act(state, PolicyCommand(self.velocity.copy(), targets))
        held = {j: v for name, q in self.joint_targets.items()
                for j, v in zip(self.robot.joint_groups[name], q, strict=True)}
        for i, name in enumerate(self.policy.joint_names):
            if name in held:
                target[i] = held[name]
        self.native.set_finger_targets(self.robot.fingers.positions(held))
        for _ in range(self.policy_config.decimation):
            self.native.step(target)

    def tick(self):
        if self.done:
            raise DeviceError("episode_finished", "T2 episode terminated", details={"effects": "none"})
        self._advance()
        self.steps += 1
        self.images = {}
        self.fallen = self._fallen()
        if not self.fallen and self.definition["version"] not in (3, 4, 5):
            self.task.evaluate(self.native)

    def read_device(self, name):
        from .controller.rotations import quat_to_matrix

        state = {"timestamp": self.steps * self.control_interval,
                 "faults": [{"code": "fallen"}] if self.fallen else []}
        if name == "base":
            pos, quat = self.native.root_pose()
            twist = self.native.root_velocity()
            return {**state, "height_m": float(pos[2]),
                    "velocities": np.r_[quat_to_matrix(quat).T @ twist[:3], twist[3:]].tolist()}
        if name not in self.robot.joint_groups:
            raise ValueError("Unknown joint device")
        joints = self.robot.joint_groups[name]
        if name.endswith("_hand"):
            pos, vel = self.native.read_fingers()
            order = self.robot.fingers.joints
        else:
            proprio = self.native.read()
            pos, vel, order = proprio.joint_pos, proprio.joint_vel, self.policy.joint_names
        idx = [order.index(j) for j in joints]
        return {**state, "positions": pos[idx].tolist(), "velocities": vel[idx].tolist(), "efforts": []}

    def read_camera(self, name, directory):
        if name != "head_camera" or self.native.camera is None:
            raise ValueError("Camera unavailable")
        if name not in self.images:
            from ...common.observations import camera_state

            self.native.capture_camera()
            self.images[name] = camera_state(self.native.camera._camera, directory,
                                             self.steps * self.control_interval)
        return copy.deepcopy(self.images[name])

    def snapshot(self):
        state = self.native.read()
        pos, quat = self.native.root_pose()
        fingers = self.native.read_fingers()
        return {"root": np.r_[pos, quat].tolist(), "root_velocity": self.native.root_velocity().tolist(),
                "references": {"velocity": self.velocity.tolist(),
                    "joints": {n: q.tolist() for n, q in self.joint_targets.items()},
                    "palms": {n: np.r_[p.pos, p.quat].tolist() for n, p in self.palm_targets.items()},
                    "last_palm_target": {n: p.tolist() for n, p in self.last_palm_target.items()}},
                "joints": state.joint_pos.tolist(), "joint_velocities": state.joint_vel.tolist(),
                "fingers": fingers[0].tolist(), "finger_velocities": fingers[1].tolist(),
                "objects": {n: {"pose": np.concatenate(self.native.item_pose(n)).tolist(),
                                  "velocity": self.native.item_velocity(n).tolist()}
                            for n in self.definition["items"]},
                "articulated": {n: self.native.articulated_state(n)
                                for n in self.definition["articulated"]},
                "controller": {"last_action": self.policy._last_action.tolist(),
                    "steps": self.policy._steps,
                    "history": {n: [v.tolist() for v in h] for n, h in self.policy._history.items()},
                    "palm_offset": {n: v.tolist() for n, v in self.palm_offset.items()}},
                "task": self.task.state.record(), "steps": self.steps, "fallen": self.fallen}

    def score(self):
        return ScoreReport(reward=float(self.completed), raw_score=float(self.completed),
                           completed=self.completed)

    def budget(self):
        return Budget(used=self.steps, max=self.max_steps, unit="action_frames")

    def record(self):
        return {"schema": "roboenv/booster-t2-record/1", "backend": "isaaclab",
                "task_name": self.task_name, "task_definition": copy.deepcopy(self.definition),
                "initialization": copy.deepcopy(self.initialization),
                "initial_state": copy.deepcopy(self.initial_state), "sources": dict(self.sources),
                "device_profiles": copy.deepcopy(self.profiles), "seed": self.seed,
                "execution_mode": self.execution_mode, "max_steps": self.max_steps,
                "stop_on_success": self.stop_on_success, "native_success": self.completed,
                "budget": self.budget().as_dict(), "events": copy.deepcopy(self.events),
                "final": self.snapshot()}

    def spectator_sources(self):
        from .spectator import sources

        return sources(self)

    def close(self):
        error = sys.exc_info()[1]
        if error is not None:
            traceback.print_exception(error)
        try:
            if self.native is not None:
                native, self.native = self.native, None
                native.sim.stop()
                native.sim.clear_instance()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if self.app is not None:
                app, self.app = self.app, None
                app.close(exit_code=1 if error is not None or sys.exc_info()[1] else 0)
