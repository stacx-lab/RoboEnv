"""Eight retained joint targets on RoboEnv's shared control clock."""

import copy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from roboenv.devices.model import DeviceError
from roboenv.provisioning.roboballet_assets import SHA256, asset_root, verify_assets
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.core.types import Budget, ScoreReport
from roboenv_runtime.paths import repo_root

from ...common.recording import source_identity
from ...common.spectator import SpectatorSource
from ...tasks.roboballet_eight_arm.reaching import DT, HOME_TOLERANCE, MAX_STEPS, ReachingTask


class RoboBalletEnv:
    control_interval = DT
    task_name = "roboballet-eight-arm"

    def __init__(self, *, max_steps=MAX_STEPS, target_set_index=1, asset_directory=None,
                 device_profiles=None, execution_mode="realtime"):
        if type(max_steps) is not int or max_steps != MAX_STEPS:
            raise ValueError("RoboBallet reaching requires the 420-frame stepped horizon")
        if type(target_set_index) is not int or not 0 <= target_set_index < 10:
            raise ValueError("target_set_index must select one of the ten original sets")
        root = repo_root()
        self.depot = (Path(asset_directory).expanduser() if asset_directory is not None
                      else asset_root(root) / "selected")
        if not self.depot.is_absolute():
            self.depot = root / self.depot
        self.assets = verify_assets(self.depot)
        self.scene_path = root / "catalog/tasks/isaaclab/roboballet-eight-arm/scene.json"
        profiles = root / "catalog/robots/isaaclab/isaaclab-roboballet-eight-panda/devices"
        self.profiles = {p.parent.name: yaml.safe_load(p.read_text())
                         for p in profiles.glob("*/profile.yaml")}
        if device_profiles is not None:
            from roboenv_runtime.bindings.backends.shared.profile import validate

            validate(device_profiles, {
                name: {key: value[key] for key in ("type", "intrinsics", "extrinsics")
                       if key in value}
                for name, value in self.profiles.items()
            }, "RoboBallet profiles")
        self.target_set_index = target_set_index
        self.native = None
        self.implementation = source_identity(self.task_name, Path(__file__).parent)
        self.configure_execution_mode(execution_mode)

    def configure_execution_mode(self, mode):
        if mode not in {"stepped", "realtime"}:
            raise ValueError("execution mode must be stepped or realtime")
        self.execution_mode = mode
        self.max_steps = MAX_STEPS if mode == "stepped" else None

    def reset(self, seed):
        if self.native is not None:
            raise ValueError("Start a fresh owner process for another RoboBallet episode")
        self.seed = seed
        self.task = ReachingTask(self.depot, self.scene_path,
                                target_set_index=self.target_set_index, seed=seed)
        self.task.validate_reachability()
        self.targets = {f"arm{i + 1}": q.tolist() for i, q in enumerate(self.task.home)}
        self.commands, self.images = [], {}
        self.initial_state = self.owner_state()
        from .native import ReachingScene

        self.native = ReachingScene(self.task, self.profiles)

    def tick(self):
        if self.budget().exhausted:
            raise DeviceError("episode_finished", "RoboBallet frame limit reached",
                              details={"effects": "none"})
        self.task.step([self.targets[f"arm{i + 1}"] for i in range(8)])
        self.native.sync()
        self.images.clear()

    def validate_target(self, name, target):
        if (name not in self.targets or not isinstance(target, list) or len(target) != 7
                or any(type(v) not in (int, float) for v in target)
                or not np.isfinite(target).all()):
            raise ValueError("RoboBallet arms require seven finite joint targets")

    def set_target(self, name, target):
        self.validate_target(name, target)
        self.targets[name] = target.copy()
        self.commands.append({"frame": self.task.frame, "op": "write",
                              "device": name, "target": target.copy()})
        return {"accepted": True}

    def stop(self, name):
        if name not in self.targets:
            raise ValueError("Unknown RoboBallet actuator")
        self.targets[name] = self.read_device(name)["positions"]
        self.commands.append({"frame": self.task.frame, "op": "stop", "device": name})
        return {"stopped": True, "behavior": "hold", "affected_devices": [name]}

    def read_device(self, name):
        if name == "workcell":
            return self.task.public_state()
        if name not in self.targets:
            raise ValueError("Unknown RoboBallet actuator")
        i = int(name.removeprefix("arm")) - 1
        base = self.task.world.GetRobotBaseTransform(self.task.robots[i])
        local = np.linalg.inv(base) @ self.task.poses()[i]
        return {"positions": self.task.positions()[i].tolist(),
                "velocities": self.task.velocity[i].tolist(), "efforts": [], "faults": [],
                "timestamp": self.task.frame * DT,
                "frame_poses": {"tool_tip": [*local[:3, 3].tolist(),
                    *Rotation.from_matrix(local[:3, :3]).as_quat().tolist()]}}

    def read_camera(self, name, directory):
        if name not in self.images:
            self.images[name] = self.native.capture(name, Path(directory))
        return copy.deepcopy(self.images[name])

    def check(self, configurations):
        return self.task.check(configurations)

    def score(self):
        task = self.task
        count = int((task.solved_by >= 0).sum())
        return ScoreReport(reward=1. if task.completed else count / 41., raw_score=count,
                           completed=task.completed,
                           detail={"targets_completed": count, "targets_total": 40,
                                   "returned_home": bool((np.abs(task.positions() - task.home)
                                                          < HOME_TOLERANCE).all()),
                                   "dwell_remaining_steps": task.dwell.tolist(),
                                   "collision_robot_steps": task.collision_robot_steps,
                                   "unsafe_proposal_steps": task.unsafe_proposal_steps,
                                   "unsafe_proposal_robot_steps": task.unsafe_proposal_robot_steps,
                                   "makespan_s": task.frame * DT,
                                   "motion_frames_by_active_arms": task.motion_histogram,
                                   "work_frames_by_active_arms": task.work_histogram})

    def budget(self):
        return Budget(used=self.task.frame, max=self.max_steps, unit="action_frames")

    def owner_state(self):
        task = self.task
        state = {"frame": task.frame, "positions": task.positions().tolist(),
                 "velocities": task.velocity.tolist(), "solved_by": task.solved_by.tolist(),
                 "dwell_steps": task.dwell.tolist(), "completed": task.completed,
                 "collision_robot_steps": task.collision_robot_steps,
                 "unsafe_proposal_steps": task.unsafe_proposal_steps,
                 "unsafe_proposal_robot_steps": task.unsafe_proposal_robot_steps,
                 "trajectory_telemetry": copy.deepcopy(task.telemetry),
                 "motion_frames_by_active_arms": task.motion_histogram.copy(),
                 "work_frames_by_active_arms": task.work_histogram.copy(),
                 "events": copy.deepcopy(task.events)}
        return {"state": state, "sha256": digest(state)}

    def record(self):
        return {
            "schema": "roboenv/roboballet-record/1", "task_name": self.task_name,
            "seed": self.seed, "target_set_index": self.target_set_index,
            "sources": self.implementation, "assets": self.assets,
            "source_archive_sha256": SHA256, "scene_sha256": self.task.scene_hash,
            "dependencies": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
            "execution_mode": self.execution_mode, "max_steps": self.max_steps,
            "control_interval_s": DT, "collision_backend": "original_mujoco",
            "collision_margin_m": 0., "collision_tie_break": "seeded_sorted_candidates",
            "workpiece_translation_m": [0, 0, .6],
            "commands": copy.deepcopy(self.commands), "targets": copy.deepcopy(self.targets),
            "initial_state": copy.deepcopy(self.initial_state), "state": self.owner_state(),
        }

    def spectator_sources(self):
        mount = self.profiles["camera_side"]["extrinsics"]["mount"]
        pose = [*mount["translation_m"], *mount["rotation_xyzw"]]
        return {"third_person": SpectatorSource(lambda: pose.copy())}

    def close(self):
        if self.native is not None:
            self.native.close()
