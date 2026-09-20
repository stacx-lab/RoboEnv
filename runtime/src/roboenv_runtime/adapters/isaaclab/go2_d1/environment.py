"""Device-facing Go2 world, retained goals, clock, terminal state and replay journal."""

import copy
import math
import time
from pathlib import Path

import numpy as np

from .gait import DEFAULT, validate_command
from .kinematics import pose
from .tasks import CONTROL, INSTRUCTIONS, TASKS


class Go2Env:
    control_interval = 0.02

    def __init__(
        self,
        asset_manifest=".cache/isaaclab/go2-d1/assets.json",
        realtime=False,
        command_timeout_s=0.5,
        device="cuda:0",
        cameras=True,
        task_name="go2-d1-control",
        max_steps=1500,
    ):
        if task_name not in TASKS:
            raise ValueError("Unknown Go2 task")
        if type(realtime) is not bool or type(cameras) is not bool:
            raise ValueError("realtime and cameras must be booleans")
        if command_timeout_s is not None and (
            type(command_timeout_s) not in (int, float)
            or not math.isfinite(command_timeout_s)
            or command_timeout_s <= 0
        ):
            raise ValueError("Base timeout must be positive and finite or null")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        from roboenv_runtime.paths import repo_root

        from .native import Native

        self.task_name = task_name
        self.realtime = realtime
        self.command_timeout_s = command_timeout_s
        self.max_steps = None if realtime and task_name == CONTROL else max_steps
        self.native = Native(
            repo_root() / asset_manifest, device=device, cameras=cameras, task_name=task_name
        )
        from .profiles import profiles

        self.profiles = profiles(self.native.model, self.native.home, realtime, command_timeout_s)
        self.implementation = self.source_identity()
        self.reset(0)

    @property
    def steps(self):
        return self.native.steps

    @property
    def done(self):
        return self.native.done or (self.max_steps is not None and self.steps >= self.max_steps)

    @property
    def completed(self):
        return self.native.task.completed

    @property
    def instruction(self):
        return INSTRUCTIONS[self.task_name]

    def reset(self, seed):
        self.native.reset(seed)
        self.targets = {
            "base": DEFAULT.copy(),
            "arm": self.native.arm_target.copy(),
            "gripper": self.native.gripper_target.copy(),
        }
        self.deadline = None
        self.events = []
        self.images = {}

    def _validate(self, name, args):
        if name == "base":
            return validate_command(args)
        if name not in ("arm", "gripper") or not isinstance(args, dict) or set(args) != {"target"}:
            raise ValueError("Joint/gripper writes require target only")
        if name == "gripper":
            v = args["target"]
            if type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("Gripper opening target must be in [0,1]")
            return np.array([0.03 * v, -0.03 * v])
        target = args["target"]
        if not isinstance(target, list) or len(target) != 6:
            raise ValueError("Arm requires six absolute joint targets")
        for n, v in zip(self.native.model.arm_names, target, strict=True):
            lo, hi = self.native.model.joints[n]["limits"]
            if type(v) not in (float, int) or not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError("Arm target is nonfinite or outside limits")
        return np.array(target)

    def write(self, name, args):
        target = self._validate(name, args)
        if self.done:
            raise ValueError("Episode terminated; final observations remain readable")
        self.targets[name] = target
        self.events.append(
            {"frame": self.steps, "op": "write", "device": name, "args": copy.deepcopy(args)}
        )
        if name == "base" and self.realtime:
            self.deadline = (
                None
                if self.command_timeout_s is None
                else time.monotonic() + self.command_timeout_s
            )
        if not self.realtime:
            self.advance()
        return {"accepted": True}

    def stop(self, name, reason="requested"):
        if name not in self.targets:
            raise ValueError("Unknown actuator")
        if name == "base":
            self.targets[name][:3] = 0
            self.deadline = None
        else:
            self.targets[name] = np.array(self.read_device(name)["positions"])
        self.events.append({"frame": self.steps, "op": "stop", "device": name, "reason": reason})
        return {
            "stopped": True,
            "behavior": "stand" if name == "base" else "hold",
            "affected_devices": [name],
        }

    def advance(self):
        if self.done:
            return
        self.native.advance(self.targets["base"], self.targets["arm"], self.targets["gripper"])
        self.images = {}

    def tick(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop("base", "timeout")
        self.advance()

    def read_device(self, name):
        state = {
            "faults": [self.native.task.failure] if self.native.task.failure else [],
            "sample_time_s": self.steps * 0.02,
        }
        if name == "base":
            p = self.native.root_pose()
            return {
                **state,
                "pose": {"translation_m": p[:3].tolist(), "rotation_xyzw": p[3:].tolist()},
                "height_m": float(p[2]),
                "frame_poses": {},
                "velocities": [
                    *self.native.array(self.native.robot.data.root_lin_vel_b).tolist(),
                    *self.native.array(self.native.robot.data.root_ang_vel_b).tolist(),
                ],
            }
        q = self.native.joints()
        names = self.native.model.arm_names if name == "arm" else ["d1_Joint7_1", "d1_Joint7_2"]
        velocities = self.native.array(self.native.robot.data.joint_vel)
        result = {
            **state,
            "positions": [q[n] for n in names],
            "velocities": [float(velocities[self.native.ids[n]]) for n in names],
            "efforts": [],
        }
        if name == "arm":
            result["frame_poses"] = {"flange": pose(self.native.model.arm_fk(result["positions"]))}
        elif name == "gripper":
            result["frame_poses"] = {}
        else:
            raise ValueError("Unknown actuator")
        return result

    def read_camera(self, name, directory):
        if name not in self.native.cameras:
            raise ValueError("Camera unavailable")
        if name not in self.images:
            self.images[name] = self.native.capture(name, Path(directory))
        return copy.deepcopy(self.images[name])

    @staticmethod
    def source_identity():
        import hashlib

        from . import gait, kinematics, native, profiles, task_scene, tasks

        sources = {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                gait.__file__,
                native.__file__,
                kinematics.__file__,
                profiles.__file__,
                tasks.__file__,
                task_scene.__file__,
                __file__,
            )
        }
        from roboenv_runtime.paths import repo_root

        for name in ("upstreams.lock.json", "runtime/locks/isaaclab-resolved.txt"):
            sources[name] = hashlib.sha256((repo_root() / name).read_bytes()).hexdigest()
        return sources

    def record(self):
        return {
            "schema": "roboenv/go2-d1-record/1",
            "asset_sha256": self.native.asset_sha,
            "sources": self.implementation,
            "seed": self.native.seed,
            "task_name": self.task_name,
            "task_definition": self.native.task.spec,
            "realtime": self.realtime,
            "max_steps": self.max_steps,
            "events": copy.deepcopy(self.events),
            "targets": {n: q.tolist() for n, q in self.targets.items()},
            "final": self.native.snapshot(),
        }

    def close(self):
        self.native.close()
