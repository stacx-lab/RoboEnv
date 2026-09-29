"""Device-facing Go2 world, retained goals, clock, terminal state and replay journal."""

import copy
import math
from pathlib import Path

import numpy as np
from roboenv.devices.model import DeviceError

from .adapter import DEFAULT, pose, validate_command
from .tasks import PICK_PLACE, TASKS, catalog_definition, definition


def load_profiles(robot=None):
    """Load device facts from the catalog, without constructing a second definition."""
    import yaml
    from roboenv_runtime.paths import repo_root

    robot = (
        Path(robot)
        if robot is not None
        else (repo_root() / "catalog/robots/isaaclab/isaaclab-go2-d1")
    )
    return {
        path.parent.name: {
            key: value
            for key, value in yaml.safe_load(path.read_text()).items()
            if key in {"type", "intrinsics", "extrinsics"}
        }
        for path in sorted((robot / "devices").glob("*/profile.yaml"))
    }


class Go2Env:
    control_interval = 0.02

    def __init__(
        self,
        asset_manifest=".cache/isaaclab/go2-d1/assets.json",
        device="cuda:0",
        cameras=True,
        task_name="go2-d1-control",
        max_steps=None,
        payload_mass_kg=None,
        task_spec=None,
        tcp_in_flange_m=None,
        device_profiles=None,
        execution_mode="stepped",
    ):
        if task_name not in TASKS:
            raise ValueError("Unknown Go2 task")
        if payload_mass_kg is not None:
            if task_name != PICK_PLACE:
                raise ValueError("Payload mass applies only to pick-and-place")
            task_spec = copy.deepcopy(
                catalog_definition(task_name) if task_spec is None else task_spec
            )
            if not isinstance(task_spec, dict):
                raise ValueError("task_spec must be a mapping")
            task_spec["object_mass_kg"] = payload_mass_kg
        self.profiles = copy.deepcopy(
            load_profiles() if device_profiles is None else device_profiles
        )
        if tcp_in_flange_m is None:
            tcp_in_flange_m = self.profiles["gripper"]["extrinsics"]["mount"]["translation_m"]
        spec = definition(task_name, task_spec, tcp_in_flange_m)
        if type(cameras) is not bool:
            raise ValueError("cameras must be a boolean")
        max_steps = 6000 if max_steps is None else max_steps
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._stepped_max_steps = max_steps
        self.configure_execution_mode(execution_mode)
        from roboenv_runtime.paths import repo_root

        from .native import Native

        self.profiles["gripper"]["extrinsics"]["mount"]["translation_m"] = spec["tcp_in_flange_m"]
        self.task_name = task_name
        self.native = Native(
            repo_root() / asset_manifest,
            device=device,
            cameras=cameras,
            task_name=task_name,
            task_spec=spec,
            device_profiles=self.profiles,
        )
        self.implementation = self.source_identity()
        self.reset(0)

    def configure_execution_mode(self, mode):
        """Select the budget only; WorldSession remains the clock authority."""
        if mode not in {"stepped", "realtime"}:
            raise ValueError("execution mode must be stepped or realtime")
        self.execution_mode = mode
        self.max_steps = None if mode == "realtime" else self._stepped_max_steps

    @property
    def steps(self):
        return self.native.steps

    @property
    def done(self):
        return self.native.done or (self.max_steps is not None and self.steps >= self.max_steps)

    @property
    def completed(self):
        return self.native.task.completed

    def reset(self, seed):
        self.native.reset(seed)
        self.targets = {
            "base": DEFAULT.copy(),
            "arm": self.native.arm_target.copy(),
            "gripper": self.native.gripper_target.copy(),
        }
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

    def set_target(self, name, args):
        """Set a reference without stepping; the shared scheduler owns its duration."""
        target = self._validate(name, args)
        if self.done:
            raise DeviceError(
                "episode_finished", "Go2 episode terminated", details={"effects": "none"}
            )
        self.targets[name] = target
        self.events.append(
            {"frame": self.steps, "op": "write", "device": name, "args": copy.deepcopy(args)}
        )
        return {"accepted": True}

    def set_posture_feedback(self, enabled):
        """Owner-only ablation switch, journaled for exact command replay."""
        if type(enabled) is not bool:
            raise ValueError("Posture feedback switch must be boolean")
        if self.done:
            raise ValueError("Episode terminated")
        self.native.posture_feedback = enabled
        self.events.append({"frame": self.steps, "op": "posture_feedback", "enabled": enabled})

    def stop(self, name, reason="requested"):
        if name not in self.targets:
            raise ValueError("Unknown actuator")
        if name == "base":
            self.targets[name][:3] = 0
        else:
            # Measured positions may overshoot a limit slightly; hold inside it.
            names = self.native.model.arm_names if name == "arm" else ["d1_Joint7_1", "d1_Joint7_2"]
            lo, hi = np.array([self.native.model.joints[n]["limits"] for n in names]).T
            self.targets[name] = np.clip(self.read_device(name)["positions"], lo, hi)
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
        # The shared clock counts every successful tick as a physical interval.
        # Report a terminal no-op explicitly so it cannot advance world time.
        if self.done:
            raise DeviceError(
                "episode_finished", "Go2 episode terminated", details={"effects": "none"}
            )
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

        from . import adapter, limits, native, tasks

        sources = {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                adapter.__file__,
                native.__file__,
                limits.__file__,
                tasks.__file__,
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
            "payload_mass_kg": self.native.payload_mass_kg,
            "task_definition": self.native.task.spec,
            "device_profiles": copy.deepcopy(self.profiles),
            "max_steps": self.max_steps,
            "execution_mode": self.execution_mode,
            "events": copy.deepcopy(self.events),
            "targets": {n: q.tolist() for n, q in self.targets.items()},
            "final": self.native.snapshot(),
        }

    def close(self):
        self.native.close()
