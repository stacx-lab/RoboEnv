"""Retained YAM device commands on the one shared world clock."""

import copy
import math
from pathlib import Path

from roboenv_runtime.core.types import Budget, ScoreReport


class YamEnv:
    control_interval = 0.034

    def __init__(
        self,
        *,
        task,
        asset_manifest=".cache/isaaclab/yam/assets.json",
        device_profiles=None,
        execution_mode="realtime",
    ):
        from roboenv_runtime.paths import repo_root

        from .native import Native

        self.task = task
        self.task_name = task.name
        self.profiles = copy.deepcopy(
            task.profiles() if device_profiles is None else device_profiles
        )
        self.configure_execution_mode(execution_mode)
        self.native = Native(repo_root() / asset_manifest, task, self.profiles)
        try:
            self.reset(0)
        except BaseException:
            self.native.close()
            raise

    def configure_execution_mode(self, mode):
        if mode not in {"stepped", "realtime"}:
            raise ValueError("execution mode must be stepped or realtime")
        self.execution_mode = mode

    def reset(self, seed):
        self.native.reset(seed)
        self.targets = {
            **{side + "_arm": list(self.native.home[side]) for side in ("left", "right")},
            **{side + "_gripper": [1.0] for side in ("left", "right")},
        }
        self.events, self.images = [], {}

    def validate_target(self, name, target):
        if name not in self.targets:
            raise ValueError("Unknown YAM actuator")
        length = 6 if name.endswith("_arm") else 1
        if (
            not isinstance(target, list)
            or len(target) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in target)
        ):
            raise ValueError(f"{name} requires {length} finite absolute targets")
        if length == 1 and not 0 <= target[0] <= 1:
            raise ValueError("Gripper opening must be in [0, 1]")
        # Native joint drives enforce the source joint ranges, as declared by the profile.

    def set_target(self, name, target):
        self.validate_target(name, target)
        self.targets[name] = target.copy()
        self.events.append(
            {"frame": self.native.steps, "op": "write", "device": name, "target": target.copy()}
        )
        return {"accepted": True}

    def stop(self, name):
        if name not in self.targets:
            raise ValueError("Unknown YAM actuator")
        if name.endswith("_arm"):
            self.targets[name] = self.read_device(name)["positions"]
        # A stopped gripper retains target/preload; only explicit open releases it.
        self.events.append({"frame": self.native.steps, "op": "stop", "device": name})
        return {"stopped": True, "behavior": "hold", "affected_devices": [name]}

    def tick(self):
        self.native.advance(self.targets)
        self.images.clear()

    def read_device(self, name):
        return self.native.read_device(name)

    def read_camera(self, name, directory):
        if name not in self.images:
            self.images[name] = self.native.capture(name, Path(directory))
        return copy.deepcopy(self.images[name])

    def score(self):
        return ScoreReport()

    def budget(self):
        return Budget(used=self.native.steps, max=None, unit="action_frames")

    def record(self):
        from ...common.recording import source_identity

        return {
            "schema": "roboenv/yam-practice-record/1",
            "task_name": self.task_name,
            "asset_sha256": self.native.asset_sha,
            "seed": self.native.seed,
            "sources": source_identity(
                self.task_name,
                Path(__file__).parent,
                extra_directories=self.task.source_directories,
                backends=["isaaclab", "abc"],
            ),
            "execution_mode": self.execution_mode,
            "events": copy.deepcopy(self.events),
            "targets": copy.deepcopy(self.targets),
            "steps": self.native.steps,
        }

    def spectator_sources(self):
        from .spectator import sources

        return sources(self)

    def close(self):
        self.native.close()
