"""G1 practice interface over the upstream pick/place task, owned by one worker."""

import json
import os
from pathlib import Path

import numpy as np

from .g1_assets import digest, load_bundle
from .g1_control import Control
from .geometry import relative_pose


class G1Env:
    def __init__(
        self,
        *,
        model_path=None,
        asset_manifest,
        realtime=False,
        device="cuda:0",
        task_name="g1-pick-place",
    ):
        if task_name != "g1-pick-place":
            raise ValueError("G1 supports pick/place only")
        self.realtime = realtime
        from roboenv_runtime.paths import repo_root

        model_path = (
            repo_root() / model_path
            if model_path is not None
            else Path(__file__).with_name("g1.json")
        )
        asset_manifest = repo_root() / asset_manifest
        _, paths = load_bundle(asset_manifest)
        self.model = json.loads(Path(model_path).read_text())
        # The native physics config owns the control period; it is resolved when
        # the episode is built, not declared in the catalog.
        self.control_interval = None
        if self.model["asset_manifest_sha256"] != digest(asset_manifest):
            raise ValueError("G1 catalog was generated from another asset bundle")
        source = repo_root() / "third_party/IsaacLab"
        for name, checksum in self.model["source_files"].items():
            if digest(source / name) != checksum:
                raise ValueError(f"G1 upstream source changed: {name}; regenerate and revalidate")
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        self.app = self.env = None
        self.images = {}
        self.steps = 0
        self.completed = self.done = False
        self.control = Control(
            self, self.model, realtime=realtime
        )
        try:
            from isaaclab.app import AppLauncher
            from .g1_native import create

            self.app = AppLauncher(headless=True, enable_cameras=True, device=device).app
            self.env, self.upper_names, self.control_interval = create(
                paths, self.model, device, realtime=realtime
            )
            self.robot = self.env.scene["robot"]
            self.ids = {
                name: self.robot.find_joints(
                    [j["name"] for j in part["joints"]], preserve_order=True
                )[0]
                for name, part in self.model["parts"].items()
            }
            if len(set(self.upper_names)) != len(self.upper_names):
                raise ValueError("G1 components overlap joints")
            for name, part in self.model["parts"].items():
                actual = self.robot.data.joint_pos_limits.torch[0, self.ids[name]].cpu().numpy()
                expected = [[j["limits"]["min"], j["limits"]["max"]] for j in part["joints"]]
                if not np.allclose(actual, expected, atol=1e-5):
                    raise ValueError(f"G1 {name} limits differ from the generated asset")
        except BaseException:
            import traceback

            traceback.print_exc()
            self.close(exit_code=1)
            raise

    @property
    def sample_time_s(self):
        return self.steps * self.control_interval

    @property
    def max_steps(self):
        return None if self.realtime else self.env.max_episode_length

    def reset(self, seed):
        self.env.allow_reset = True
        self.env.reset(seed=seed)
        self.env.allow_reset = False
        self.images = {}
        self.steps = 0
        self.completed = self.done = False
        self.control = Control(
            self, self.model, realtime=self.realtime
        )
        for name in self.model["parts"]:
            self.control.targets[name] = self.read_device(name)["positions"]

    def _pose(self, name):
        index = self.robot.body_names.index(name)
        pose = self.robot.data.body_link_pose_w.torch[0, index].cpu().numpy().copy()
        # Upstream initial rotations are rounded decimal quaternions. Normalize
        # telemetry at the device boundary before frame composition.
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose.tolist()

    def read_camera(self, name, directory):
        from .observations import camera_state

        if name not in self.model["cameras"]:
            raise ValueError("Unknown G1 camera")
        cached = self.images.get(name)
        if cached is not None and cached["sample_time_s"] == self.sample_time_s:
            return dict(cached)
        self.env.sim.render()
        value = camera_state(self.env.scene[name], directory, self.sample_time_s)
        self.images[name] = value
        return dict(value)

    def read_device(self, name):
        state = {"faults": [], "sample_time_s": self.sample_time_s}
        if name == "base":
            pose = self._pose("pelvis")
            data = self.robot.data
            return {
                **state,
                "pose": {"translation_m": pose[:3], "rotation_xyzw": pose[3:]},
                "height_m": pose[2],
                "frame_poses": {},
                "velocities": [
                    *data.root_lin_vel_b.torch[0].cpu().tolist(),
                    *data.root_ang_vel_b.torch[0].cpu().tolist(),
                ],
            }
        part = self.model["parts"][name]
        ids = self.ids[name]
        state.update(
            positions=self.robot.data.joint_pos.torch[0, ids].cpu().tolist(),
            velocities=self.robot.data.joint_vel.torch[0, ids].cpu().tolist(),
        )
        return {
            **state,
            "efforts": [],
            "frame_poses": {}
            if name.endswith("hand")
            else {"flange": relative_pose(self._pose(part["base"]), self._pose(part["tip"]))},
        }

    def require_active(self):
        if self.done or (self.max_steps is not None and self.steps >= self.max_steps):
            raise ValueError("G1 episode has terminated")

    def advance(self, targets):
        self.require_active()
        import torch

        by_joint = {
            joint["name"]: q
            for name, part in self.model["parts"].items()
            for joint, q in zip(part["joints"], targets[name], strict=True)
        }
        actions = [by_joint[name] for name in self.upper_names] + targets["base"]
        with torch.inference_mode():
            _, _, terminated, truncated, _ = self.env.step(
                torch.tensor([actions], device=self.env.device)
            )
        self.steps += 1
        self.completed = bool(self.env.termination_manager.get_term("success")[0])
        self.done = bool(terminated[0] or truncated[0])

    def close(self, exit_code=0):
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.app is not None:
            self.app.close(exit_code=exit_code)
            self.app = None
