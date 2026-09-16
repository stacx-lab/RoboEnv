"""Explicit, synchronous OpenPI rollouts on the existing native LIBERO world.

The evaluation harness owns inference through the shared VLA runner. These hooks
only adapt observations and native control; they do not attach to device workspaces.
Joint initialization, success latching and the declared episode budget are retained,
so this is a local comparison path, not an official benchmark evaluation protocol.
"""

import numpy as np

from roboenv_runtime.vla.client import quat_xyzw_to_axisangle, resize_with_pad, rotate180


class LiberoPolicyHooks:
    def __init__(self, adapter, spec, instruction=None):
        self.adapter, self.spec, self.native = adapter, spec, adapter._env
        if spec.simulator != "libero" or spec.action_dim != 7:
            raise ValueError("LIBERO policy requires seven OSC pose/gripper actions")
        if spec.action.get("controller") != "OSC_POSE":
            raise ValueError("LIBERO policy controller must be OSC_POSE")
        if self.native.realtime or self.native.get_step_budget()["max"] is None:
            raise ValueError("VLA comparison requires realtime=False and a finite max_steps")
        if not self.native.initialization:
            raise ValueError("Reset the native episode before constructing policy hooks")
        self.instruction = instruction or adapter.goal()
        self._saved = None

    def begin(self):
        from robosuite.controllers import controller_factory, load_controller_config
        from robosuite.utils.buffers import DeltaBuffer

        native, robot = self.native, self.native._robot
        if native.realtime or self._saved is not None or self.steps_left() <= 0:
            raise RuntimeError("Policy rollout is active, realtime, or out of budget")
        native._joint_controller()
        config = {
            **robot.controller_config,
            **load_controller_config(default_controller="OSC_POSE"),
        }
        controller = controller_factory("OSC_POSE", config)
        self._saved = robot.controller, robot.controller_config, robot.recent_actions
        robot.controller, robot.controller_config = controller, config
        native.env._action_dim = robot.action_dim
        robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
        native.control_protocol = "joint_position_with_osc_pose_rollouts"
        native.command_log.append({"frame": native._step_count, "op": "policy_begin"})

    def end(self):
        if self._saved is None:
            return
        native, robot = self.native, self.native._robot
        robot.controller, robot.controller_config, robot.recent_actions = self._saved
        self._saved = None
        native.env._action_dim = robot.action_dim
        native.command_log.append({"frame": native._step_count, "op": "policy_end"})
        # Hold measured state after restoring the original joint controller.
        native.stop_part("arm")
        native.stop_part("gripper")

    def _camera(self, name):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        native, obs = self.native, self.spec.obs
        if not native._render or name not in native._camera_names:
            raise ValueError("Policy cameras must be enabled in the native episode")
        size = int(obs.get("render_size", 256))
        image = native.sim.render(camera_name=name, width=size, height=size)
        image = np.asarray(image)[:: IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]]
        if obs.get("rotate180", True):
            image = rotate180(image)
        size = int(obs.get("resize", 224))
        return np.ascontiguousarray(resize_with_pad(image, size, size))

    def element(self):
        obs = self.native.env._get_observations(force_update=True)
        cameras = self.spec.obs["cameras"]
        return {
            "observation/image": self._camera(cameras["base"]),
            "observation/wrist_image": self._camera(cameras["wrist"]),
            "observation/state": np.concatenate(
                [
                    np.asarray(obs["robot0_eef_pos"]).reshape(3),
                    quat_xyzw_to_axisangle(obs["robot0_eef_quat"]),
                    np.asarray(obs["robot0_gripper_qpos"]).reshape(2),
                ]
            ),
            "prompt": self.instruction,
        }

    def step(self, action):
        if self._saved is None or self.native.realtime:
            raise RuntimeError("Begin an exclusive synchronous policy rollout first")
        action = np.asarray(action)
        if action.shape != (7,) or action.dtype.kind not in "iuf" or not np.isfinite(action).all():
            raise ValueError("Policy action must contain seven finite numbers")
        if self.steps_left() <= 0:
            raise RuntimeError("LIBERO action frame budget exhausted")
        # Preserve robosuite's native OSC scaling/clipping, including the gripper.
        action = action.astype(float)
        self.native.command_log.append(
            {
                "frame": self.native._step_count,
                "op": "policy_step",
                "action": action.tolist(),
            }
        )
        self.native._advance(action)

    def steps_left(self):
        budget = self.native.get_step_budget()
        return budget["max"] - budget["used"]

    def poses(self):
        # Proprioception for the generic progress report; no scene truth goes to the policy.
        robot, sim = self.native._robot, self.native.sim
        return {"gripper": sim.data.site_xpos[robot.eef_site_id].copy()}
