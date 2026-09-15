"""Metric commands and joint references over SIMPLE's in-process GR00T WBC."""

import numpy as np


class WholeBodyControl:
    def __init__(self, robot):
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

        self.robot = robot
        self.time = float(robot.mjData.time)
        self.model = instantiate_g1_robot_model(waist_location="lower_body")
        config = ControlLoopConfig(enable_waist=True).load_wbc_yaml()
        self.policy = get_wbc_policy("g1", self.model, config, init_time=self.time)
        self.policy.clock = lambda: self.time
        self.policy.reset(init_time=self.time)
        self.policy.lower_body_policy.use_policy_action = True
        indices = set(self.model.get_joint_group_indices("upper_body"))
        self.upper_names = [
            name
            for name, index in sorted(
                self.model.joint_to_dof_index.items(), key=lambda item: item[1]
            )
            if index in indices
        ]
        if len(self.upper_names) != 28:
            raise ValueError("SIMPLE upper controller must expose only 14 arm and 14 hand references")

    def initial_references(self):
        return dict(zip(self.upper_names, self.model.get_initial_upper_body_pose(), strict=True))

    def observation(self):
        # Reuse the upstream conversion without its agent queue or wall-clock setup.
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

        class Converter:
            _dwbc_robot_model = self.model

        return SonicDecoupledWbcAgent._build_wbc_observation(Converter(), self.robot.prepare_obs())

    def action(self, targets, parts):
        from simple.core.action import ActionCmd

        self.time = float(self.robot.mjData.time)
        positions = {
            joint["name"]: value
            for name, part in parts.items() if name != "waist"
            for joint, value in zip(part["joints"], targets[name], strict=True)
        }
        torso_rpy = np.asarray(targets["base"][4:7])
        velocity, height = targets["base"][:3], targets["base"][3]
        goal = {
            "target_upper_body_pose": np.asarray([positions[n] for n in self.upper_names]),
            "navigate_cmd": np.asarray([*velocity, 0.0]),
            "base_height_command": np.array([height]),
            "target_time": self.time + 0.02,
            "interpolation_garbage_collection_time": self.time - 0.04,
        }
        self.policy.set_goal(goal)
        upper = self.policy.upper_body_policy.get_action(self.time + 0.02)
        q = np.zeros(self.model.num_dofs)
        q[self.model.get_joint_group_indices("upper_body")] = upper["target_upper_body_pose"]
        lower = self.policy.lower_body_policy
        lower.set_metric_command(velocity, height, torso_rpy)
        self.policy.set_observation(self.observation())
        # One target vector: upper interpolation owns arms/hands, native WBC owns
        # legs/waist. Torso orientation now comes directly from the base command.
        lower_action = lower.get_action(
            time=self.time + 0.02,
            arms_target_pose=q[self.model.get_joint_group_indices("arms")],
            base_height_command=np.array([height]),
            torso_orientation_rpy=torso_rpy,
            interpolated_navigate_cmd=goal["navigate_cmd"],
        )
        indices = self.model.get_joint_group_indices("lower_body")
        q[indices] = lower_action["body_action"][0][:len(indices)]
        self.policy.last_action = {"q": q}
        action = q
        return ActionCmd(
            "decoupled_wbc",
            target_q=self.model.get_body_actuated_joints(action),
            left_hand_q=self.model.get_hand_actuated_joints(action, side="left"),
            right_hand_q=self.model.get_hand_actuated_joints(action, side="right"),
        )

    def state(self):
        lower = self.policy.lower_body_policy
        return {
            "history": lower.obs_buffer.tolist(),
            "action": lower.action.tolist(),
            "gait": lower.gait_indices.tolist(),
            "time": self.time,
        }
