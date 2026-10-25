"""Runs the T2 loco-manipulation policy: the robot's proprioception and a command in, joint
position targets out. Shared by simulation and the real robot."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.kinematics import RobotKinematics
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.policy import OnnxPolicy, PolicyConfig
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import RobotConfig
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.rotations import (
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_to_matrix,
    yaw_of,
    yaw_quat,
)
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.interface import SIDES, Pose


@dataclass(frozen=True)
class RobotState:
    """Proprioception for one policy step, as the joint encoders and the trunk IMU report it."""

    joint_pos: np.ndarray
    """Joint positions, `LocoManipController.joint_names` order. Shape (num_joints,), rad."""
    joint_vel: np.ndarray
    """Joint velocities, same order. Shape (num_joints,), rad/s."""
    base_quat: np.ndarray
    """Orientation of the root body in the world frame, wxyz. Shape (4,)."""
    base_ang_vel: np.ndarray
    """Angular velocity of the root body in its own frame (the gyro). Shape (3,), rad/s."""


@dataclass(frozen=True)
class PolicyCommand:
    """What the policy is asked to do at one step."""

    velocity: np.ndarray
    """Base velocity in the trunk heading frame, within the trained range: vx m/s, vy m/s,
    wz rad/s. Shape (3,)."""
    palm_targets: dict[str, Pose]
    """Palm target per side, command frame (trunk origin, trunk heading, world z up)."""


class LocoManipController:
    """The loco-manipulation policy: a `RobotState` and a `PolicyCommand` in, joint position
    targets out, once per policy step.

    Runs no simulation and keeps no command. Its only state is the policy's own memory (the
    observation history, the last action and the gait phase clock), which `reset` clears. The
    caller applies the targets with `gains`: Booster's PD in simulation, the joint drivers on the
    robot. Joints the policy does not drive (the head) get their default position.

    Args:
        config: Policy config. Default: `configs/locomanip_policy.yaml`.
        robot: Robot config. Default: the one `config` names.
        policy: The exported actor. Default: the one `config` names.

    Raises:
        ValueError: The policy file disagrees with `config` or with the robot config (joints,
            default pose or PD gains).
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        robot: RobotConfig | None = None,
        policy: OnnxPolicy | None = None,
    ) -> None:
        self.config = config if config is not None else PolicyConfig()
        self.robot = robot if robot is not None else RobotConfig(self.config.robot)
        self.policy = policy if policy is not None else OnnxPolicy(self.config.onnx)
        md = self.policy.metadata
        self.joint_names: tuple[str, ...] = md.joint_names
        """The robot's joints in the policy's model order: the order of `RobotState` and of the
        targets."""
        self.kinematics = RobotKinematics(
            self.robot, self.joint_names, [self.config.palm_frames[s] for s in SIDES]
        )
        self.default_joint_pos = self.robot.keyframe_joint_pos(self.joint_names)
        """Keyframe joint positions, `joint_names` order. Shape (num_joints,), rad."""
        self.gains = self.robot.pd_gains(self.joint_names)
        """PD gains and motor limits the targets are meant for, `joint_names` order."""
        self._check(md)
        self._joint_idx = np.array([self.joint_names.index(n) for n in self.config.joints])
        self._history = {
            name: deque(maxlen=h)
            for name, h in zip(md.observation_names, md.history_lengths, strict=True)
            if h > 0
        }
        self.reset()

    def _check(self, md) -> None:
        if md.action_joint_names != self.config.joints:
            raise ValueError("policy file and policy config joints differ")
        if not np.allclose(md.default_joint_pos, self.default_joint_pos, atol=1e-3):
            raise ValueError("policy file and robot config default poses differ")
        if not (
            np.allclose(md.stiffness, self.gains.stiffness, rtol=1e-3)
            and np.allclose(md.damping, self.gains.damping, rtol=1e-3)
        ):
            raise ValueError("policy file and robot config PD gains differ")

    @property
    def policy_dt(self) -> float:
        """Policy period, s."""
        return self.robot.timestep * self.config.decimation

    def reset(self) -> None:
        """Clear the policy's memory: observation history, last action, gait phase clock."""
        self._last_action = np.zeros(self.policy.action_dim)
        for history in self._history.values():
            history.clear()
        self._steps = 0

    def act(self, state: RobotState, command: PolicyCommand) -> np.ndarray:
        """One policy step. Returns joint position targets, `joint_names` order, shape
        (num_joints,), rad. Advances the policy's memory."""
        action = self.policy(self._observation(state, command))
        self._last_action = action
        target = self.default_joint_pos.copy()
        target[self._joint_idx] += self.policy.metadata.action_scale * action
        self._steps += 1
        return target

    def palm_poses(self, state: RobotState) -> dict[str, Pose]:
        """Palm pose per side in the command frame (trunk origin, trunk heading, world z up)."""
        positions, quats = self.kinematics.frame_poses(state.joint_pos, state.base_quat)
        yaw_inv = quat_conj(yaw_quat(yaw_of(state.base_quat)))
        return {
            side: Pose.of(quat_rotate(yaw_inv, positions[i]), quat_mul(yaw_inv, quats[i]))
            for i, side in enumerate(SIDES)
        }

    def _observation(self, state: RobotState, command: PolicyCommand) -> np.ndarray:
        """The actor observation, terms in the policy's order. Shape (obs_dim,). Pushes the
        current frame into the history terms; a fresh history is filled with it."""
        frames = self._frames(state, command)
        parts = []
        for name in self.policy.metadata.observation_names:
            history = self._history.get(name)
            if history is None:
                parts.append(frames[name])
                continue
            history.extend([frames[name]] * (history.maxlen if not history else 1))
            parts.extend(history)
        obs = np.concatenate(parts)
        if obs.shape != (self.policy.obs_dim,):
            raise RuntimeError(
                f"observation has {obs.shape[0]} values, policy expects {self.policy.obs_dim}"
            )
        return obs

    def _frames(self, state: RobotState, command: PolicyCommand) -> dict[str, np.ndarray]:
        """Current value of every observation term, by name."""
        c = self.config
        quat = state.base_quat
        if np.linalg.norm(command.velocity) < c.standing_threshold:
            phase = np.zeros(2)
        else:
            t = self._steps * self.policy_dt
            angle = 2.0 * math.pi * (t % c.gait_period) / c.gait_period
            phase = np.array([math.sin(angle), math.cos(angle)])
        frames = {
            "base_ang_vel": np.asarray(state.base_ang_vel, dtype=float),
            "projected_gravity": quat_rotate(quat_conj(quat), [0.0, 0.0, -1.0]),
            "command": np.asarray(command.velocity, dtype=float),
            "phase": phase,
            "joint_pos": (state.joint_pos - self.default_joint_pos)[self._joint_idx],
            "joint_vel": np.asarray(state.joint_vel, dtype=float)[self._joint_idx],
            "actions": self._last_action.copy(),
        }
        yaw = yaw_quat(yaw_of(quat))
        yaw_inv = quat_conj(yaw)
        positions, quats = self.kinematics.frame_poses(state.joint_pos, quat)
        poses, errors = [], []
        for i, side in enumerate(SIDES):
            target = command.palm_targets[side]
            frames[f"{side}_palm_cmd"] = np.concatenate([target.pos, target.quat])
            pos, q = positions[i], quats[i]
            q_b = quat_mul(yaw_inv, q)
            poses.append(np.concatenate([quat_rotate(yaw_inv, pos), -q_b if q_b[0] < 0 else q_b]))
            rel = quat_to_matrix(quat_mul(quat_conj(q), quat_mul(yaw, target.quat)))
            errors.append(
                np.concatenate(
                    [quat_rotate(yaw_inv, quat_rotate(yaw, target.pos) - pos), rel[:, 0], rel[:, 1]]
                )
            )
        frames["palm_pose"] = np.concatenate(poses)
        frames["palm_pose_err"] = np.concatenate(errors)
        return frames
