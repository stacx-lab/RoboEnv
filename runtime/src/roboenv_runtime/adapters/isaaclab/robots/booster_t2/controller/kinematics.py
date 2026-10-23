"""Forward kinematics of the robot alone, for observations that need link poses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pinocchio as pin

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import RobotConfig
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.rotations import matrix_to_quat, quat_to_matrix


class RobotKinematics:
    """Forward kinematics of the robot config's URDF with the joints not named locked at zero,
    root link at the origin. Needs no simulation, so it works the same on the real robot. Mimic
    joints are independent joints here.

    Args:
        robot: The robot config.
        joint_names: Order of the joint positions `frame_poses` takes; distinct URDF joints.
        frames: URDF links whose poses `frame_poses` returns, in that order.

    Raises:
        ValueError: A joint or a frame is not in the URDF, or a joint is named twice.
    """

    def __init__(
        self, robot: RobotConfig, joint_names: Sequence[str], frames: Sequence[str]
    ) -> None:
        full = pin.buildModelFromUrdf(str(robot.urdf))
        unknown = set(joint_names) - set(full.names[1:])
        if unknown or len(set(joint_names)) != len(joint_names):
            raise ValueError(f"unknown or repeated joints: {sorted(unknown) or joint_names}")
        locked = [full.getJointId(n) for n in full.names[1:] if n not in set(joint_names)]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.data = self.model.createData()
        self.joint_names: tuple[str, ...] = tuple(joint_names)
        """Order of `frame_poses`' joint positions."""
        self._q_index = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names]
        )
        self._frames = []
        for name in frames:
            if not self.model.existFrame(name, pin.FrameType.BODY):
                raise ValueError(f"no link {name!r} in {robot.urdf.name}")
            self._frames.append(self.model.getFrameId(name, pin.FrameType.BODY))

    def frame_poses(
        self, joint_pos: np.ndarray, base_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pose of each frame relative to the root link origin, along world axes.

        Args:
            joint_pos: Joint positions, `joint_names` order. Shape (num_joints,), rad.
            base_quat: Root link orientation in the world frame, wxyz. Shape (4,).

        Returns:
            Positions, shape (num_frames, 3), m; orientations wxyz, shape (num_frames, 4).
        """
        q = pin.neutral(self.model)
        q[self._q_index] = joint_pos
        pin.framesForwardKinematics(self.model, self.data, q)
        base = quat_to_matrix(base_quat)
        positions = np.empty((len(self._frames), 3))
        quats = np.empty((len(self._frames), 4))
        for i, frame in enumerate(self._frames):
            placement = self.data.oMf[frame]
            positions[i] = base @ placement.translation
            quats[i] = matrix_to_quat(base @ placement.rotation)
        return positions, quats
