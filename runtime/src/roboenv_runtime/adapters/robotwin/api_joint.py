"""Route B: joint-level control API for RoboTwin dual Piper arms.

No motion planner: the agent solves IK (pyroki server) and commands joint
targets directly (PD drive, not collision-checked). Mirrors the Franka
"reduced" APIs.
"""

# Moved from third_party/cap-x/capx/integrations/piper/robotwin_joint.py on 2026-08-26 (refactor P2); cap-x keeps a shim.

from __future__ import annotations

from typing import Any

import numpy as np
from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.motion.pyroki import init_pyroki

PIPER_ARM_DOF = 6


class PiperJointControlApi(ApiBase):
    """Joint-level control of a dual-arm Piper robot (6 DoF per arm).

    Conventions:
    - End-effector poses are in the WORLD frame, quaternions WXYZ.
    - Joint angles are radians, 6 per arm.
    - Gripper fraction: 0.0 = fully closed, 1.0 = fully open.
    - Motions are NOT collision-checked: decompose large motions into waypoints.
    """

    def __init__(
        self,
        env: BaseEnv,
        pyroki_url_left: str = "http://127.0.0.1:8127",
        pyroki_url_right: str | None = None,
    ) -> None:
        super().__init__(env)
        self.ik_solve_left = init_pyroki(pyroki_url_left)
        self.ik_solve_right = init_pyroki(pyroki_url_right or pyroki_url_left)

    def functions(self) -> dict[str, Any]:
        return {
            "list_objects": self.list_objects,
            "get_object_pose": self.get_object_pose,
            "get_gripper_pose_left": self.get_gripper_pose_left,
            "get_gripper_pose_right": self.get_gripper_pose_right,
            "get_joint_positions_left": self.get_joint_positions_left,
            "get_joint_positions_right": self.get_joint_positions_right,
            "solve_ik_left": self.solve_ik_left,
            "solve_ik_right": self.solve_ik_right,
            "move_to_joints_left": self.move_to_joints_left,
            "move_to_joints_right": self.move_to_joints_right,
            "move_to_joints_both": self.move_to_joints_both,
            "open_gripper_left": self.open_gripper_left,
            "close_gripper_left": self.close_gripper_left,
            "open_gripper_right": self.open_gripper_right,
            "close_gripper_right": self.close_gripper_right,
        }

    # ------------------------- perception -------------------------

    def list_objects(self) -> list[str]:
        """List the names of all manipulable objects in the scene.

        Returns:
            List of object name strings usable with get_object_pose().
        """
        return self._env.list_objects()

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the world-frame pose of an object.

        Args:
            object_name: Name of the object (see list_objects()).

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        return self._env.get_object_pose(object_name)

    def get_gripper_pose_left(self) -> np.ndarray:
        """Get the left end-effector pose in the world frame.

        Returns:
            (7,) array [x, y, z, qw, qx, qy, qz].
        """
        return self._env.get_arm_pose("left")

    def get_gripper_pose_right(self) -> np.ndarray:
        """Get the right end-effector pose in the world frame.

        Returns:
            (7,) array [x, y, z, qw, qx, qy, qz].
        """
        return self._env.get_arm_pose("right")

    def get_joint_positions_left(self) -> np.ndarray:
        """Get the LEFT arm's measured joint positions.

        Returns:
            (7,) array: 6 arm joint angles in radians + gripper fraction (0..1).
        """
        return self._env.get_joint_positions("left")

    def get_joint_positions_right(self) -> np.ndarray:
        """Get the RIGHT arm's measured joint positions.

        Returns:
            (7,) array: 6 arm joint angles in radians + gripper fraction (0..1).
        """
        return self._env.get_joint_positions("right")

    # ------------------------- IK -------------------------

    def solve_ik_left(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the LEFT arm.

        Args:
            position: (3,) target end-effector XYZ in the world frame.
            quaternion_wxyz: (4,) target orientation WXYZ in the world frame.

        Returns:
            (6,) joint angles in radians.

        Example:
            >>> joints = solve_ik_left(np.array([-0.2, -0.1, 0.9]), np.array([-0.5, 0.5, -0.5, -0.5]))
            >>> move_to_joints_left(joints)
        """
        target = self._env.endpose_to_ee_link_wxyz_xyz("left", position, quaternion_wxyz)
        cfg = self.ik_solve_left(target_pose_wxyz_xyz=target, prev_cfg=None)
        return np.asarray(cfg, dtype=np.float64).reshape(-1)[:PIPER_ARM_DOF]

    def solve_ik_right(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve inverse kinematics for the RIGHT arm.

        Args:
            position: (3,) target end-effector XYZ in the world frame.
            quaternion_wxyz: (4,) target orientation WXYZ in the world frame.

        Returns:
            (6,) joint angles in radians.
        """
        target = self._env.endpose_to_ee_link_wxyz_xyz("right", position, quaternion_wxyz)
        cfg = self.ik_solve_right(target_pose_wxyz_xyz=target, prev_cfg=None)
        return np.asarray(cfg, dtype=np.float64).reshape(-1)[:PIPER_ARM_DOF]

    # ------------------------- motion (direct joint control) -------------------------

    def move_to_joints_left(self, joints: np.ndarray) -> None:
        """Drive the LEFT arm to target joint positions (blocking, no planning).

        Args:
            joints: (6,) target joint angles in radians.
        """
        self._env.move_to_joints_blocking(joints, "left")

    def move_to_joints_right(self, joints: np.ndarray) -> None:
        """Drive the RIGHT arm to target joint positions (blocking, no planning).

        Args:
            joints: (6,) target joint angles in radians.
        """
        self._env.move_to_joints_blocking(joints, "right")

    def move_to_joints_both(self, left_joints: np.ndarray, right_joints: np.ndarray) -> None:
        """Drive BOTH arms simultaneously to target joint positions (blocking).

        Args:
            left_joints: (6,) target joint angles for the left arm in radians.
            right_joints: (6,) target joint angles for the right arm in radians.
        """
        self._env.move_to_joints_blocking_both(left_joints, right_joints)

    # ------------------------- gripper -------------------------

    def open_gripper_left(self, fraction: float = 1.0) -> None:
        """Open the LEFT gripper.

        Args:
            fraction: Target opening, 0.0 (closed) to 1.0 (fully open).
        """
        self._env.set_gripper_blocking("left", fraction)

    def close_gripper_left(self, fraction: float = 0.0) -> None:
        """Close the LEFT gripper.

        Args:
            fraction: Target opening, 0.0 (fully closed) to 1.0 (open).
        """
        self._env.set_gripper_blocking("left", fraction)

    def open_gripper_right(self, fraction: float = 1.0) -> None:
        """Open the RIGHT gripper.

        Args:
            fraction: Target opening, 0.0 (closed) to 1.0 (fully open).
        """
        self._env.set_gripper_blocking("right", fraction)

    def close_gripper_right(self, fraction: float = 0.0) -> None:
        """Close the RIGHT gripper.

        Args:
            fraction: Target opening, 0.0 (fully closed) to 1.0 (open).
        """
        self._env.set_gripper_blocking("right", fraction)


__all__ = ["PiperJointControlApi"]


def register() -> None:
    """Register the API with cap-x's registry (idempotent)."""
    from capx.integrations.base_api import register_api

    register_api("PiperJointControlApi", PiperJointControlApi)


register()  # idempotent: make the API resolvable through capx.integrations.base_api.get_api
