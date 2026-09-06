"""Planner-level API over RoboDojo for the dual ARX X5 arms.

Privileged tier: object poses come straight from the simulator's layout manager,
motion goes through cuRobo. This is the RoboDojo counterpart of
`capx.integrations.piper.robotwin_planner`.
"""

# Moved from third_party/cap-x/capx/integrations/arx_x5/robodojo_planner.py on 2026-08-26 (refactor P2); cap-x keeps a shim.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class RoboDojoPlannerPrivilegedApi(ApiBase):
    """Dual-arm ARX X5 control with privileged object state (RoboDojo)."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Callable[..., Any]]:
        fns = {
            "list_objects": self.list_objects,
            "get_object_pose": self.get_object_pose,
            "get_object_bbox_extent": self.get_object_bbox_extent,
            "get_gripper_pose": self.get_gripper_pose,
            "get_joint_positions": self.get_joint_positions,
            "goto_pose": self.goto_pose,
            "move_by_displacement": self.move_by_displacement,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "wait": self.wait,
            "go_home": self.go_home,
            "get_gripper_state": self.get_gripper_state,
            "get_step_budget": self.get_step_budget,
        }
        # tool-ablation hook: the low-level env can carry an exclusion list
        # (set from YAML via the simulator node's `api_exclude`)
        for name in getattr(self._env, "api_exclude", ()) or ():
            fns.pop(name, None)
        return fns

    def go_home(self, arm: str) -> None:
        """Return one arm to its initial (reset) configuration. Scoring only
        counts once BOTH arms are back near their starting pose, so call this
        for each arm after finishing manipulation.

        Args:
            arm: "left" or "right".
        """
        self._log_step("go_home", arm)
        self._env.go_home(arm)

    # --------------------------------------------------------------- perception

    def list_objects(self) -> list[str]:
        """List the names of every object in the scene.

        Returns:
            Names usable with get_object_pose()/get_object_bbox_extent(),
            e.g. ['box', 'car', 'hammer', 'shoe'].

        Example:
            objects = list_objects()
        """
        names = self._env.list_objects()
        self._log_step("list_objects", f"{len(names)} objects: {names}")
        return names

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get an object's world-frame pose.

        Args:
            object_name: Name of the object (see list_objects()).

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ orientation.

        Example:
            box_pos, box_quat = get_object_pose("box")
        """
        pos, quat = self._env.get_object_pose(object_name)
        self._log_step(
            "get_object_pose",
            f"{object_name}: pos={np.round(pos, 3).tolist()} quat={np.round(quat, 3).tolist()}",
        )
        return pos, quat

    def get_object_bbox_extent(self, object_name: str) -> np.ndarray:
        """Get an object's axis-aligned bounding-box size.

        Use this to reason about whether things fit: an object's footprint is
        extent[0] x extent[1], and a container's interior is roughly its extent
        minus wall thickness.

        Args:
            object_name: Name of the object (see list_objects()).

        Returns:
            extent: (3,) full XYZ side lengths in meters.

        Example:
            extent = get_object_bbox_extent("shoe")   # e.g. [0.182, 0.082, 0.079]
        """
        verts = self._env.get_object_bbox(object_name)
        extent = verts.max(axis=0) - verts.min(axis=0)
        self._log_step("get_object_bbox_extent", f"{object_name}: {np.round(extent, 3).tolist()}")
        return extent

    # ------------------------------------------------------------------- state

    def get_gripper_pose(self, arm: str) -> np.ndarray:
        """Get one arm's current fingertip (gripper tip) pose.

        Args:
            arm: "left" or "right".

        Returns:
            pose: (7,) [x, y, z, qw, qx, qy, qz] in world frame, at the
            fingertips — the same frame goto_pose() targets.
        """
        pose = self._env.get_ee_pose(arm)
        self._log_step("get_gripper_pose", f"{arm}: {np.round(pose, 3).tolist()}")
        return pose

    def get_step_budget(self) -> dict:
        """The episode's simulation-step budget.

        Every motion, gripper action, and wait() consumes simulation steps.
        When the budget runs out the episode ends immediately and is scored
        as-is. Check this to pace calibration experiments and retries — a
        typical single pick-and-place costs roughly 1500-2500 steps.

        Returns:
            dict with "used" and "max" (ints, simulation steps).
        """
        b = self._env.get_step_budget()
        self._log_step("get_step_budget", f"{b['used']}/{b['max']}")
        return b

    def get_gripper_state(self, arm: str) -> dict:
        """Observe the physical gripper directly: both finger positions and
        the jaw width. Use it to CALIBRATE the gripper frame empirically —
        e.g. read it at the start pose, or command a test orientation and read
        it again: the finger-to-finger direction is the jaw opening axis, and
        the vector from the wrist toward the finger midpoint is the tool axis.

        Args:
            arm: "left" or "right".

        Returns:
            dict with:
              "finger_a", "finger_b": (3,) world positions of the two fingers,
              "width": float, distance between them (jaw opening) in meters,
              "tip_pose": (7,) same as get_gripper_pose().
        """
        fa, fb = self._env.get_finger_positions(arm)
        pose = self._env.get_ee_pose(arm)
        width = float(np.linalg.norm(np.asarray(fa) - np.asarray(fb)))
        self._log_step("get_gripper_state", f"{arm}: width={width:.3f}")
        return {"finger_a": fa, "finger_b": fb, "width": width, "tip_pose": pose}

    def get_joint_positions(self, arm: str) -> np.ndarray:
        """Get one arm's six joint angles in radians.

        Args:
            arm: "left" or "right".

        Returns:
            joints: (6,) radians.
        """
        q = self._env.get_joints(arm)
        self._log_step("get_joint_positions", f"{arm}: {np.round(q, 3).tolist()}")
        return q

    # ------------------------------------------------------------------ motion

    def goto_pose(self, arm: str, position, quaternion_wxyz, max_tilt_deg: float = 0.0) -> None:
        """Move one arm so its FINGERTIPS reach a world-frame pose.

        The position is where the gripper fingertips end up. Raises if the
        pose is unreachable or was not actually reached; catch the error and
        try a nearby pose or the other arm rather than letting the program
        abort.

        The motion follows an approximately STRAIGHT Cartesian path from the
        current fingertip position to the target. There is NO automatic
        obstacle avoidance: if the straight line (or the object hanging below
        the jaws) passes through another object or a container wall, the arm
        will plow into it. Plan your own clearance waypoints — go up, across,
        then down.

        WARNING: this gripper's tool-frame convention is NOT the common
        "identity-points-up, [0,1,0,0]-points-down" one. Before your first
        grasp, calibrate it empirically with get_gripper_state(): command a
        test orientation, then read the finger positions back — they tell you
        which way the tool points and along which axis the jaws open for that
        quaternion.

        IMPORTANT: this small 6-DoF arm often cannot reach an exactly vertical
        top-down wrist away from its base. Pass max_tilt_deg=25 when
        approaching or grasping — the motion then automatically uses the
        nearest reachable orientation within that cone.

        Args:
            arm: "left" or "right".
            position: (3,) fingertip XYZ target in meters.
            quaternion_wxyz: (4,) WXYZ orientation (calibrate the convention
                via get_gripper_state(); see WARNING above).
            max_tilt_deg: allowed deviation from the requested orientation in
                degrees (0 = exact only; 25 recommended for grasping).
        """
        self._log_step(
            "goto_pose",
            f"{arm} -> pos={np.round(np.asarray(position, float), 3).tolist()} "
            f"quat={np.round(np.asarray(quaternion_wxyz, float), 3).tolist()} "
            f"tilt<={max_tilt_deg:g}",
        )
        self._env.goto_pose(arm, position, quaternion_wxyz, max_tilt_deg=max_tilt_deg)
        if self._webui_enabled:  # render is expensive; only pay it when logging
            self._log_step_update(images=self._env.render())

    def move_by_displacement(
        self, arm: str, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> None:
        """Move one arm's end-effector by a world-frame offset, keeping orientation.

        Args:
            arm: "left" or "right".
            x: Displacement along world X in meters.
            y: Displacement along world Y in meters.
            z: Displacement along world Z in meters (positive = up).

        Example:
            move_by_displacement("left", z=0.15)   # lift straight up
        """
        pose = self._env.get_ee_pose(arm)
        self._log_step("move_by_displacement", f"{arm} by [{x:.3f}, {y:.3f}, {z:.3f}]")
        # Reachability degradation: retry with a larger tilt allowance, and for
        # pure upward lifts progressively shorten the lift instead of failing —
        # "as high as reachable" is what every caller of a lift actually wants.
        scales = [1.0, 0.6, 0.35] if (x == 0.0 and y == 0.0 and z > 0.0) else [1.0]
        last_err = None
        for sc in scales:
            target = pose[:3] + np.array([x * sc, y * sc, z * sc], dtype=float)
            try:
                self._env.goto_pose(arm, target, pose[3:7], max_tilt_deg=30.0)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
        if last_err is not None:
            raise last_err
        if self._webui_enabled:
            self._log_step_update(images=self._env.render())

    def open_gripper(self, arm: str, fraction: float = 1.0) -> None:
        """Open one gripper.

        Args:
            arm: "left" or "right".
            fraction: 1.0 = fully open, 0.0 = fully closed.
        """
        self._log_step("open_gripper", f"{arm} -> {fraction:.2f}")
        self._env.set_gripper(arm, fraction)

    def close_gripper(self, arm: str, fraction: float = 0.0) -> None:
        """Close one gripper onto an object.

        Args:
            arm: "left" or "right".
            fraction: 0.0 = fully closed, 1.0 = fully open.
        """
        self._log_step("close_gripper", f"{arm} -> {fraction:.2f}")
        self._env.set_gripper(arm, fraction)

    def wait(self, steps: int = 30) -> None:
        """Let the simulation settle without commanding the arms.

        Useful right after releasing an object so it comes to rest before you
        read its pose again.

        Args:
            steps: Number of simulation steps to advance.
        """
        self._log_step("wait", f"{steps} steps")
        self._env.settle(steps)


__all__ = ["RoboDojoPlannerPrivilegedApi"]


# Self-registering: importing this module makes the API resolvable through
# ``capx.integrations.base_api.get_api`` (cap-x launcher configs).
def register() -> None:
    """Register the API with cap-x's registry (idempotent)."""
    from capx.integrations.base_api import register_api

    register_api("RoboDojoPlannerPrivilegedApi", RoboDojoPlannerPrivilegedApi)


register()
