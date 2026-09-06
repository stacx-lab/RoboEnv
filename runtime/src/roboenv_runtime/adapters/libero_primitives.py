"""Closed-loop primitives for the LIBERO (MuJoCo / Panda) adapter.

The cap-x ``FrankaLiberoPrivilegedApi`` gives an IK ``goto_pose``, a grasp
sampler and the gripper — enough for pick-and-place, not for the drawer, stove
and multi-object tasks.  Everything here is verified against MuJoCo ground truth
and reports what it measured, so a program can branch on the result.

Seeing the scene
    ``get_object_extents``    measured bounding box, top and bottom of an object
    ``get_region_boxes``      the site boxes the BDDL goal is actually scored against
    ``get_joint_info``        range/type/axis of every articulated joint
    ``body_position``         any body, including the ``_2`` instance of a type
    ``in_contact``            the contact the ``On``/``In`` predicates demand
    ``clearance_at``          free space around a point
    ``get_step_budget``       steps left before LIBERO's hard 4000-step horizon

Taking hold
    ``plan_pinch``            where and from which side to pinch, from the object's own
                              geometry: rim, handle or body, never a fingertip inside the
                              object, never a direction jammed against a drawer wall
    ``grasp_object``          run that plan, lift, verify, retry lower

Putting down
    ``fit_yaw``/``turn_held``  how far to turn a held object about the vertical for it to
                              fit a compartment, and the wrist turn that does it
    ``carry_to``/``descend_over``  move the OBJECT (not the fingertips) and re-centre
    ``place_gently``          lower until it rests; stop the moment the support slides
    ``place_on``              onto another object (``On a b`` allows 3 cm)
    ``place_on_region`` / ``place_in_region``  onto/into a scored box region

Articulation
    ``open_articulated``      pull a drawer/door open in verified strokes
    ``close_articulated``     push it shut with closed fingers
    ``turn_knob``             pinch a dial and rotate the wrist about the joint's axis

Round 4 (2026-08-27)
    ``insert_flat_into_region``  front-load a flat object into a compartment open only at
                              the front (one wrist turn, origin servoed, no retreat)
    ``palm_clearance``        would the hand body land on the object at this pinch?
                              (``top_pick`` raises a blocked pinch to the deepest clear one)
    ``align_held_to_region``  tilt a held object onto a rolled site's axis (a wine rack)
    ``place_on(retries=)``, ``scored_name=`` on the place tools, ``open_articulated``
    re-hooks (``rehooks``) when the arm, not the pinch, stalled

Reaching low
    ``reachable_tool_quat``   the flattest tool pitch whose IK reaches a set of points —
                              a level wrist runs out of solutions a few centimetres above
                              the table, which is what a bottom drawer needs
    ``park``                  back out along the tool axis and lift clear (after every
                              articulation move)

Plus ``move_by_displacement``, ``check_reachable``, ``get_gripper_opening``,
``open_fingers``/``close_fingers`` and ``checkpoint``/``restore`` (practice only).

Frames: the cap-x API works in the **robot base frame** (objects, hand pose,
``goto_pose`` targets); MuJoCo reports world coordinates.  ``base_from_world``
/ ``world_from_base`` convert, and every tool here speaks the base frame so a
program can mix them with ``get_object_pose`` freely.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "LiberoPrimitives",
    "directed_joint_progress",
    "frame_quat_wxyz",
    "mat_to_quat_wxyz",
    "quat_wxyz_to_mat",
    "tool_down_quat_wxyz",
    "tool_horizontal_quat_wxyz",
    "tool_pitched_quat_wxyz",
]

TCP_OFFSET_HAND = np.array([0.0, 0.0, 0.1])  # fingertip midpoint, in the hand frame
MJ_SLIDE, MJ_HINGE = 2, 3
MJ_PLANE = 0
MJ_SPHERE, MJ_CAPSULE, MJ_CYLINDER, MJ_BOX, MJ_MESH = 2, 3, 5, 6, 7
MJ_SITE_BOX = 6
GRIPPER_MAX_OPENING = 0.077  # measured finger-pad gap of the LIBERO Panda, fully open
HELD_OPENING = 0.004  # a smaller gap after close_gripper means the fingers met on nothing
# Finger geometry of the LIBERO Panda, measured from the collision meshes 2026-08-27
# (hand frame, relative to the fingertip midpoint = hand + TCP_OFFSET_HAND): the finger
# mesh spans -0.0375 m (finger base) .. +0.016 m (tip end) along the tool axis, each
# finger is 0.0265 m thick OUTBOARD of its pad face and 0.021 m wide; the hand body's
# box begins 0.024 m above the fingertip midpoint and spans +-0.104 m along the close
# axis.  A top-down grasp's free space is therefore sampled at these heights above the
# fingertip (the last one is already in the hand zone) ...
FINGER_PROBES = (0.0, 0.02, 0.0375)
# ... and at the finger's OUTER face, this far outboard of the pad contact point: a pad
# point that clears a drawer wall by a few mm can still have the finger body inside it
FINGER_THICKNESS_M = 0.0265
#: how far the lower pad's outer face dips below the tool point on a level wrist (m)
PAD_BELOW_TOOL_M = 0.01


def directed_joint_progress(
    qpos: float,
    start_qpos: float,
    direction: int,
    span: float,
) -> float:
    """Fraction of a joint range moved in the requested signed direction."""
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    width = float(span)
    if width <= 0.0:
        raise ValueError("span must be positive")
    return float(direction) * (float(qpos) - float(start_qpos)) / width
# ... and the finger tips extend this far PAST the fingertip midpoint along the tool axis
# (a pinch on a 1.8 cm box with the midpoint 1.4 cm below its top drives the tips into the
# table and the descent stalls short)
FINGERTIP_BELOW_TCP = 0.016
# downward tool pitches tried, flattest first, when a horizontal push has to reach low
PUSH_PITCHES = (0.0, 15.0, 30.0, 45.0, 60.0)
# the hand body (palm) begins this far behind the tool point along the tool axis, and its
# box spans this much either side of the tool point along the finger-closing axis and
# across it (measured 2026-08-27): a top-down pinch deeper than HAND_BOX_ABOVE_TCP below an
# object's highest material puts the palm ON the object and the descent stalls there (the
# wine bottle of libero.90_26: every pinch 4 cm above its base stalled 11 cm short, exactly
# at top_z - 0.024, and the same descent with the bottle removed reached the point)
HAND_BOX_ABOVE_TCP = 0.024
HAND_HALF_ALONG_CLOSE = 0.104
HAND_HALF_ACROSS = 0.0105


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"quaternion must be finite and non-zero, got {q.tolist()}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got shape {m.shape}")
    if not np.allclose(m.T @ m, np.eye(3), atol=1e-5) or np.linalg.det(m) < 0.0:
        raise ValueError("expected a proper rotation matrix (orthonormal, det +1)")
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array(
        [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    )


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` radians about the unit vector ``axis`` (Rodrigues)."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / max(np.linalg.norm(ax), 1e-9)
    skew = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def _frame_quat(hand_z: np.ndarray, close_dir: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) of the hand frame whose z (tool axis) is ``hand_z`` and whose y
    (finger travel) is ``close_dir`` projected perpendicular to z; the sign of y is chosen
    so hand x keeps a positive world-x component (the Panda's elbow-friendly branch).
    """
    z = np.asarray(hand_z, dtype=float).reshape(3)
    zn = float(np.linalg.norm(z))
    if not np.isfinite(zn) or zn < 1e-6:
        raise ValueError(f"tool_axis must be a finite, non-zero direction, got {z.tolist()}")
    z = z / zn
    y = np.asarray(close_dir, dtype=float).reshape(3)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"close_dir must be finite, got {y.tolist()}")
    y = y - z * (y @ z)
    if np.linalg.norm(y) < 1e-6:
        # a degenerate close_dir (parallel to the tool, or zero): close along the base
        # axis least aligned with the tool (cross(z, x) is zero for a tool along x)
        y = np.cross(z, np.eye(3)[int(np.argmin(np.abs(z)))])
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    if x[0] < 0:
        y, x = -y, -x
    return _mat_to_quat_wxyz(np.column_stack([x, y, z]))


def frame_quat_wxyz(tool_axis: Any, close_dir: Any) -> np.ndarray:
    """Hand quaternion (wxyz) from two base-frame directions: the TOOL axis (points out of
    the fingertips) and the FINGER-CLOSING direction.

    The single source of every grasp orientation: ``close_dir`` is projected perpendicular
    to the tool axis and normalised, the third axis completes a right-handed frame, and the
    sign of the finger axis is chosen so hand x keeps a positive base-x component (the
    Panda's elbow-friendly branch — two of the four equivalent frames are wrist-flipped
    and the IK finds them only near a joint limit).  ``tool_down_quat_wxyz`` is
    ``frame_quat_wxyz([0, 0, -1], close_dir)`` and ``tool_horizontal_quat_wxyz`` is
    ``frame_quat_wxyz(point_dir, close_dir)``.  A ``close_dir`` parallel to the tool axis
    (or zero) is replaced by ``cross(tool, e)`` with ``e`` the base axis least aligned with
    the tool; a zero or non-finite ``tool_axis`` raises.  The branch rule is a step: when
    hand x comes out perpendicular to base x (a top-down pinch closing along base x) the
    SIGN of ``close_dir`` decides the wrist branch, so ``[1, 0, 0]`` and ``[-1, 0, 0]``
    give frames 180 degrees apart while ``[0, 1, 0]`` and ``[0, -1, 0]`` give the same
    one — pass the sign you mean.  Over the worker protocol the result is a JSON list.
    """
    return _frame_quat(np.asarray(tool_axis, dtype=float), np.asarray(close_dir, dtype=float))


def quat_wxyz_to_mat(quat_wxyz: Any) -> np.ndarray:
    """3x3 rotation matrix of a wxyz quaternion (normalised first).

    Applied to a hand quaternion the COLUMNS are the hand axes in the base frame: column 2
    is the tool axis (out of the fingertips; ``wrist + TCP_OFFSET_HAND[2] * column2`` is the
    API's tool point), column 1 the finger-closing axis, column 0 the axis across the hand.
    Index it as ``np.asarray(quat_wxyz_to_mat(q))[:, 2]``: in the live loop the value
    arrives as a JSON list, where ``[:, 2]`` raises.  A zero or non-finite quaternion
    raises ValueError.
    """
    return _quat_wxyz_to_mat(np.asarray(quat_wxyz, dtype=float))


def mat_to_quat_wxyz(mat: Any) -> np.ndarray:
    """The wxyz quaternion of a 3x3 rotation matrix (Shepperd's branch selection, so no
    division by a vanishing term near 180-degree rotations).  Inverse of
    ``quat_wxyz_to_mat`` for a PROPER rotation matrix (orthonormal, det +1: anything else
    — a reflection, a scaled or 4x4 matrix — raises ValueError); the sign of the result is
    not canonical (q and -q are the same rotation).
    """
    return _mat_to_quat_wxyz(np.asarray(mat, dtype=float))


def tool_down_quat_wxyz(close_dir: Any = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Hand orientation with the tool pointing straight down and the fingers closing
    along the horizontal direction ``close_dir`` (Panda fingers travel along hand y).

    ``(0, 1, 0)`` reproduces the ``[0, 1, 0, 0]`` top-down quaternion the LIBERO wins used.
    """
    return _frame_quat(np.array([0.0, 0.0, -1.0]), np.asarray(close_dir, dtype=float))


def _horizontal(point_dir: Any, what: str) -> np.ndarray:
    d = np.asarray(point_dir, dtype=float).reshape(3)
    flat = np.array([d[0], d[1], 0.0])
    if np.linalg.norm(flat) < 1e-6:
        raise ValueError(f"{what} must have a horizontal component, got {d.tolist()}")
    return flat / np.linalg.norm(flat)


def tool_horizontal_quat_wxyz(point_dir: Any, close_dir: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Hand orientation with the tool axis LEVEL, pointing along the horizontal part of
    ``point_dir`` (e.g. at a drawer front; a vertical component is dropped, a vertical
    vector raises) and the fingers closing along ``close_dir`` (default: vertical, to
    pinch a horizontal handle bar).
    """
    return _frame_quat(_horizontal(point_dir, "point_dir"), np.asarray(close_dir, dtype=float))


def tool_pitched_quat_wxyz(
    point_dir: Any, pitch_deg: float = 0.0, close_dir: Any = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Hand orientation pointing along ``point_dir`` but tilted ``pitch_deg`` degrees
    DOWNWARD, about the horizontal axis perpendicular to ``point_dir``.

    Pitch is what buys reach and push force low over the table.  Two measured facts
    (LIBERO Panda, 2026-08-27) — keep them apart: (a) the SEEDLESS IK probe behind
    :meth:`LiberoPrimitives.check_reachable` does not converge for a level wrist at some
    low heights (base-frame z 0.02-0.03, residual 4.3-4.6 cm) although the same pose
    solves exactly when seeded from a neighbouring solution — a solver local minimum, not
    a reach boundary; (b) a level wrist pushing along +x low over the table (0.55, 0,
    0.03) genuinely violates the wrist joint limits and needs ~60 deg of pitch under any
    solver — the dependence is on the push DIRECTION.  Tilting the tool down keeps the
    elbow and the hand body up while the fingertips come down, at the price of a
    downward force component.  ``pitch_deg=0`` is exactly
    :func:`tool_horizontal_quat_wxyz`; a negative pitch tilts UP; at 90 deg the tool is
    straight down and the default ``close_dir`` (vertical) degenerates — the fingers are
    then laid along ``cross(z, x)``.  ``close_dir`` is projected perpendicular to the
    TILTED tool, so at 45-60 deg a "vertical" pinch on a horizontal bar is diagonal.
    ``point_dir`` is taken as its HORIZONTAL part, so the pitch is always measured from
    level (a vector with a vertical component used to pitch from that slope, and
    ``[1, 0, 1]`` at 30 deg still pointed UP); a purely vertical ``point_dir`` has nothing
    to pitch about and comes back as the plain frame along it.  Assumes base-frame z is
    up (true for this arm).
    """
    d = np.asarray(point_dir, dtype=float).reshape(3)
    if np.linalg.norm(d[:2]) < 1e-6:  # already vertical: nothing to pitch about
        return _frame_quat(d, np.asarray(close_dir, dtype=float))
    d = _horizontal(d, "point_dir")
    angle = np.radians(float(pitch_deg))
    tool = np.cos(angle) * d + np.sin(angle) * np.array([0.0, 0.0, -1.0])
    return _frame_quat(tool, np.asarray(close_dir, dtype=float))


class LiberoPrimitives:
    """Builds the extra tools for one :class:`roboenv_runtime.adapters.libero.LiberoAdapter`."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._checkpoints: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ plumbing

    @property
    def lle(self) -> Any:
        """The cap-x ``FrankaLiberoEnv`` wrapper."""
        return self.adapter._lle

    @property
    def api(self) -> Any:
        """The cap-x ``FrankaLiberoPrivilegedApi`` (owns ``goto_pose`` and the IK warm start)."""
        for api in getattr(self.adapter._env, "_apis", {}).values():
            if hasattr(api, "goto_pose") and hasattr(api, "_pks"):
                return api
        raise RuntimeError("LIBERO privileged API not attached")

    @property
    def sim(self) -> Any:
        """Robosuite ``MjSim``."""
        return self.adapter._sim()

    def _raw(self) -> tuple[Any, Any]:
        sim = self.sim
        return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)

    def base_from_world(self, p: Any) -> np.ndarray:
        """World → robot-base frame (the frame of the cap-x API)."""
        wxyz_xyz = np.asarray(self.lle.base_link_wxyz_xyz, dtype=float)
        rot = _quat_wxyz_to_mat(wxyz_xyz[:4])
        return rot.T @ (np.asarray(p, dtype=float) - wxyz_xyz[4:7])

    def world_from_base(self, p: Any) -> np.ndarray:
        """Robot-base → world frame."""
        wxyz_xyz = np.asarray(self.lle.base_link_wxyz_xyz, dtype=float)
        return _quat_wxyz_to_mat(wxyz_xyz[:4]) @ np.asarray(p, dtype=float) + wxyz_xyz[4:7]

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Hand position and wxyz quaternion in the base frame."""
        obs = self.lle.get_observation()["robot_cartesian_pos"]
        return np.asarray(obs[:3], dtype=float), np.asarray(obs[3:7], dtype=float)

    def tcp(self) -> np.ndarray:
        """The API tool point in the base frame (see :meth:`get_hand_pose`); an ndarray
        for arithmetic in a program.
        """
        """Fingertip midpoint (the point ``goto_pose`` targets), base frame."""
        pos, quat = self.hand_pose()
        return pos + _quat_wxyz_to_mat(quat) @ TCP_OFFSET_HAND

    def goto(
        self,
        position: Any,
        quat_wxyz: Any,
        *,
        tolerance_m: float = 0.005,
        iterations: int = 3,
        joint_tolerance: float = 0.006,
        max_steps: int = 80,
        max_iterations: int = 8,
    ) -> float:
        """Move the fingertip midpoint to ``position`` (base frame) with orientation
        ``quat_wxyz`` and return the remaining position error.

        The API's IK (``solve_ik_vel_cost``) trades pose accuracy for smoothness, so a
        1 cm request can come back as a null-space wiggle with no hand motion.  This
        closes the loop: measure the fingertips, add the residual to the commanded
        target, solve again — up to ``iterations`` times or until within ``tolerance_m``.

        A long move (across the table, or away from a drawer the hand has just pulled)
        does not fit in ``iterations`` * ``max_steps`` controller steps, and stopping
        there used to leave the hand 5-20 cm short with no warning.  So the loop keeps
        going past ``iterations`` while the error is still shrinking by a fifth each
        time, up to ``max_iterations``.
        """
        api = self.api
        target = np.asarray(position, dtype=float).reshape(3)
        quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
        command = target.copy()
        error = float("inf")
        previous = float("inf")
        for step in range(max(int(iterations), int(max_iterations))):
            if step >= int(iterations) and error > 0.8 * previous:
                break  # no longer closing in: more controller steps would be wasted
            previous = error
            hand_target = command + _quat_wxyz_to_mat(quat) @ (-TCP_OFFSET_HAND)
            if api.cfg is None:
                api.cfg = api._pks.solve_ik(
                    robot=api._robot,
                    target_link_name=api._target_link_name,
                    target_wxyz=quat,
                    target_position=hand_target,
                )
            else:
                api.cfg = api._pks.solve_ik_vel_cost(
                    robot=api._robot,
                    target_link_name=api._target_link_name,
                    target_position=hand_target,
                    target_wxyz=quat,
                    prev_cfg=api.cfg,
                )
            joints = np.asarray(api.cfg[:-1], dtype=np.float64).reshape(7)
            allowed = self._steps_left()
            if allowed <= 1:
                break  # the horizon is hard: one step past it kills the whole program
            self.lle.move_to_joints_blocking(
                joints, tolerance=joint_tolerance, max_steps=min(int(max_steps), allowed)
            )
            residual = target - self.tcp()
            error = float(np.linalg.norm(residual))
            if error < tolerance_m:
                break
            command = command + residual
        return error

    def _steps_left(self) -> int:
        """Simulator steps still allowed before LIBERO's hard horizon."""
        used = int(getattr(self.lle, "_sim_step_count", 0))
        cap = int(getattr(self.lle, "max_steps", 0))
        return max(cap - used, 0) if cap else 10**9

    #: the LIBERO gripper command that leaves the jaws where they are.  robosuite's
    #: ``PandaGripper.format_action`` integrates ``sign(action) * 0.01`` into the actuator
    #: target, and cap-x maps ``fraction`` to ``1 - 2 * fraction`` — so fraction 0.5 sends
    #: sign(0.0) = 0 and the jaws hold, which is what makes a PARTIAL opening expressible.
    GRIPPER_HOLD_FRACTION = 0.5
    #: how close the measured gap has to get to a requested partial opening (m)
    PREOPEN_TOLERANCE_M = 0.0015

    def open_fingers(self, gap_m: float | None = None, *, max_steps: int = 120) -> bool:
        """Open the gripper, unless the 40 steps it costs would cross the horizon.

        With ``gap_m`` the jaws are driven — open or closed, whichever way they have to
        go — only until the MEASURED finger gap (:meth:`get_gripper_opening`) reaches it,
        and then held there for the rest of the program (the hold survives ``goto`` and
        every other motion; ``open_fingers()`` / ``close_fingers()`` release it).  The
        fully open jaw spans ``gap + 2 * 0.0265`` m across its closing axis, which is what
        decides whether a tool fits in a slot: on libero.90_6 the bottom drawer's handle
        bar sits 3.3 cm below the middle drawer's face, so the full 7.7 cm gap (13 cm of
        envelope) cannot straddle it and a 2 cm one can.  Returns False when the horizon
        refused the call.
        """
        if gap_m is None:
            if self._steps_left() <= 45:
                return False
            self.api.open_gripper()
            return True
        target = float(np.clip(float(gap_m), 0.0, GRIPPER_MAX_OPENING))
        steps = min(int(max_steps), max(self._steps_left() - 1, 0))
        if steps <= 0:
            return False
        lle = self.lle
        command = self._gripper_command(target)
        lle._set_gripper(self.GRIPPER_HOLD_FRACTION)
        spent = 0
        for _ in range(4):
            if command is not None:
                command()
            elif spent == 0:  # no handle on the gripper model: drive and freeze instead
                lle._set_gripper(1.0 if self.get_gripper_opening() < target else 0.0)
            while spent < steps:
                gap = self.get_gripper_opening()
                lle._step_once()
                spent += 1
                if abs(self.get_gripper_opening() - gap) < 1e-5:
                    break
                if command is None and abs(self.get_gripper_opening() - target) <= (
                    self.PREOPEN_TOLERANCE_M
                ):
                    break
            lle._set_gripper(self.GRIPPER_HOLD_FRACTION)
            if command is None or abs(self.get_gripper_opening() - target) <= (
                self.PREOPEN_TOLERANCE_M
            ) or spent >= steps:
                break
        return True

    def _gripper_command(self, gap_m: float) -> Any:
        """A callable that commands the finger gap ``gap_m``, or None on an unknown model.

        The LIBERO gripper action is a *sign*: robosuite's ``PandaGripper.format_action``
        integrates ``sign(action) * 0.01`` into ``current_action`` and the actuator target
        is ``bias + weight * current_action``, so the only openings a fraction can express
        are "keep opening", "keep closing" and (fraction 0.5 -> sign 0) "hold".  A partial
        opening is therefore the *integrator* set straight to the value the wanted gap
        needs, and then held: driving-and-freezing instead leaves the actuator target ~2 cm
        ahead of the measured gap and the jaws creep on for the rest of the motion
        (measured 2026-08-28: a freeze at 0.031 m settled at 0.011 m).
        """
        try:
            robot = self.sim  # only used to fail fast when the sim has gone away
            if robot is None:
                return None
            env = getattr(self.lle, "handle", None)
            env = getattr(env, "env", env)
            for _ in range(6):
                if getattr(env, "robots", None):
                    break
                env = getattr(env, "env", None)
            gripper = env.robots[0].gripper
            model = self.sim.model
            ids = [model.actuator_name2id(a) for a in gripper.actuators]
            joints = [model.joint_id2name(int(model.actuator_trnid[i][0])) for i in ids]
            ranges = np.asarray(model.actuator_ctrlrange, dtype=float)[ids]
        except Exception:
            return None
        if len(ids) != 2 or not all(j for j in joints):
            return None
        bias = 0.5 * (ranges[:, 1] + ranges[:, 0])
        weight = 0.5 * (ranges[:, 1] - ranges[:, 0])
        if float(np.min(np.abs(weight))) < 1e-9:
            return None
        # get_gripper_opening is finger_joint1 - finger_joint2, so half the gap each way
        wanted = np.array(
            [0.5 * gap_m if j.endswith("1") else -0.5 * gap_m for j in joints], dtype=float
        )
        value = np.clip((wanted - bias) / weight, -1.0, 1.0)

        def command() -> None:
            gripper.current_action = value.copy()

        return command

    def close_fingers(self) -> bool:
        """Close the gripper, unless the 60 steps it costs would cross the horizon."""
        if self._steps_left() <= 65:
            return False
        self.api.close_gripper()
        return True

    def settle(self, steps: int = 10) -> None:
        """Advance the simulation without moving the arm target (never past the horizon)."""
        for _ in range(min(int(steps), max(self._steps_left() - 1, 0))):
            self.lle._step_once()

    # ------------------------------------------------------------ tools

    def get_step_budget(self) -> dict[str, int]:
        """Simulator steps used and left in this episode.

        LIBERO's horizon is hard: the 4001st ``step`` raises "executing action in
        terminated episode" and the program dies mid-motion, so a plan that needs a
        second grasp attempt should check here first.  Measured costs (2026-08-27):
        ``goto_pose`` ~50-250, ``move_by_displacement`` 30-240 (240 when fighting a
        contact), ``open_gripper`` 40, ``close_gripper`` 60, ``grasp_object`` 500-800 per
        successful attempt, ``place_on``/``place_in_region`` 600-900 (up to 1800-2350 with a
        long ``place_gently`` descent), ``open_articulated`` 900-1500 (up to 2800),
        ``close_articulated`` ~250 per stroke (+ ~300 per seat), ``turn_knob`` ~1100-2000,
        ``park`` ~50 (retreat only) to ~225, ``push_object`` 650-720.  The primitives here
        clamp themselves to the steps left, and so do the raw ``goto_pose`` /
        ``open_gripper`` / ``close_gripper`` tools (a refused or truncated call reports
        ``budget_exhausted``).
        """
        used = int(getattr(self.lle, "_sim_step_count", 0))
        cap = int(getattr(self.lle, "max_steps", 0))
        return {"used": used, "max": cap, "remaining": max(cap - used, 0)}

    def get_gripper_opening(self) -> float:
        """Finger-pad gap in metres (≈0.077 fully open, ≈0.0 closed on nothing)."""
        sim = self.sim
        q1 = float(np.atleast_1d(sim.data.get_joint_qpos("gripper0_finger_joint1"))[0])
        q2 = float(np.atleast_1d(sim.data.get_joint_qpos("gripper0_finger_joint2"))[0])
        return max(q1 - q2, 0.0)

    def move_by_displacement(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, **goto_kwargs: Any
    ) -> list[float]:
        """Move the fingertips by a base-frame offset, keeping the hand orientation.

        ``goto_kwargs`` reach :meth:`goto`.  Pass ``max_iterations=2`` when the motion is
        meant to be *resisted* — pulling a drawer, lowering a held object onto a support:
        the default keeps re-solving until it arrives, which shoves whatever is in the way.
        """
        _, quat = self.hand_pose()
        target = self.tcp() + np.array([x, y, z], dtype=float)
        self.goto(target, quat, **goto_kwargs)
        return [float(v) for v in self.tcp()]

    def check_reachable(self, position: Any, quaternion_wxyz: Any = None) -> dict[str, Any]:
        """IK probe (nothing moves): solve for the fingertip pose and report the FK error.

        What it answers: "does the SEEDLESS solver (pyroki ``solve_ik`` from the zero
        configuration, position weight 50, orientation weight 10, joint limits only)
        find a configuration whose fingertips land within 1 cm?".  What it does NOT
        answer: whether the arm will get there from its CURRENT posture (``goto`` uses
        the velocity-cost solver seeded from the current joints, which can end 2-10 cm
        and 10-27 deg away on the same target), whether the table or the hand body is in
        the way (no collision check at all — a point 5 cm below the table top reports
        reachable), or whether the orientation is honoured (``orientation_dev_deg`` is
        reported, not gated).  Measured 2026-08-27: a level wrist at base z 0.03 fails
        here (4.6 cm) while it solves exactly when seeded from z 0.00 or 0.05.
        """
        api = self.api
        pos = np.asarray(position, dtype=float).reshape(3)
        quat = (
            tool_down_quat_wxyz() if quaternion_wxyz is None else np.asarray(quaternion_wxyz, float)
        )
        rot = _quat_wxyz_to_mat(quat)
        hand_target = pos + rot @ (-TCP_OFFSET_HAND)
        try:
            cfg = api._pks.solve_ik(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_wxyz=quat,
                target_position=hand_target,
            )
            fk = np.asarray(api._robot.forward_kinematics(np.asarray(cfg)))
            idx = api._robot.links.names.index(api._target_link_name)
            reached = np.asarray(fk[idx])[4:7]
            err = float(np.linalg.norm(reached - hand_target))
            got = _quat_wxyz_to_mat(np.asarray(fk[idx])[0:4])
            dev = got.T @ rot
            ori = float(np.degrees(np.arccos(np.clip((np.trace(dev) - 1.0) / 2.0, -1.0, 1.0))))
        except Exception as exc:
            return {"reachable": False, "error": f"ik: {exc}"[:200]}
        return {
            "reachable": err < 0.01,
            "error_m": err,
            "orientation_dev_deg": round(ori, 2),
            "quat_wxyz": [float(v) for v in quat],
        }

    #: ``probe_goto`` calls a pose reachable below this seeded-solve shortfall (m) ...
    PROBE_RESIDUAL_M = 0.03
    #: ... and below this largest single-joint move (rad).  Both measured 2026-08-28 over
    #: 56 poses on libero.90_6 / spatial_0 / 90_8 / goal_1 against ``goto``'s own verdict.
    PROBE_TRAVEL_RAD = 0.80

    def joint_positions(self) -> np.ndarray:
        """The arm's seven joint angles as the SIMULATOR has them (rad).

        ``lle._current_joints`` is the last commanded configuration, not the achieved one
        — they differ by up to 0.32 rad when the controller saturates, which is exactly
        the case :meth:`probe_goto` exists to catch, so read the sim.
        """
        sim = self.sim
        addrs = [sim.model.get_joint_qpos_addr(f"robot0_joint{i + 1}") for i in range(7)]
        return np.array([float(sim.data.qpos[a]) for a in addrs], dtype=float)

    def probe_goto(
        self,
        position: Any,
        quaternion_wxyz: Any = None,
        *,
        iterations: int = 8,
    ) -> dict[str, Any]:
        """Will ``goto_pose_closed`` actually ARRIVE at this pose from where the arm is
        standing?  Nothing moves and no budget is spent.

        ``check_reachable`` answers a different question — *"does a configuration
        exist?"* — by running the seedless ``solve_ik`` from the zero configuration.  Over
        56 measured poses on four LIBERO episodes its ``error_m`` has **rank correlation
        -0.05 with the residual ``goto`` actually leaves**: it is not a weak predictor,
        it is no predictor, because the question it answers is not the one a program is
        asking.  It said ``reachable`` for 19 poses that ``goto_pose_closed`` then missed
        by 1.4-23.7 cm.

        This probe runs the solve ``goto`` runs — ``solve_ik_vel_cost`` **seeded from the
        configuration the arm is in** — and reports the two numbers that do predict it
        (rank correlation +0.46 and +0.64):

        * ``error_m`` — how far short one velocity-cost solve from THIS posture lands.
          The velocity cost is what makes it posture-dependent: a target the solver
          reaches in one solve is one ``goto`` walks to; a big residual means ``goto`` has
          to iterate, over-command, and hope the controller keeps up.
        * ``joint_travel_rad`` — the largest single-joint move that solution asks for.
          ``goto`` gives the LIBERO joint controller 80 steps per iteration and it
          saturates; past ~0.8 rad the arm arrives somewhere else.

        ``reachable`` is True when ``error_m < 0.03`` **and** ``joint_travel_rad < 0.80``
        — the measured gate, 86 % agreement with ``goto``'s own ``reached`` over those 56
        poses against 66 % for ``check_reachable``, and 14 of the 19 poses
        ``check_reachable`` wrongly accepted are rejected here.  Note the 3 cm: it is a
        gate on the SOLVER's shortfall, not a predicted landing error, and it is looser
        than ``check_reachable``'s 1 cm on purpose.

        Which to use:

        * **``probe_goto`` before any motion you are about to spend steps on** — a grasp
          approach, a descent into a slot, a placement.  False here means pick another
          pose now instead of burning 100-600 steps finding out.
        * **``check_reachable`` (and :meth:`reachable_tool_quat`, which runs it) to ask
          about the WORKSPACE** — "is this shelf inside the arm's envelope at all", or
          when the arm will be parked or reset before the move, which is the one case
          where the current posture genuinely does not matter.

        What neither can see is **collision**: both are pure kinematics on the arm alone.
        The residuals it accepts that ``goto`` still misses are all contact — the jaw
        driven into the table top (check the finger geometry with :meth:`plan_pinch` or
        :meth:`clearance_at`), or the elbow swinging into a fixture.  ``free_error_m`` is
        the residual left when the same solve is re-seeded from its own last solution up
        to ``iterations`` times, i.e. what a freely travelling arm would leave: near zero
        for almost every pose, so a nonzero one is a hard workspace limit rather than a
        posture problem.  ``api.cfg`` is never written back.

        Returns ``{"reachable", "error_m", "joint_travel_rad", "orientation_dev_deg",
        "free_error_m", "free_iterations", "error_trace_m", "quat_wxyz", "steps_left"}``.
        """
        api = self.api
        target = np.asarray(position, dtype=float).reshape(3)
        quat = (
            tool_down_quat_wxyz() if quaternion_wxyz is None else np.asarray(quaternion_wxyz, float)
        ).reshape(4)
        rot = _quat_wxyz_to_mat(quat)
        hand_target = target + rot @ (-TCP_OFFSET_HAND)
        try:
            idx = api._robot.links.names.index(api._target_link_name)
            here = self.joint_positions()
            seed = api.cfg
            seed = np.concatenate([here, [0.0]]) if seed is None else np.asarray(seed, float)
            cfg = seed
            trace: list[float] = []
            fk_link = None
            fk_first = None
            first_cfg = seed
            for _ in range(max(1, int(iterations))):
                cfg = np.asarray(
                    api._pks.solve_ik_vel_cost(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=hand_target,
                        target_wxyz=quat,
                        prev_cfg=cfg,
                    ),
                    dtype=float,
                )
                fk_link = np.asarray(api._robot.forward_kinematics(cfg))[idx]
                if fk_first is None:
                    first_cfg, fk_first = cfg, fk_link
                reached = np.asarray(fk_link)[4:7] + _quat_wxyz_to_mat(
                    np.asarray(fk_link)[0:4]
                ) @ TCP_OFFSET_HAND
                trace.append(round(float(np.linalg.norm(target - reached)), 4))
                if trace[-1] < 0.005:
                    break
                if len(trace) >= 2 and trace[-1] > 0.8 * trace[-2]:
                    break  # the solve has stopped closing in: more of them buy nothing
        except Exception as exc:
            return {"reachable": False, "error": f"ik: {exc}"[:200]}
        if fk_link is None:
            return {"reachable": False, "error": "ik: no solve was attempted"}
        got = _quat_wxyz_to_mat(np.asarray(fk_first)[0:4])
        dev = got.T @ rot
        ori = float(np.degrees(np.arccos(np.clip((np.trace(dev) - 1.0) / 2.0, -1.0, 1.0))))
        travel = float(np.max(np.abs(np.asarray(first_cfg)[:7] - here)))
        error = trace[0]
        return {
            "reachable": bool(error < self.PROBE_RESIDUAL_M and travel < self.PROBE_TRAVEL_RAD),
            "error_m": round(error, 5),
            "joint_travel_rad": round(travel, 4),
            "orientation_dev_deg": round(ori, 2),
            "free_error_m": round(trace[-1], 5),
            "free_iterations": len(trace),
            "error_trace_m": trace,
            "quat_wxyz": [float(v) for v in quat],
            "steps_left": self._steps_left(),
        }

    def reachable_tool_quat(
        self,
        points: Any,
        point_dir: Any,
        close_dir: Any = (0.0, 0.0, 1.0),
        pitches: Any = PUSH_PITCHES,
    ) -> dict[str, Any]:
        """Pick the flattest DOWNWARD tool pitch whose IK probe reaches every base-frame
        point in ``points`` (an (N, 3) array) while the tool axis points along
        ``point_dir``.

        Nothing moves: this is :meth:`check_reachable` run over
        :func:`tool_pitched_quat_wxyz`, with that probe's limits — a seedless solve with
        no collision check, so "reachable" means an IK solution exists, not that ``goto``
        from the current posture will get there (read ``goto_pose``'s residual for that).
        Use it before a horizontal push/pull (drawer fronts, handles, a microwave cavity):
        the probe rejects a level wrist a few centimetres above the table and a +x push
        low over the table really does need ~60 deg, so pitch, not a different contact
        point, is usually the fix.  Returns ``{"quat_wxyz", "pitch_deg", "reachable",
        "tried"}``; when no pitch works the pitch with the smallest worst-case error is
        returned with ``reachable=False`` so the caller can still try (callers should
        read the flag).
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        pitches = tuple(float(p) for p in pitches)
        if pts.shape[0] == 0:
            raise ValueError("points must hold at least one base-frame [x, y, z]")
        if not pitches:
            raise ValueError("pitches must not be empty")
        tried: list[dict[str, Any]] = []
        for pitch in pitches:
            quat = tool_pitched_quat_wxyz(point_dir, float(pitch), close_dir)
            errs = [self.check_reachable(p, quat) for p in pts]
            ok = all(bool(e.get("reachable")) for e in errs)
            entry: dict[str, Any] = {
                "pitch_deg": float(pitch),
                "reachable": ok,
                "worst_error_m": round(max(float(e.get("error_m", 1.0)) for e in errs), 4),
            }
            messages = [str(e["error"]) for e in errs if "error" in e]
            if messages:
                entry["error"] = messages[0]
            tried.append(entry)
            if ok:
                return {
                    "quat_wxyz": [float(v) for v in quat],
                    "pitch_deg": float(pitch),
                    "reachable": True,
                    "tried": tried,
                }
        fallback = min(tried, key=lambda t: t["worst_error_m"])["pitch_deg"]
        quat = tool_pitched_quat_wxyz(point_dir, float(fallback), close_dir)
        return {
            "quat_wxyz": [float(v) for v in quat],
            "pitch_deg": float(fallback),
            "reachable": False,
            "tried": tried,
        }

    #: a retreat-only park costs ~50 steps, retreat + rise ~225 (measured 2026-08-27 on
    #: libero_goal/0): below this nothing is attempted and the report says so
    PARK_MIN_STEPS = 230

    def park(self, height_m: float = 0.30, retreat_m: float = 0.08) -> dict[str, Any]:
        """Back the hand out of whatever it was working on and lift it clear.

        Retreats ``retreat_m`` along the CURRENT tool axis (straight out of a drawer, a
        cavity or a handle — moving sideways first is what hooks the closed fingertips
        behind a handle bar and drags the part back open), then rises so the fingertips
        end at least ``height_m`` above the robot base with the wrist levelled to
        tool-down and the finger direction KEPT (a fitted yaw on a held object survives).
        Idempotent: a tool-down hand already at ``height_m`` does not retreat (which would
        be a further lift every call).  Call it after an articulation move — the next
        ``goto`` interpolates in joint space from wherever the hand is, and a hand left
        inside an open drawer sweeps the scene on its way to the next object; parking out
        of a cavity needs ``height_m`` above the fixture's ``top_z`` and a ``retreat_m``
        longer than its overhang.  Returns ``{"tcp", "retreated_m" (along the tool
        axis), "raised_m" (from the start height), "reached", "budget_exhausted"}``.
        """
        start = self.tcp()
        if self._steps_left() < self.PARK_MIN_STEPS:
            return {
                "tcp": [float(v) for v in start],
                "retreated_m": 0.0,
                "raised_m": 0.0,
                "reached": bool(start[2] >= float(height_m) - 0.01),
                "budget_exhausted": True,
            }
        _, quat = self.hand_pose()
        rot = _quat_wxyz_to_mat(quat)
        tool = rot @ np.array([0.0, 0.0, 1.0])
        if tool[2] < -0.9 and start[2] >= float(height_m) - 0.005:
            here = start  # nothing to back out of: already parked
        else:
            self.goto(start - tool * float(retreat_m), quat, iterations=2, max_iterations=3)
            here = self.tcp()
        retreated = float((start - here) @ tool)
        if here[2] < float(height_m) - 0.005:
            close = rot[:, 1].copy()
            close[2] = 0.0
            down = tool_down_quat_wxyz(
                close if np.linalg.norm(close) > 1e-3 else np.array([0.0, 1.0, 0.0])
            )
            self.goto(
                np.array([here[0], here[1], float(height_m)]), down, iterations=2, max_iterations=4
            )
        end = self.tcp()
        return {
            "tcp": [float(v) for v in end],
            "retreated_m": round(retreated, 4),
            "raised_m": round(float(end[2] - start[2]), 4),
            "reached": bool(end[2] >= float(height_m) - 0.01),
            "budget_exhausted": self._steps_left() < self.PARK_MIN_STEPS,
        }

    def checkpoint(self) -> str:
        """Snapshot the whole MuJoCo state (practice/rehearsal only)."""
        cid = f"ck-{len(self._checkpoints) + 1}-{uuid.uuid4().hex[:6]}"
        self._checkpoints[cid] = {
            "state": self.sim.get_state(),
            "cfg": None if self.api.cfg is None else np.array(self.api.cfg, copy=True),
            "joints": np.array(self.lle._current_joints, copy=True),
            "gripper": float(self.lle._gripper_fraction),
        }
        return cid

    def restore(self, checkpoint_id: str) -> None:
        """Rewind to a checkpoint; the step budget is not rewound."""
        snap = self._checkpoints[checkpoint_id]
        self.sim.set_state(snap["state"])
        self.sim.forward()
        self.api.cfg = None if snap["cfg"] is None else np.array(snap["cfg"], copy=True)
        self.lle._current_joints = np.array(snap["joints"], copy=True)
        self.lle._gripper_fraction = snap["gripper"]
        self.settle(1)  # refreshes the cached observation the pose queries read

    def place_gently(
        self,
        object_name: str,
        max_descent: float = 0.25,
        step: float = 0.015,
        watch_name: str | None = None,
        watch_tolerance_m: float = 0.004,
        target_xy: Any = None,
        correction_per_step_m: float = 0.008,
    ) -> dict[str, Any]:
        """Lower the HELD object in small steps until it rests (it stops following the
        hand, or the hand can no longer descend), then open the gripper and retreat 6 cm.

        ``watch_name`` names a body that must NOT be shoved — the support you are stacking
        onto.  A light plate or bowl slides away when the load keeps pressing after first
        contact, and the object then ends up beside it instead of on it; once that body has
        moved ``watch_tolerance_m`` the descent stops and the object is released at once.
        ``target_xy`` keeps the descent honest: a hanging load drifts a couple of
        centimetres on the way down, which is enough to miss a plate whose ``(On)`` test
        allows 3 cm, so each step also slides the hand by up to
        ``correction_per_step_m`` toward putting the OBJECT on that spot (at most that much
        per step, and only while the hand is still descending freely — centre with
        ``carry_to``/``descend_over`` first).  ``target_xy`` is in the robot-base frame.
        The object is released whether or not it rested (``rested`` False = dropped from
        wherever ``max_descent`` ran out); near the step horizon the release can be
        refused, which ``released`` reports — nothing else in the report can tell you.
        Returns ``{"rested", "held", "released", "descent_m", "object_z", "pushed_support",
        "xy_error_m" (None without target_xy), "budget_exhausted"}``.
        """
        goal_xy = None if target_xy is None else np.asarray(target_xy, dtype=float)[:2]
        watch0 = None if watch_name is None else self.body_position(watch_name)
        descended = 0.0
        rested = False
        pushed = False
        blocked = 0
        budget_out = False
        # nothing in the jaws: say so instead of "resting" on the first step
        if self.get_gripper_opening() > GRIPPER_MAX_OPENING - 0.005:
            obj = self.body_position(object_name)
            return {
                "rested": False,
                "held": False,
                "released": True,
                "descent_m": 0.0,
                "object_z": float(obj[2]),
                "pushed_support": False,
                "xy_error_m": None
                if goal_xy is None
                else round(float(np.linalg.norm(obj[:2] - goal_xy)), 4),
                "budget_exhausted": self._steps_left() <= 1,
            }
        while descended < max_descent:
            if self._steps_left() <= 1:
                budget_out = True
                break
            tcp0 = self.tcp()
            obj0 = self.body_position(object_name)
            slide = np.zeros(2)
            if goal_xy is not None and blocked == 0:
                residual = goal_xy - obj0[:2]
                norm = float(np.linalg.norm(residual))
                if norm > 1e-4:
                    slide = residual * min(1.0, correction_per_step_m / norm)
            self.move_by_displacement(
                float(slide[0]), float(slide[1]), -step, iterations=3, max_iterations=3
            )
            tcp1 = self.tcp()
            obj1 = self.body_position(object_name)
            hand_dz = float(tcp0[2] - tcp1[2])
            obj_dz = float(obj0[2] - obj1[2])
            descended += max(hand_dz, 0.0)
            if watch0 is not None:
                moved = float(np.linalg.norm(self.body_position(watch_name) - watch0))
                if moved > watch_tolerance_m:
                    rested, pushed = True, True
                    break
            if hand_dz < 0.2 * step:  # the hand itself cannot descend: resting on support
                blocked += 1
                if blocked >= 3:
                    rested = True
                    break
                continue
            blocked = 0
            if obj_dz < 0.4 * hand_dz:  # the object stopped following the hand
                rested = True
                break
        released = bool(self.open_fingers())
        self.settle(10)
        self.move_by_displacement(z=0.06)
        obj = self.body_position(object_name)
        return {
            "rested": rested,
            "held": True,
            "released": released,
            "descent_m": round(descended, 3),
            "object_z": float(obj[2]),
            "pushed_support": pushed,
            "xy_error_m": None
            if goal_xy is None
            else round(float(np.linalg.norm(obj[:2] - goal_xy)), 4),
            "budget_exhausted": budget_out or self._steps_left() <= 1,
        }

    # ------------------------------------------------------- scene geometry

    def resolve_body(self, name: str) -> str:
        """MuJoCo body name for an object/fixture description.

        ``get_object_pose`` only ever answers for the ``_1`` instance of a type, so a
        scene with two bowls or two plates is half invisible to it.  This accepts a
        body name (``akita_black_bowl_2_main``), an instance name
        (``akita_black_bowl_2``), a type name (``plate``) or any unique substring, and
        returns the body other tools here take.  Use ``get_body_positions()`` to list
        the candidates.
        """
        sim = self.sim
        model = sim.model
        raw_m, _ = self._raw()
        names = [model.body_id2name(i) for i in range(raw_m.nbody)]
        names = [n for n in names if n and "robot" not in n and "gripper" not in n]
        query = str(name).strip().lower().replace(" ", "_")
        if not query:
            raise KeyError("empty body name; try get_body_positions()")
        for candidate in (query, f"{query}_main", f"{query}_1_main"):
            if candidate in names:
                return candidate
        hits = [n for n in names if query in n.lower()]
        mains = [n for n in hits if n.endswith("_main")]
        hits = mains or hits
        if not hits:
            raise KeyError(f"no body matches {name!r}; try get_body_positions()")
        # a substring that matches several distinct objects is not "unique": say so
        # instead of silently taking the shortest name (which is scene-dependent)
        roots = {h.rsplit("_main", 1)[0] for h in hits}
        if len(roots) > 1:
            raise KeyError(
                f"{name!r} is ambiguous: {sorted(hits)}; give the instance "
                f"(e.g. {sorted(hits)[0]!r}) or see get_body_positions()"
            )
        return sorted(hits, key=len)[0]

    def body_position(self, name: str) -> np.ndarray:
        """Body-frame origin of any scene body, in the robot-base frame.

        This is the point LIBERO's own predicates compare (``On``/``In`` measure body
        positions), and unlike ``get_object_pose`` it also answers for the second
        instance of a type (``plate_2``, ``moka_pot_2``) and for fixture sub-bodies.  A
        bare type name resolves to the ``_1`` instance.  It is NOT the part's geometric
        centre (a drawer sub-body's origin can sit at the cabinet's centre): aim motions
        with ``get_object_extents(name)["center"]``.
        """
        body = self.resolve_body(name)
        return self.base_from_world(self.sim.data.get_body_xpos(body))

    def _geom_corners(self, geom: int) -> np.ndarray:
        """World-frame corner cloud of one geom (mesh vertices, else its box hull)."""
        raw_m, raw_d = self._raw()
        pos = np.asarray(raw_d.geom_xpos[geom], dtype=float)
        rot = np.asarray(raw_d.geom_xmat[geom], dtype=float).reshape(3, 3)
        size = np.asarray(raw_m.geom_size[geom], dtype=float)
        kind = int(raw_m.geom_type[geom])
        if kind == MJ_MESH:
            mesh = int(raw_m.geom_dataid[geom])
            start = int(raw_m.mesh_vertadr[mesh])
            count = int(raw_m.mesh_vertnum[mesh])
            verts = np.asarray(raw_m.mesh_vert[start : start + count], dtype=float).reshape(-1, 3)
            return (rot @ verts.T).T + pos
        if kind == MJ_SPHERE:
            half = np.array([size[0]] * 3)
        elif kind == MJ_CAPSULE:
            half = np.array([size[0], size[0], size[1] + size[0]])
        elif kind == MJ_CYLINDER:
            half = np.array([size[0], size[0], size[1]])
        elif kind == MJ_BOX:
            half = size[:3].copy()
        else:
            half = np.array([float(raw_m.geom_rbound[geom])] * 3)
        corners = np.array(
            [
                [sx, sy, sz]
                for sx in (-half[0], half[0])
                for sy in (-half[1], half[1])
                for sz in (-half[2], half[2])
            ]
        )
        return (rot @ corners.T).T + pos

    def _body_geoms(self, body_id: int, collision_only: bool = True) -> list[int]:
        raw_m, _ = self._raw()
        out = []
        for g in range(raw_m.ngeom):
            if int(raw_m.geom_bodyid[g]) != body_id:
                continue
            solid = raw_m.geom_contype[g] != 0 or raw_m.geom_conaffinity[g] != 0
            if collision_only and not solid:
                continue
            out.append(g)
        return out

    def get_object_extents(self, object_name: str, collision_only: bool = True) -> dict[str, Any]:
        """Axis-aligned bounding box of an object's geoms, in the robot-base frame.

        The pose tools give you a centre with no size, so every grasp height and every
        placement clearance had to be guessed.  This measures it:
        ``{"body", "center", "size", "lo", "hi", "top_z", "bottom_z"}`` (metres).
        ``collision_only`` (default) ignores the visual mesh, so ``size`` is what the
        fingers actually meet; when the body has no collision geoms the visual ones are
        measured instead and ``collision_only`` in the result says which.  ``center`` is
        the box centre, not the body origin the predicates compare (``body_position``);
        ``top_z`` is the highest point (a bowl's or basket's rim, not its floor).  A body
        with no geoms of its own (``flat_stove_1_main``, ``wooden_cabinet_1_main``) is
        measured over its sub-body tree (``bodies`` lists them); pass the sub-body name
        for one part (``flat_stove_1_burner``).
        """
        body, bodies, geoms, effective = self._object_geoms(object_name, collision_only)
        cloud = np.concatenate([self._geom_corners(g) for g in geoms], axis=0)
        # rotate the whole cloud into the base frame (two world corners only work for a
        # base yawed by a multiple of 90 deg)
        wxyz_xyz = np.asarray(self.lle.base_link_wxyz_xyz, dtype=float).reshape(-1)
        rot = _quat_wxyz_to_mat(wxyz_xyz[:4])
        cloud_b = (rot.T @ (cloud - wxyz_xyz[4:7]).T).T
        lo, hi = cloud_b.min(axis=0), cloud_b.max(axis=0)
        return {
            "body": body,
            "bodies": [self.sim.model.body_id2name(b) for b in bodies],
            "collision_only": bool(effective),
            "center": [float(v) for v in (lo + hi) / 2],
            "size": [float(v) for v in (hi - lo)],
            "lo": [float(v) for v in lo],
            "hi": [float(v) for v in hi],
            "top_z": float(hi[2]),
            "bottom_z": float(lo[2]),
        }

    def _object_geoms(
        self, object_name: str, collision_only: bool = True
    ) -> tuple[str, list[int], list[int], bool]:
        """``(body, bodies, geoms, collision_only)`` behind :meth:`get_object_extents` and
        :meth:`object_cloud`: the geoms of the resolved body, else of its sub-body tree
        (``flat_stove_1_main``, ``white_cabinet_1_main``), visual ones when it has no
        collision geoms.  Raises when nothing can be measured.
        """
        body = self.resolve_body(object_name)
        body_id = self.sim.model.body_name2id(body)
        bodies = [body_id]
        geoms = self._body_geoms(body_id, collision_only)
        effective = collision_only
        if not geoms and collision_only:
            geoms = self._body_geoms(body_id, False)
            effective = False
        if not geoms:
            raw_m, _ = self._raw()
            parent = raw_m.body_parentid
            bodies = [
                b
                for b in range(int(raw_m.nbody))
                if b != body_id and self._is_descendant(int(b), body_id, parent)
            ]
            effective = collision_only
            geoms = [g for b in bodies for g in self._body_geoms(b, collision_only)]
            if not geoms and collision_only:
                geoms = [g for b in bodies for g in self._body_geoms(b, False)]
                effective = False
        if not geoms:
            raise RuntimeError(f"{body} has no geoms to measure (nor its sub-bodies)")
        return body, bodies, geoms, effective

    @staticmethod
    def _is_descendant(body: int, root: int, parent: Any) -> bool:
        k = body
        while k > 0:
            k = int(parent[k])
            if k == root:
                return True
        return False

    def in_contact(self, name_a: str, name_b: str) -> bool:
        """Are these two bodies touching right now?

        LIBERO's ``(On a b)`` and ``(In a region)`` both require real contact with the
        support or the fixture, so an object hovering a millimetre above a hob or resting
        on a second object instead of the shelf scores zero however good the position
        looks.  Sub-bodies count: contact with any body of the same articulated object.
        """
        raw_m, raw_d = self._raw()
        want = set()
        for name in (name_a, name_b):
            root = self.resolve_body(name).rsplit("_main", 1)[0]
            ids = {
                i
                for i in range(raw_m.nbody)
                if (self.sim.model.body_id2name(i) or "").startswith(root)
            }
            want.add(frozenset(ids))
        pair = tuple(want)
        first, second = (pair[0], pair[-1])
        for c in range(int(raw_d.ncon)):
            con = raw_d.contact[c]
            b1 = int(raw_m.geom_bodyid[con.geom1])
            b2 = int(raw_m.geom_bodyid[con.geom2])
            if (b1 in first and b2 in second) or (b2 in first and b1 in second):
                return True
        return False

    def get_region_boxes(self, keyword: str = "") -> dict[str, dict[str, Any]]:
        """The named box regions the BDDL goal is scored against, in the robot-base frame.

        LIBERO's ``(In obj region)`` is true when the object's body centre is inside the
        region site's box, and ``(On obj region)`` when it sits just above the box's top
        face and touches the parent fixture.  Those boxes are invisible in the camera and
        in ``get_all_object_poses``; this returns
        ``{name: {"center", "half_size", "top_z", "lo", "hi", "site_size", "site_mat",
        "parent"}}`` so a program can aim at them: ``center``/``lo``/``hi`` are the
        world-axis-aligned box (what ``In`` tests, with the floor relaxed 1 cm), ``top_z``
        its top; ``site_size`` (half extents in the site's own frame) and ``site_mat``
        (the site's WORLD rotation, row-major) are what ``On`` tests; ``parent`` is the
        fixture the region belongs to — ``On`` needs contact with it, and it is ``None``
        for table/floor patches, which need no contact.  ``keyword`` filters
        case-insensitively (``"contain"``, ``"top_region"``, ``"top_side"``; drawers are
        ``<cabinet>_top/middle/bottom_region``); an empty dict means nothing matched —
        call it with no keyword to list every region.  Doubles as the disambiguator for
        "the bowl at the back": the scene's own ``*_back_init_region`` box.
        """
        sim = self.sim
        raw_m, raw_d = self._raw()
        sites = self._object_sites()
        out: dict[str, dict[str, Any]] = {}
        for i in range(raw_m.nsite):
            name = sim.model.site_id2name(i)
            if not name or "robot" in name or "gripper" in name:
                continue
            if int(raw_m.site_type[i]) != MJ_SITE_BOX:
                continue
            if keyword and keyword.lower() not in name.lower():
                continue
            centre = self.base_from_world(raw_d.site_xpos[i])
            half = np.abs(
                np.asarray(raw_d.site_xmat[i], dtype=float).reshape(3, 3)
                @ np.asarray(raw_m.site_size[i], dtype=float)
            )
            out[name] = {
                "center": [float(v) for v in centre],
                "half_size": [float(v) for v in half],
                "top_z": float(centre[2] + half[2]),
                "lo": [float(v) for v in centre - half],
                "hi": [float(v) for v in centre + half],
                "site_size": [float(v) for v in np.asarray(raw_m.site_size[i], dtype=float)],
                "site_mat": [float(v) for v in np.asarray(raw_d.site_xmat[i], dtype=float)],
                "parent": getattr(sites.get(name), "parent_name", None),
            }
        return out

    def _object_sites(self) -> dict[str, Any]:
        """LIBERO's own ``object_sites_dict`` (region name → SiteObject), or ``{}``."""
        env = getattr(self.lle, "handle", None)
        for _ in range(6):
            if env is None:
                break
            sites = getattr(env, "object_sites_dict", None)
            if isinstance(sites, dict):
                return sites
            env = getattr(env, "env", None)
        return {}

    # ------------------------------------------------------------ grasping

    def _geom_distance(self, geom: int, point_world: np.ndarray) -> float:
        """Distance from a world point to one geom (exact for boxes/spheres/cylinders)."""
        raw_m, raw_d = self._raw()
        pos = np.asarray(raw_d.geom_xpos[geom], dtype=float)
        rot = np.asarray(raw_d.geom_xmat[geom], dtype=float).reshape(3, 3)
        size = np.asarray(raw_m.geom_size[geom], dtype=float)
        kind = int(raw_m.geom_type[geom])
        local = rot.T @ (np.asarray(point_world, dtype=float) - pos)
        if kind == MJ_BOX:
            outside = np.maximum(np.abs(local) - size[:3], 0.0)
            return float(np.linalg.norm(outside))
        if kind == MJ_SPHERE:
            return float(max(np.linalg.norm(local) - size[0], 0.0))
        if kind in (MJ_CYLINDER, MJ_CAPSULE):
            radial = max(float(np.linalg.norm(local[:2])) - size[0], 0.0)
            axial = max(abs(float(local[2])) - size[1], 0.0)
            return float(np.hypot(radial, axial))
        return float(max(np.linalg.norm(local) - float(raw_m.geom_rbound[geom]), 0.0))

    def clearance_at(self, point: Any, ignore_bodies: Any = (), limit: float = 0.15) -> float:
        """Free space around a base-frame point: distance to the nearest collision geom
        that does not belong to ``ignore_bodies`` (names or body ids).

        The gripper needs room for the outer finger; this is how a primitive tells a rim
        it can reach from a rim jammed against a drawer wall.  Capped at ``limit``.
        """
        return float(self.clearance_many([point], ignore_bodies, limit)[0])

    def clearance_many(
        self, points: Any, ignore_bodies: Any = (), limit: float = 0.15
    ) -> np.ndarray:
        """:meth:`clearance_at` for many base-frame points at once (one pass over the
        scene's geoms instead of one pass per point): ``points`` is (N, 3) in the base
        frame, the result (N,) metres capped at ``limit``.

        Distances are to box/sphere/capsule/cylinder geoms exactly and to MESH geoms by
        their bounding sphere (conservative: a low clearance near a mesh fixture can be a
        phantom); the robot's own bodies, planes and the table/floor slab are skipped.
        Used by :meth:`plan_pinch` to sample free space at several heights along each
        finger and at the finger's outer face, not only at the fingertip: the pads are
        ~4 cm long and 2.65 cm thick outboard, so a tip that clears a drawer wall by a
        millimetre still has its finger body inside it.
        """
        raw_m, _ = self._raw()
        if np.size(points) == 0:
            return np.zeros(0, dtype=float)
        pts = np.asarray(points, dtype=float)
        if pts.size % 3 != 0:
            raise ValueError("points must be (N, 3) base-frame coordinates")
        pts = pts.reshape(-1, 3)
        world = np.array([self.world_from_base(p) for p in pts])
        skip = set()
        for item in (
            ignore_bodies if isinstance(ignore_bodies, (list, tuple, set)) else [ignore_bodies]
        ):
            skip.add(
                int(item)
                if isinstance(item, (int, np.integer))
                else int(self.sim.model.body_name2id(self.resolve_body(str(item))))
            )
        best = np.full(world.shape[0], float(limit), dtype=float)
        for g in range(raw_m.ngeom):
            if raw_m.geom_contype[g] == 0 and raw_m.geom_conaffinity[g] == 0:
                continue
            if int(raw_m.geom_type[g]) == MJ_PLANE:
                continue  # the floor: rbound 0 made it a phantom point obstacle at its origin
            if int(raw_m.geom_bodyid[g]) in skip:
                continue
            name = self.sim.model.body_id2name(int(raw_m.geom_bodyid[g])) or ""
            if "robot" in name or "gripper" in name:
                continue
            if float(raw_m.geom_rbound[g]) > 0.5:  # the table/floor slab
                continue
            best = np.minimum(best, self._geom_distances(g, world))
        return np.maximum(best, 0.0)

    def _geom_distances(self, geom: int, points_world: np.ndarray) -> np.ndarray:
        """:meth:`_geom_distance` vectorised over an (N, 3) array of world points."""
        raw_m, raw_d = self._raw()
        pos = np.asarray(raw_d.geom_xpos[geom], dtype=float)
        rot = np.asarray(raw_d.geom_xmat[geom], dtype=float).reshape(3, 3)
        size = np.asarray(raw_m.geom_size[geom], dtype=float)
        kind = int(raw_m.geom_type[geom])
        local = (np.asarray(points_world, dtype=float) - pos) @ rot
        if kind == MJ_BOX:
            outside = np.maximum(np.abs(local) - size[:3], 0.0)
            return np.linalg.norm(outside, axis=1)
        if kind == MJ_SPHERE:
            return np.maximum(np.linalg.norm(local, axis=1) - size[0], 0.0)
        if kind in (MJ_CYLINDER, MJ_CAPSULE):
            radial = np.maximum(np.linalg.norm(local[:, :2], axis=1) - size[0], 0.0)
            axial = np.maximum(np.abs(local[:, 2]) - size[1], 0.0)
            return np.hypot(radial, axial)
        return np.maximum(np.linalg.norm(local, axis=1) - float(raw_m.geom_rbound[geom]), 0.0)

    def plan_pinch(
        self,
        object_name: str,
        *,
        depth_m: float = 0.03,
        directions: int = 12,
        finger_gap_m: float = 0.008,
        local_radius_m: float = 0.05,
    ) -> dict[str, Any]:
        """Choose where and from which direction to pinch an object, top-down.

        Every collision geom of the object is a candidate grasp site — its body, its rim,
        its handle — and for each site the *local* material within ``local_radius_m`` is
        measured along ``directions`` horizontal directions.  A direction is usable when
        that local width fits between the fingers, when neither fingertip would land
        inside the object itself (the mistake that closes the jaws on a solid wall), and
        when there is free space around both fingertips for everything else in the scene.
        Among the usable ones it takes the roomiest, then the tightest.

        That is what lets one call take a bowl by its rim, a moka pot by its handle and a
        carton across its body, and take a bowl standing in a drawer from the open side.
        Returns ``{"point", "close_dir", "width_m", "clearance_m", "own_gap_m", "mode"}``.
        """
        ext = self.get_object_extents(object_name)
        centre = np.asarray(ext["center"], dtype=float)
        # the same geom lookup as get_object_extents: an object whose collision geoms hang
        # on a child body (new_salad_dressing_1: 0 geoms on _main, 15 on the sub-body)
        # used to raise "need at least one array to concatenate" here
        _body, _bodies, geoms, _eff = self._object_geoms(object_name, True)
        clouds = {
            g: np.array([self.base_from_world(p) for p in self._geom_corners(g)]) for g in geoms
        }
        cloud = np.concatenate(list(clouds.values()))

        sites: list[tuple[np.ndarray, float, float, str]] = []
        seen = set()
        whole = (centre[:2], float(ext["top_z"]), float(ext["size"][2]), "body")
        for spot in [whole] + [
            (c[:, :2].mean(axis=0), float(c[:, 2].max()), float(c[:, 2].ptp()), "part")
            for c in clouds.values()
        ]:
            key = (round(float(spot[0][0]), 2), round(float(spot[0][1]), 2), round(spot[1], 2))
            if key in seen:
                continue
            seen.add(key)
            sites.append(spot)

        best = None
        for xy, top, height, kind in sites:
            z = float(top - min(depth_m, max(0.4 * height, 0.005)))
            near = cloud[
                (np.abs(cloud[:, 2] - z) < 0.015)
                & (np.linalg.norm(cloud[:, :2] - xy, axis=1) < local_radius_m)
            ]
            if near.shape[0] < 2:
                continue
            local_centre = np.array([xy[0], xy[1], z])
            for deg in np.linspace(0.0, 180.0, int(directions), endpoint=False):
                unit = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])
                width = 2 * float(np.abs((near[:, :2] - xy) @ unit[:2]).max())
                if width > GRIPPER_MAX_OPENING - 0.003:
                    continue
                half = 0.5 * width + finger_gap_m
                fingers = [local_centre + unit * half, local_centre - unit * half]
                own = min(
                    min(self._geom_distance(g, self.world_from_base(f)) for g in geoms)
                    for f in fingers
                )
                if own < 0.002:  # a fingertip inside the object's own material
                    continue
                # more of the finger than its tip has to fit: a column of points up each
                # pad face, and the finger's OUTER face (2.65 cm outboard of the pad) —
                # a pad that clears a drawer wall by a few mm can have its finger body
                # inside that wall.  The palm is not modelled.
                probes = [f + np.array([0.0, 0.0, dz]) for f in fingers for dz in FINGER_PROBES]
                clearance = float(self.clearance_many(probes, ignore_bodies=[ext["body"]]).min())
                if clearance < 0.004:
                    continue
                outer_pts = [
                    f + s * unit * FINGER_THICKNESS_M + np.array([0.0, 0.0, dz])
                    for f, s in zip(fingers, (1.0, -1.0), strict=True)
                    for dz in FINGER_PROBES[:2]
                ]
                outer = float(self.clearance_many(outer_pts, ignore_bodies=[ext["body"]]).min())
                if outer < 0.001:
                    continue  # the finger body itself would sit inside a wall
                quat = tool_down_quat_wxyz(unit)
                reach = self.check_reachable(local_centre, quat)
                # roomy first, then the grasp closest to the robot (the arm is most
                # dexterous there, and a far-side grasp puts the load out of reach when
                # it is carried), then the tightest pinch
                # reachable first; then enough room for the fingers (3 cm is plenty);
                # then the tighter pinch to the nearest centimetre — a jaw-wide grasp on a
                # mug's body slips where its handle holds; then the grasp nearest the
                # robot, because a far-side hold puts the load out of reach on the carry
                score = (
                    1.0 if reach.get("reachable") else 0.0,
                    round(min(clearance, 0.03), 3),
                    -round(width, 2),
                    -round(float(np.linalg.norm(local_centre[:2])), 3),
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        {
                            "mode": kind,
                            "point": [float(v) for v in local_centre],
                            "close_dir": [float(v) for v in unit],
                            "width_m": round(width, 4),
                            "clearance_m": round(clearance, 4),
                            "outer_clearance_m": round(outer, 4),
                            "own_gap_m": round(float(own), 4),
                            "base_z": float(top),
                            "reachable": bool(reach.get("reachable")),
                            "quat_wxyz": [float(v) for v in quat],
                        },
                    )
        if best is None:
            # nothing fitted cleanly (single-hull boxes and cans defeat the bounding-sphere
            # mesh distance and the vertex slab): fall back to the narrowest axis of the
            # whole object and let the verified attempts in grasp_object decide.  The
            # clearance/own-gap were NOT measured here, and the width is the whole AABB.
            axis = 0 if ext["size"][0] <= ext["size"][1] else 1
            unit = np.array([1.0, 0.0, 0.0]) if axis == 0 else np.array([0.0, 1.0, 0.0])
            z = float(ext["top_z"] - min(depth_m, 0.5 * ext["size"][2]))
            quat = tool_down_quat_wxyz(unit)
            probe = self.check_reachable([float(centre[0]), float(centre[1]), z], quat)
            width = float(min(ext["size"][0], ext["size"][1]))
            return {
                "mode": "fallback",
                "point": [float(centre[0]), float(centre[1]), z],
                "close_dir": [float(v) for v in unit],
                "width_m": round(width, 4),
                "fits_gripper": bool(width <= GRIPPER_MAX_OPENING - 0.003),
                "clearance_m": None,
                "own_gap_m": None,
                "base_z": float(ext["top_z"]),
                "reachable": bool(probe.get("reachable", False)),
                "quat_wxyz": [float(v) for v in quat],
            }
        return best[1]

    def grasp_object(
        self,
        object_name: str,
        *,
        mode: str = "auto",
        pinch_axis: Any = None,
        grasp_z: float | None = None,
        depth_m: float = 0.03,
        approach_m: float = 0.10,
        lift_m: float = 0.12,
        attempts: int = 3,
    ) -> dict[str, Any]:
        """Grasp an object top-down and lift it, verified against MuJoCo.

        Where and from which side to pinch comes from :meth:`plan_pinch`, which measures
        the object's own geometry: a bowl is taken by its rim, a moka pot by its handle, a
        carton across its body, and a bowl standing in a drawer from the side the drawer
        is open.

        Each attempt is checked — the finger gap must stay open on material and the object
        must rise with the hand — and a failed attempt retries ``0.5 * depth_m`` lower
        per attempt (never below the object; with ``grasp_z`` the height is fixed and
        retries only repeat).  ``mode`` does NOT steer the planner: the report's ``mode``
        is the plan's own (``body`` / ``part`` / ``fallback``) and the value you passed is
        echoed as ``requested_mode``; ``pinch_axis`` overrides the closing direction only
        (the pinch POINT stays the plan's).  ``plan["reachable"]`` is an IK probe, and in
        ``fallback`` mode ``fits_gripper`` says whether the whole object even fits.
        Returns ``{"grasped", "mode", "requested_mode", "body", "opening_m", "lift_m",
        "attempts", "grasp_point", "plan", "budget_exhausted"}``.
        """
        ext = self.get_object_extents(object_name)
        plan = self.plan_pinch(object_name, depth_m=depth_m)
        chosen = plan["mode"]
        point = np.array([plan["point"][0], plan["point"][1], 0.0])
        close_dir = np.asarray(plan["close_dir"], dtype=float)
        if pinch_axis is not None:
            close_dir = np.asarray(pinch_axis, dtype=float)
        quat = tool_down_quat_wxyz(close_dir)

        report: dict[str, Any] = {
            "grasped": False,
            "mode": chosen,
            "requested_mode": mode,
            "body": ext["body"],
            "plan": plan,
        }
        floor = float(ext["bottom_z"]) + 0.004
        for attempt in range(int(attempts)):
            # a failed pinch retries a little lower, but never below the object
            z = (
                float(grasp_z)
                if grasp_z is not None
                else max(float(plan["point"][2]) - 0.5 * depth_m * attempt, floor)
            )
            target = np.array([point[0], point[1], z])
            before = self.body_position(object_name)
            self.open_fingers()
            # rise, translate, descend: approaching diagonally from wherever the arm
            # happens to be sweeps the hand through the scene and leaves the IK badly
            # conditioned, which is how an otherwise good pinch closes on air.  The
            # hover height is IK-probed: a grasp deep in a drawer is reachable while the
            # point 10 cm above it is not, and a hover the arm cannot make leaves the
            # fingers 9 cm short and 20 deg off before the descent even starts.
            hover, hover_ok = self._reachable_hover(target, quat, approach_m)
            report["hover_m"] = round(float(hover), 4)
            report["hover_reachable"] = bool(hover_ok)
            here = self.tcp()
            # axis-aligned legs only: vertical at the current xy, then horizontal at the
            # staging height, then straight down — never a diagonal through the scene
            self.goto(np.array([here[0], here[1], hover]), quat)
            self.goto(np.array([target[0], target[1], hover]), quat)
            approach_err = self.goto(target, quat)
            report["approach_error_m"] = round(float(approach_err), 4)
            self.close_fingers()
            self.settle(5)
            opening = self.get_gripper_opening()
            self.move_by_displacement(z=lift_m)
            after = self.body_position(object_name)
            rise = float(after[2] - before[2])
            report.update(
                {
                    "opening_m": round(opening, 4),
                    "lift_m": round(rise, 4),
                    "attempts": attempt + 1,
                    "grasp_point": [round(float(v), 4) for v in target],
                }
            )
            if opening > HELD_OPENING and rise > 0.4 * lift_m:
                report["grasped"] = True
                return report
            self.open_fingers()
            self.settle(10)
            self.move_by_displacement(z=-lift_m + 0.02)
            ext = self.get_object_extents(object_name)
            plan = self.plan_pinch(object_name, depth_m=depth_m)
            point = np.array([plan["point"][0], plan["point"][1], 0.0])
            report["plan"] = plan
            if pinch_axis is None:
                close_dir = np.asarray(plan["close_dir"], dtype=float)
                quat = tool_down_quat_wxyz(close_dir)
        return report

    def _reachable_hover(self, target: Any, quat: Any, approach_m: float) -> tuple[float, bool]:
        """A staging height over ``target`` that the position-only IK probe accepts with
        orientation ``quat`` (nothing moves), and whether any candidate was accepted.

        The first candidate is ``max(current hand height, target + approach_m)``; lower
        fractions of ``approach_m`` follow.  The probe is :meth:`check_reachable` — a
        seedless solve that ignores collisions and the current posture — so in every
        LIBERO cabinet scene measured (goal_3, spatial_4, 90_6, 90_24, 10_3) it accepts
        the first candidate and this is exactly ``max(here, target + approach_m)``; it
        does NOT detect the drawer-wall stall that ``grasp_object`` reports as
        ``approach_error_m`` (spatial_4: 7.4 cm short with every candidate "reachable").
        When nothing is accepted the lowest fraction is returned with ``False``.
        """
        if float(approach_m) <= 0.0:
            raise ValueError("approach_m must be > 0")
        point = np.asarray(target, dtype=float)
        here = float(self.tcp()[2])
        top = float(point[2] + float(approach_m))
        candidates = [max(here, top)] + [
            float(point[2] + float(approach_m) * f) for f in (1.0, 0.7, 0.5, 0.35, 0.2)
        ]
        for z in dict.fromkeys(round(c, 6) for c in candidates):
            if self.check_reachable([point[0], point[1], z], quat).get("reachable"):
                return float(z), True
        return float(point[2] + 0.2 * float(approach_m)), False

    def carry_to(
        self,
        x: float,
        y: float,
        *,
        height_m: float | None = None,
        clearance_m: float = 0.10,
        object_name: str | None = None,
        corrections: int = 2,
        tolerance_m: float = 0.004,
    ) -> list[float]:
        """Move the (held) object over ``x, y`` without dragging it through the scene:
        lift to a travel height first, then translate, keeping the hand orientation.

        With ``object_name`` the target is the OBJECT's centre, not the fingertips — the
        offset between the two is measured now, so a rim-pinched bowl still lands centred.
        ``height_m`` overrides the travel height (default: the current fingertip height
        raised by ``clearance_m``).  Returns the fingertip position.
        """
        _, quat = self.hand_pose()
        tcp = self.tcp()
        offset = np.zeros(2)
        if object_name is not None:
            obj = self.body_position(object_name)
            offset = obj[:2] - tcp[:2]
        travel = float(tcp[2] + clearance_m) if height_m is None else float(height_m)
        travel = max(travel, float(tcp[2]))
        if travel > tcp[2] + 1e-3:
            self.goto(np.array([tcp[0], tcp[1], travel]), quat)
        goal = np.array([float(x) - offset[0], float(y) - offset[1], travel])
        self.goto(goal, quat)
        if object_name is not None:
            # the load swings while it travels, so close the loop on the OBJECT: measure
            # where it actually hangs and slide the hand by the residual
            for _ in range(int(corrections)):
                obj = self.body_position(object_name)
                residual = np.array([float(x), float(y)]) - obj[:2]
                if np.linalg.norm(residual) < tolerance_m:
                    break
                self.move_by_displacement(float(residual[0]), float(residual[1]), 0.0)
        return [float(v) for v in self.tcp()]

    def release_over(
        self, x: float, y: float, z: float, *, object_name: str | None = None, settle: int = 30
    ) -> list[float]:
        """Carry to ``x, y``, descend to fingertip height ``z`` and open the gripper.

        The drop for containers: a basket or a drawer is scored on the object's centre
        being inside the box, and lowering the *fingers* into a narrow box usually jams
        them on the rim.  ``object_name`` centres the OBJECT (not the fingertips) on
        ``x, y`` and reports where it came to rest.
        """
        _, quat = self.hand_pose()
        self.carry_to(x, y, height_m=max(z, self.tcp()[2]), object_name=object_name)
        tcp = self.tcp()
        if z < tcp[2] - 1e-3:
            self.goto(np.array([tcp[0], tcp[1], float(z)]), quat)
        self.open_fingers()
        self.settle(int(settle))
        if object_name is None:
            return [float(v) for v in self.tcp()]
        return [float(v) for v in self.body_position(object_name)]

    def descend_over(
        self,
        object_name: str,
        x: float,
        y: float,
        fingertip_z: float,
        *,
        max_step_m: float = 0.05,
        max_moves: int = 8,
        max_xy_step_m: float = 0.02,
    ) -> dict[str, Any]:
        """Lower the HELD ``object_name`` so its body centre sits over ``(x, y)`` (robot-base
        frame) with the fingertips at ``fingertip_z`` (base frame), re-centring the OBJECT
        (not the fingertips) by at most ``max_xy_step_m`` per move.  Free-air stage only:
        the gripper stays closed and a ``fingertip_z`` below the support shoves it — use
        :meth:`place_gently` for the last centimetres.  Descends at most
        ``max_moves * max_step_m`` and stops after two stalled moves, when the object
        stops following the hand, or at the step horizon.
        Returns ``{"fingertip_z", "reached", "moves", "stalls", "following", "xy_error_m",
        "reason"}``.

        (:meth:`place_gently` deliberately gives up the moment the hand stops descending,
        which is the right rule against a support but the wrong one in free air: after a
        long reach the first small step can stall on its own and the load then gets
        dropped from 5 cm up — that is why this stage exists.)
        """
        max_step_m = abs(float(max_step_m))
        _, quat = self.hand_pose()
        stalls = moves = 0
        following = True
        reason = "reached"
        for _ in range(int(max_moves)):
            tcp = self.tcp()
            if tcp[2] <= float(fingertip_z) + 0.006:
                break
            if self._steps_left() < 100:
                reason = "horizon"
                break
            obj = self.body_position(object_name)
            residual = np.array([float(x), float(y)]) - obj[:2]
            norm = float(np.linalg.norm(residual))
            if norm > max_xy_step_m:
                residual = residual * (max_xy_step_m / norm)
            nxt = max(float(fingertip_z), float(tcp[2]) - max_step_m)
            self.goto(np.array([tcp[0] + residual[0], tcp[1] + residual[1], nxt]), quat)
            moves += 1
            obj1 = self.body_position(object_name)
            hand_dz = float(tcp[2] - self.tcp()[2])
            if hand_dz > 0.01 and float(obj[2] - obj1[2]) < 0.4 * hand_dz:
                following = False
                reason = "object not following the hand"
                break
            if self.tcp()[2] > tcp[2] - 0.3 * (tcp[2] - nxt):
                stalls += 1
                if stalls >= 2:
                    reason = "stalled"
                    break
            else:
                stalls = 0
        else:
            reason = "max_moves"
        z = float(self.tcp()[2])
        obj = self.body_position(object_name)
        return {
            "fingertip_z": z,
            "reached": bool(z <= float(fingertip_z) + 0.006),
            "moves": moves,
            "stalls": stalls,
            "following": following,
            "xy_error_m": round(float(np.hypot(float(x) - obj[0], float(y) - obj[1])), 4),
            "reason": reason if z > float(fingertip_z) + 0.006 else "reached",
        }

    def object_cloud(self, object_name: str) -> np.ndarray:
        """Every collision-geom corner of ``object_name`` as base-frame points (N, 3).

        The raw material behind :meth:`get_object_extents`; use it when a footprint has to
        be measured at some rotation other than the current one.
        """
        _body, _bodies, geoms, _eff = self._object_geoms(object_name, True)
        return np.array(
            [self.base_from_world(p) for g in geoms for p in self._geom_corners(g)], dtype=float
        )

    def fit_yaw(
        self,
        object_name: str,
        region_name: str,
        *,
        candidates: Any = (0, 30, 45, 60, 90, 120, 135),
        clearance_m: float = 0.005,
    ) -> dict[str, Any]:
        """How far the HELD object would have to be turned about the vertical to fit the
        footprint of a scored box region.  Nothing moves.

        Every candidate yaw (plus the two yaws that align the object's principal axis
        with the box, so a coarse grid never misses the exact alignment) is applied to the
        object's own measured corner cloud and compared with the box LIBERO scores —
        ``get_region_boxes``' world-axis ``half_size``, which is the ``In`` predicate's box
        and not the physical walls (for a fixture yawed by other than a quarter turn the
        two differ).  The footprint is the true half extent ``(max - min) / 2`` of the
        rotated cloud (a mesh whose vertex cloud is lopsided about its box centre used to
        be over-sized by up to 6 cm); ``fits`` means it fits with ``clearance_m`` to spare
        on both sides, and candidates are ranked by overhang, then by the LARGEST margin,
        then by the smallest turn — a zero-margin skewed entry no longer beats an aligned
        one.  A rectangular object has the same footprint at ``+yaw`` and ``-yaw``: try
        the negative turn when the wrist limit is near.  ``+yaw`` is counter-clockwise
        seen from above, the sign :meth:`turn_held` takes.  Returns ``{"yaw_deg",
        "overhang_m", "margin_m", "fits", "half_extent", "region_half", "tried"}``.
        """
        cands = [float(c) for c in candidates]
        if not cands:
            raise ValueError("candidates must be non-empty")
        region = self._region(region_name)
        rhalf = np.asarray(region["half_size"], dtype=float)[:2] - float(clearance_m)
        cloud = self.object_cloud(object_name)
        # the box is world-axis aligned: compare in the world frame (the base may be yawed)
        wxyz_xyz = np.asarray(self.lle.base_link_wxyz_xyz, dtype=float).reshape(-1)
        base_rot = _quat_wxyz_to_mat(wxyz_xyz[:4])
        world = (base_rot @ cloud.T).T
        rel = world[:, :2] - world[:, :2].mean(axis=0)
        # principal axis of the footprint -> the yaws that lay it along world x / y
        _, _, vt = np.linalg.svd(rel, full_matrices=False)
        principal = float(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
        for extra in (-principal, 90.0 - principal):
            extra = ((extra + 180.0) % 360.0) - 180.0
            if all(abs(extra - c) > 0.5 for c in cands):
                cands.append(round(extra, 2))
        tried = []
        best = None
        for yaw in cands:
            angle = np.radians(float(yaw))
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
            )
            pts = rel @ rot.T
            half = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
            margin = rhalf - half
            over = float(np.maximum(-margin, 0.0).sum())
            tried.append(
                {
                    "yaw_deg": float(yaw),
                    "overhang_m": round(over, 4),
                    "margin_m": round(float(margin.min()), 4),
                }
            )
            score = (over, -float(margin.min()), abs(float(yaw)))
            if best is None or score < best[0]:
                best = (score, float(yaw), over, half, float(margin.min()))
        assert best is not None
        _, yaw, over, half, margin_min = best
        return {
            "yaw_deg": yaw,
            "overhang_m": round(float(over), 4),
            "margin_m": round(margin_min, 4),
            "fits": bool(over <= 0.0),
            "half_extent": [round(float(v), 4) for v in half],
            "region_half": [round(float(v), 4) for v in (rhalf + float(clearance_m))],
            "tried": tried,
        }

    def turn_held(self, yaw_deg: float, *, object_name: str | None = None) -> dict[str, Any]:
        """Turn the wrist ``yaw_deg`` degrees about the VERTICAL axis, keeping the
        fingertips where they are, so a held object turns with it.

        Do it in free air well above the support: the object swings around the fingertips,
        not around its own centre, so re-centre afterwards (``carry_to(..., object_name=)``
        and the ``place_*`` primitives already do).  The wrist rotation is VERIFIED: ``goto``
        only closes the position loop, so a large yaw used to come back 25-35 deg short
        with no report (measured: 170 commanded, 145 turned); the residual is now
        re-commanded up to twice and ``hand_turned_deg`` / ``reached`` (within 3 deg) say
        what the hand did.  With ``object_name`` the report also says how far the OBJECT
        turned (signed, from its body rotation) and ``slip_deg`` = hand minus object, so a
        grasp that slipped in the wrist is visible.  Returns ``{"commanded_deg",
        "hand_turned_deg", "turned_deg", "slip_deg", "reached", "position_error_m"}``.
        """

        def yaw_of(rot: np.ndarray) -> float:
            return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

        def signed(delta: float) -> float:
            return ((delta + 180.0) % 360.0) - 180.0

        def object_rot() -> np.ndarray | None:
            if object_name is None:
                return None
            ext = self.get_object_extents(object_name)
            quat = np.asarray(self.sim.data.get_body_xquat(ext["body"]), dtype=float)
            return _quat_wxyz_to_mat(quat)

        obj_before = object_rot()
        _, quat0 = self.hand_pose()
        hand0 = _quat_wxyz_to_mat(quat0)
        target = self.tcp()
        remaining = float(yaw_deg)
        error = float("inf")
        turned_hand = 0.0
        for _attempt in range(3):
            angle = np.radians(remaining)
            spin = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            _, quat = self.hand_pose()
            error = self.goto(target, _mat_to_quat_wxyz(spin @ _quat_wxyz_to_mat(quat)))
            _, quat1 = self.hand_pose()
            delta_rot = _quat_wxyz_to_mat(quat1) @ hand0.T
            turned_hand = signed(yaw_of(delta_rot))
            remaining = signed(float(yaw_deg) - turned_hand)
            if abs(remaining) < 3.0:
                break
        turned = None
        slip = None
        obj_after = object_rot()
        if obj_before is not None and obj_after is not None:
            turned = round(signed(yaw_of(obj_after @ obj_before.T)), 1)
            slip = round(signed(turned_hand - turned), 1)
        return {
            "commanded_deg": float(yaw_deg),
            "hand_turned_deg": round(turned_hand, 1),
            "turned_deg": turned,
            "slip_deg": slip,
            "reached": bool(abs(remaining) < 3.0),
            "position_error_m": round(float(error), 4),
        }

    def place_in_region(
        self,
        object_name: str,
        region_name: str,
        *,
        mode: str = "lower",
        margin_m: float = 0.06,
        settle: int = 40,
        offset_xy: Any = (0.0, 0.0),
        floor_gap_m: float = 0.01,
        scored_name: str | None = None,
    ) -> dict[str, Any]:
        """Put the HELD object inside a scored box region — a basket, an open drawer, a
        microwave cavity, a caddy compartment (:meth:`get_region_boxes`).
        ``scored_name`` names the body the ``In`` predicate is judged on when the grip
        body is a sub-body (the carry centres ``object_name``; ``inside``/``object_pos``
        read ``scored_name`` — the two origins were 4.9 cm apart on a salad dressing).

        Centres the OBJECT (not the fingertips) over the region, then lowers it in with
        :meth:`place_gently` (``mode="lower"``) — a tall carton released above the rim
        topples straight back out, so the default really is to reach in and let go once the
        object is standing.  ``mode="drop"`` releases ``margin_m`` above the top face
        instead, for a region the hand cannot enter.  ``mode="seat"`` is the middle case
        and the one narrow compartments need: the object is lowered until ITS OWN bottom is
        ``floor_gap_m`` above the region's BOX floor and released there, with the fingers
        left wherever that puts them — a book dropped from the rim of a 5.5 cm caddy slot
        topples and lands outside it, while the same book let go 1 cm off the floor stands.
        The box floor is the scorer's, not the physical one (caddy: box floor +0.0001 vs
        base plate -0.0112; basket: -0.0025 vs +0.017), so the real drop can be 1-2 cm
        more or less than ``floor_gap_m`` — ``release_gap_m`` reports the measured gap at
        release.  ``margin_m`` is not used by "seat" and ``floor_gap_m`` only by it.
        ``inside`` reports LIBERO's own ``(In object region)`` test on the object's resting
        body origin — a standing object can still fail it when its origin sits below its
        box (libero.10_5's book: -0.0134 against a -0.0099 floor, measured), and it is
        False whenever the release was refused at the horizon (``released``).  Measured
        cost of "seat": ~480-1150 steps.
        """
        if mode not in ("lower", "drop", "seat"):
            raise ValueError("mode must be 'lower', 'drop' or 'seat'")
        judge = scored_name or object_name
        region = self._region(region_name)
        centre = np.asarray(region["center"], dtype=float)
        centre[:2] += np.asarray(offset_xy, dtype=float)[:2]
        if (why := self._not_held(object_name)) is not None:
            return {
                "inside": False,
                "object_pos": [float(v) for v in self.body_position(judge)],
                "region": region_name,
                "mode": mode,
                "reason": why,
            }
        held = self.get_object_extents(object_name)
        obj = self.body_position(object_name)
        tcp = self.tcp()
        extra: dict[str, Any] = {}
        hover = float(region["top_z"]) + margin_m + (obj[2] - float(held["bottom_z"]))
        z_fingers = hover + float(tcp[2] - obj[2])
        if mode == "lower":
            self.carry_to(
                centre[0], centre[1], height_m=max(z_fingers, tcp[2]), object_name=object_name
            )
            self.descend_over(object_name, centre[0], centre[1], z_fingers)
            self.place_gently(object_name, max_descent=0.30, target_xy=centre[:2])
            self.settle(int(settle))
            rest = [float(v) for v in self.body_position(object_name)]
        elif mode == "seat":
            floor = float(np.asarray(region["lo"], dtype=float)[2]) + float(floor_gap_m)
            seat_z = float(tcp[2]) - (float(held["bottom_z"]) - floor)
            # travel high enough for the object's bottom to clear the RIM (z_fingers),
            # not only the box floor: a book carried 1 cm below the caddy rim sweeps it
            self.carry_to(
                centre[0],
                centre[1],
                height_m=max(seat_z + 0.10, z_fingers, tcp[2]),
                object_name=object_name,
            )
            # the load settles in the jaws during the carry (a rim-pinched bowl slipped
            # 3 cm): re-measure before computing the release height
            tcp_now = self.tcp()
            bottom_now = float(self.get_object_extents(object_name)["bottom_z"])
            seat_z = float(tcp_now[2]) - (bottom_now - floor)
            moves = int(np.ceil(max(float(tcp_now[2]) - seat_z, 0.0) / 0.03)) + 2
            descent = self.descend_over(
                object_name, centre[0], centre[1], seat_z, max_step_m=0.03, max_moves=moves
            )
            released = self.open_fingers()
            self.settle(int(settle))
            rest = [float(v) for v in self.body_position(object_name)]
            gap = float(self.get_object_extents(object_name)["bottom_z"]) - (
                floor - float(floor_gap_m)
            )
            extra = {
                "released": bool(released),
                "descent_reached": bool(descent.get("reached")),
                "descent_reason": descent.get("reason"),
                "following": descent.get("following"),
                "release_gap_m": round(gap, 4),
                "budget_exhausted": not released,
            }
        else:
            rest = self.release_over(
                centre[0], centre[1], z_fingers, object_name=object_name, settle=settle
            )
        if extra.get("released", True):
            # lift the open fingers clear FIRST: rising past a basket handle or a drawer
            # front can still shove the fixture, and the verdict must be read after it
            self.move_by_displacement(z=0.08)
            self.settle(5)
        rest = [float(v) for v in self.body_position(judge)]
        # the box rides with its fixture: re-read it, a basket or drawer shoved during the
        # placement moves the scorer's box out from under the stale one
        region = self._region(region_name)
        lo, hi = np.asarray(region["lo"], dtype=float), np.asarray(region["hi"], dtype=float)
        lo = lo - np.array([0.0, 0.0, 0.01])
        inside = bool(np.all(np.asarray(rest) > lo) and np.all(np.asarray(rest) < hi))
        if not extra.get("released", True):
            inside = False  # still in the jaws: the scorer sees a held object, not a placed one
        return {"inside": inside, "object_pos": rest, "region": region_name, "mode": mode, **extra}

    def push_object(
        self,
        object_name: str,
        direction: Any,
        distance_m: float = 0.03,
        *,
        height_frac: float = 0.5,
        standoff_m: float = 0.04,
        settle: int = 20,
        close_dir: Any = "vertical",
        pitches: Any = None,
    ) -> dict[str, Any]:
        """Push a resting object along a horizontal base-frame ``direction`` with the
        CLOSED gripper.

        The fingers are shut into a stub, placed against the object's NEAR face (the one
        the hand comes from, at the centre of that face's band) at ``height_frac`` of its
        height (0 = bottom, 1 = top; the contact is floored a pad-thickness above the
        support) and driven ``distance_m`` along ``direction``.  The wrist pitches down as
        far as the IK probe needs (:meth:`reachable_tool_quat`); at 45-60 deg the stub
        pushes DOWN as well as along, so for a slide keep ``height_frac`` <= 0.3 and read
        ``tilt_deg`` (measured 2026-08-27: a bowl pushed at half height with a 60 deg wrist
        moved 1.7 of 3 cm and tilted 29 deg; a wine bottle fell over).  The primitive
        refuses before moving when no pitch reaches the approach, and aborts with
        ``reason="approach missed"`` when the hand ends more than 2 cm from the approach
        point (it used to press from 27 cm away and report a blocked push).  The hand is
        left in contact: ``park()`` before the next primitive.  ``close_dir`` is the
        direction the closed fingers lie along: ``"vertical"`` (default, the certified
        behaviour) makes the hand 20 cm tall and low pushes need 45-60 deg of pitch;
        ``"horizontal"`` lays the fingers across the push so the hand is 2 cm tall and a
        bowl or a pan on the table is pushed level at pitch 0 (the same fact as the
        bottom-drawer close, libero.90_22) — but a contact lower than ~3 cm over the table
        puts the WRIST in the table with a level tool (the approach ended 24 cm off on a
        bowl at z 0.013): pass ``pitches=(30.0,)`` there, the pitch lifts the wrist 5 cm
        for the same fingertip height.  ``pitches`` pins the wrist-pitch ladder (default
        :data:`PUSH_PITCHES`, flattest the IK probe accepts).  Returns ``{"moved_m",
        "tilt_deg", "before", "after", "pitch_deg", "reached", "reason",
        "approach_error_m", "fingers_closed", "budget_exhausted"}``; ``tilt_deg`` is how
        much the object's own vertical axis leaned.  Measured cost 650-720 steps for a
        first push, ~350 for a follow-up from the contact.
        """
        unit = np.asarray(direction, dtype=float)
        unit = np.array([unit[0], unit[1], 0.0])
        norm = float(np.linalg.norm(unit))
        if norm < 1e-6:
            raise ValueError("direction must have a horizontal component")
        unit = unit / norm
        ext = self.get_object_extents(object_name)
        lo, hi = np.asarray(ext["lo"], dtype=float), np.asarray(ext["hi"], dtype=float)
        z = float(lo[2] + np.clip(float(height_frac), 0.0, 1.0) * (hi[2] - lo[2]))
        z = max(z, float(lo[2]) + 0.012)  # the pads are ~1 cm tall: do not aim at the support
        cloud = self.object_cloud(object_name)
        band = cloud[np.abs(cloud[:, 2] - z) < 0.02]
        if band.shape[0] < 2:
            band = cloud
        # the near face: its extreme along the push, centred across it (an extreme
        # VERTEX of a yawed box is a corner, and pushing a corner spins the object)
        proj = band[:, :2] @ unit[:2]
        perp = np.array([-unit[1], unit[0]])
        lateral = float(np.median(band[:, :2] @ perp))
        face_xy = unit[:2] * float(proj.min()) + perp * lateral
        face = np.array([face_xy[0], face_xy[1], z])
        approach = face - unit * float(standoff_m)
        press = face + unit * float(distance_m)
        before_pos = self.body_position(object_name)
        before_axis = self._object_up_axis(object_name)

        def report(reached: bool, reason: str, pitch: float, err: float, closed: bool) -> dict:
            after_pos = self.body_position(object_name)
            after_axis = self._object_up_axis(object_name)
            cos = float(np.clip(before_axis @ after_axis, -1.0, 1.0))
            return {
                "moved_m": round(float(np.linalg.norm(after_pos[:2] - before_pos[:2])), 4),
                "tilt_deg": round(float(np.degrees(np.arccos(cos))), 2),
                "before": [round(float(v), 4) for v in before_pos],
                "after": [round(float(v), 4) for v in after_pos],
                "pitch_deg": pitch,
                "reached": bool(reached),
                "reason": reason,
                "approach_error_m": round(float(err), 4),
                "fingers_closed": bool(closed),
                "budget_exhausted": self._steps_left() <= 1,
            }

        finger_dir = self._finger_dir(close_dir, unit)
        perp = finger_dir - unit * float(finger_dir @ unit)
        if float(np.linalg.norm(perp)) < 0.1 * max(float(np.linalg.norm(finger_dir)), 1e-9):
            raise ValueError("close_dir must not be parallel to the push direction")
        pose = self.reachable_tool_quat(
            [approach, press],
            unit,
            finger_dir,
            pitches=PUSH_PITCHES if pitches is None else pitches,
        )
        quat = np.asarray(pose["quat_wxyz"], dtype=float)
        if not pose["reachable"]:
            return report(
                False, "no pitch reaches the approach/press points", pose["pitch_deg"], 0.0, False
            )
        opening0 = float(self.get_gripper_opening())
        if HELD_OPENING < opening0 < GRIPPER_MAX_OPENING - 0.01:
            return report(
                False,
                f"the gripper is holding something (opening {opening0:.3f} m): release it first",
                pose["pitch_deg"],
                0.0,
                False,
            )
        closed = self.close_fingers()
        if not closed:
            return report(False, "no budget to close the fingers", pose["pitch_deg"], 0.0, False)
        raised = np.array([approach[0], approach[1], max(approach[2] + 0.10, self.tcp()[2])])
        self.goto(raised, quat)
        err = self.goto(approach, quat)
        if err > 0.02:
            return report(False, "approach missed", pose["pitch_deg"], err, closed)
        reached = self.goto(press, quat, iterations=2, max_iterations=3) < 0.02
        self.settle(int(settle))
        return report(reached, "reached" if reached else "blocked", pose["pitch_deg"], err, closed)

    def _object_up_axis(self, object_name: str) -> np.ndarray:
        """The object body's own +z axis, in the base frame (how upright it stands)."""
        ext = self.get_object_extents(object_name)
        quat = np.asarray(self.sim.data.get_body_xquat(ext["body"]), dtype=float)
        world = _quat_wxyz_to_mat(quat) @ np.array([0.0, 0.0, 1.0])
        return self.base_from_world(world) - self.base_from_world([0.0, 0.0, 0.0])

    def place_on(
        self,
        object_name: str,
        support_name: str,
        *,
        clearance_m: float = 0.06,
        xy_tolerance_m: float = 0.025,
        settle: int = 30,
        retries: int = 0,
        regrasp: str = "rim",
        scored_name: str | None = None,
    ) -> dict[str, Any]:
        """Put the HELD object on top of another object, centred well enough to score.

        LIBERO's ``(On a b)`` needs the two bodies in contact and their centres within
        3 cm horizontally, so "roughly above" is not enough: this measures the support's
        top face and centre, carries the OBJECT (not the fingertips) over it, lowers with
        :meth:`place_gently` and reports the residual centre-to-centre distance.
        A first placement lands 0.9-2.9 cm off run to run and can SHOVE a light plate
        5.7-6.7 cm (``pushed_support``, measured on the stove group): with ``retries`` > 0
        a residual over ``xy_tolerance_m`` (or a lost contact) is closed by
        :meth:`recentre_on` — re-pinch (``regrasp`` "rim" for bowls/plates/mugs, "box" for
        boxes and cans), hop by the measured centre-to-centre vector, set down — up to
        ``retries`` hops (~900-1400 steps each; ``hops`` reports how many ran, 0 when the
        first placement was good).  ``scored_name`` names the body the predicate is judged
        on when the grip body is a sub-body (a salad dressing's ``_main`` is 4.9 cm from
        the sub-body that carries its geoms): the carry still centres ``object_name``,
        ``xy_offset_m``/``touching``/``object_pos`` are read on ``scored_name``, and so is
        the residual the retry hops by.  ``descent_m``/``pushed_support``/``released``
        describe the FIRST placement — ``hops`` says whether more happened after them.
        """
        judge = scored_name or object_name
        support = self.get_object_extents(support_name)
        centre = self.body_position(support_name)  # what the (On a b) predicate compares
        held = self.get_object_extents(object_name)
        obj = self.body_position(object_name)
        tcp = self.tcp()
        if (why := self._not_held(object_name)) is not None:
            return {
                "placed": False,
                "xy_offset_m": None,
                "object_pos": [float(v) for v in self.body_position(judge)],
                "descent_m": 0.0,
                "touching": False,
                "hops": 0,
                "reason": why,
            }
        travel = float(
            support["top_z"] + clearance_m + (obj[2] - held["bottom_z"]) + tcp[2] - obj[2]
        )
        self.carry_to(centre[0], centre[1], height_m=max(travel, tcp[2]), object_name=object_name)
        obj = self.body_position(object_name)
        near = float(
            support["top_z"] + 0.015 + (obj[2] - held["bottom_z"]) + (self.tcp()[2] - obj[2])
        )
        self.descend_over(object_name, centre[0], centre[1], near)
        report = self.place_gently(
            object_name,
            max_descent=0.30,
            step=0.012,
            watch_name=support_name,
            target_xy=centre[:2],
        )
        self.settle(int(settle))
        obj = self.body_position(judge)
        centre = self.body_position(support_name)
        offset = float(np.linalg.norm(obj[:2] - centre[:2]))
        touching = self.in_contact(judge, support_name)  # (On a b) needs contact too
        hops = 0
        fix: dict[str, Any] | None = None
        if int(retries) > 0 and report["released"] and (offset >= xy_tolerance_m or not touching):
            fix = self.recentre_on(
                object_name,
                support_name,
                tolerance_m=float(xy_tolerance_m),
                tries=int(retries),
                regrasp=regrasp,
                scored_name=scored_name,
            )
            hops = int(fix["hops"])
            self.settle(int(settle))
            obj = self.body_position(judge)
            centre = self.body_position(support_name)
            offset = float(np.linalg.norm(obj[:2] - centre[:2]))
            touching = self.in_contact(judge, support_name)
        # `rested` is a measured post-condition of place_gently, so a hop only carries it
        # when the hop itself succeeded (a hop that RAN and failed proves nothing)
        rested = bool(report["rested"] or (fix is not None and fix["ok"]))
        return {
            "placed": bool(rested and offset < xy_tolerance_m and touching),
            "xy_offset_m": round(offset, 4),
            "touching": touching,
            "object_pos": [float(v) for v in obj],
            "descent_m": report["descent_m"],
            "pushed_support": report["pushed_support"],
            "released": report["released"],
            "hops": hops,
            "recentre_reason": None if fix is None else fix.get("reason"),
        }

    def place_on_region(
        self,
        object_name: str,
        region_name: str,
        *,
        clearance_m: float = 0.06,
        settle: int = 30,
        offset_xy: Any = (0.0, 0.0),
        scored_name: str | None = None,
    ) -> dict[str, Any]:
        """Put the HELD object down on a scored box region (a table patch, a shelf, a
        cabinet top — anything :meth:`get_region_boxes` lists).

        ``(On object region)`` is true when the object's centre sits just above the box's
        top face, inside its xy footprint, and touches the fixture the region belongs to.
        This carries the OBJECT (not the fingertips) over the box centre, lowers with
        :meth:`place_gently` and checks that exact condition in the region's own frame.
        ``offset_xy`` shifts the aim off the centre, which is how two objects fit in one
        region (two pots on one hob) without the second landing on the first.
        ``scored_name`` names the body the predicate is judged on when it is not the grip
        body (an object whose geoms hang on a sub-body: the carry centres ``object_name``,
        ``delta_in_region`` / ``touching_fixture`` / ``object_pos`` read ``scored_name``).
        """
        judge = scored_name or object_name
        region = self._region(region_name)
        centre = np.asarray(region["center"], dtype=float)
        centre[:2] += np.asarray(offset_xy, dtype=float)[:2]
        held = self.get_object_extents(object_name)
        obj = self.body_position(object_name)
        tcp = self.tcp()
        if (why := self._not_held(object_name)) is not None:
            return {
                "on_region": False,
                "touching_fixture": None,
                "delta_in_region": None,
                "object_pos": [float(v) for v in self.body_position(judge)],
                "descent_m": 0.0,
                "reason": why,
            }
        travel = float(
            region["top_z"] + clearance_m + (obj[2] - held["bottom_z"]) + (tcp[2] - obj[2])
        )
        self.carry_to(centre[0], centre[1], height_m=max(travel, tcp[2]), object_name=object_name)
        obj = self.body_position(object_name)
        near = float(
            region["top_z"] + 0.015 + (obj[2] - held["bottom_z"]) + (self.tcp()[2] - obj[2])
        )
        self.descend_over(object_name, centre[0], centre[1], near)
        report = self.place_gently(object_name, max_descent=0.30, step=0.012, target_xy=centre[:2])
        self.settle(int(settle))
        obj = self.body_position(judge)
        region = self._region(region_name)
        centre = np.asarray(region["center"], dtype=float)
        size = np.asarray(region["site_size"], dtype=float)  # the box the predicate uses
        mat = np.asarray(region["site_mat"], dtype=float).reshape(3, 3)
        delta = mat @ (obj - centre)  # LIBERO's own "under" test, in the region's frame
        # (On x region) also demands contact with the fixture the region belongs to —
        # LIBERO's SiteObject.parent_name; table/floor patches have none and need no contact
        parent = region.get("parent")
        touching: bool | None = True
        if parent:
            try:
                touching = self.in_contact(judge, parent)
            except KeyError:
                touching = None  # the parent is not a body we can see: unknown, not True
        on = bool(
            size[2] - 0.005 < delta[2] < size[2] + 0.10
            and np.all(np.abs(delta[:2]) < size[:2])
            and touching is True
        )
        return {
            "on_region": on,
            "touching_fixture": touching,
            "delta_in_region": [round(float(v), 4) for v in delta],
            "object_pos": [float(v) for v in obj],
            "descent_m": report["descent_m"],
            "released": report["released"],
        }

    def _region(self, region_name: str) -> dict[str, Any]:
        """One scored region by exact name, or a unique case-insensitive substring."""
        boxes = self.get_region_boxes()
        if region_name in boxes:
            return boxes[region_name]
        hits = [n for n in boxes if region_name.lower() in n.lower()]
        if len(hits) == 1:
            return boxes[hits[0]]
        raise KeyError(
            f"region {region_name!r} "
            + ("is ambiguous: " + ", ".join(sorted(hits)) if hits else "not found")
            + "; see get_region_boxes()"
        )

    def _not_held(self, object_name: str) -> str | None:
        """Why ``object_name`` is plainly not in the jaws right now (None when it may be).

        Only the unambiguous cases are refused: an OPEN gripper, or the object far from
        the fingertips.  A near-closed reading is NOT one — a bowl held by its 5 mm rim
        reads 0.004-0.005 after a lift (measured 2026-08-27, libero.spatial_5) and the
        grasp was verified by the object rising, not by the gap.
        """
        if self.get_gripper_opening() > GRIPPER_MAX_OPENING - 0.005:
            return "nothing held: the gripper is open"
        # a body origin can sit 10+ cm from the pinch (a bottle held near its top): only
        # refuse when it is clearly elsewhere
        if float(np.linalg.norm(self.body_position(object_name) - self.tcp())) > 0.25:
            return f"{object_name} is not at the fingertips"
        return None

    # ------------------------------------------------------------ articulation

    def get_joint_info(self, joint_name: str = "") -> dict[str, Any]:
        """Range, type and axis of the scene's articulated joints (base frame).

        ``get_joint_states`` only gives you the numbers; without the *range* you cannot
        tell a shut drawer from a fully pulled one, or how far a knob still has to turn.
        Returns ``{joint: {"qpos", "range", "type", "fraction", "axis", "anchor",
        "limited"}}`` — ``fraction`` is only the position inside ``[lo, hi]`` and carries
        NO open/closed meaning (LIBERO's cabinet drawers rest at qpos 0 in a range of
        [-0.16, 0.01], so a SHUT drawer reads 0.94 and a pulled one 0.05; Open is
        qpos < -0.14, Close is qpos > 0), ``type`` is ``"slide"``/``"hinge"``.  Pass
        ``joint_name`` for a single joint; a free/ball joint raises.
        """
        sim = self.sim
        raw_m, _ = self._raw()
        names = []
        for i in range(raw_m.njnt):
            n = sim.model.joint_id2name(i)
            if not n or "robot" in n or "gripper" in n:
                continue
            if int(raw_m.jnt_type[i]) not in (MJ_SLIDE, MJ_HINGE):
                continue
            names.append(n)
        if joint_name:
            names = [n for n in names if n == joint_name]
            if not names:
                raise KeyError(f"{joint_name!r} is not a slide/hinge joint of the scene")
        out: dict[str, Any] = {}
        for n in names:
            info = self._joint_info(n)
            lo, hi = (float(v) for v in info["range"])
            span = hi - lo
            anchor = self.base_from_world(info["anchor_world"])
            axis = self.base_from_world(info["anchor_world"] + info["axis_world"]) - anchor
            out[n] = {
                "qpos": info["qpos"],
                "range": [lo, hi],
                "type": "slide" if info["type"] == MJ_SLIDE else "hinge",
                "fraction": round((info["qpos"] - lo) / span, 4) if span > 0 else 0.0,
                "axis": [round(float(v), 4) for v in axis / max(np.linalg.norm(axis), 1e-9)],
                "anchor": [round(float(v), 4) for v in anchor],
                "limited": bool(raw_m.jnt_limited[info["id"]]),
            }
        return out

    def _joint_body_cloud(self, body_id: int) -> np.ndarray:
        """Base-frame corner cloud of every collision geom on a body."""
        geoms = self._body_geoms(body_id, True) or self._body_geoms(body_id, False)
        return np.concatenate(
            [np.array([self.base_from_world(p) for p in self._geom_corners(g)]) for g in geoms]
        )

    def turn_knob(
        self,
        joint_name: str,
        *,
        target_qpos: float | None = None,
        target_fraction: float = 0.35,
        step_rad: float = 0.12,
        max_turns: int = 20,
        depth_m: float = 0.008,
        band_m: float = 0.015,
        attempts: int = 2,
    ) -> dict[str, Any]:
        """Turn a knob/dial (a hinge joint with a roughly vertical axis) by pinching its
        top and rotating the WRIST about the joint's own axis.

        Pushing a dial tangentially slips; a top-down pinch centred on the hinge axis does
        not, because the fingers and the knob then share a rotation axis.  The pinch
        direction is measured: within ``band_m`` of the knob's top the narrowest horizontal
        width is chosen, so the fingers close across the dial's short side.
        ``target_qpos`` (radians) or ``target_fraction`` of the joint range sets how far to
        turn — LIBERO's stove counts as on from ``qpos >= 0.5``.
        Returns ``{"turned", "qpos", "start_qpos", "turns", "opening_m"}``.
        """
        info = self._joint_info(joint_name)
        if info["type"] != MJ_HINGE:
            raise RuntimeError(f"{joint_name} is not a hinge joint; use open_articulated")
        lo, hi = (float(v) for v in info["range"])
        start = info["qpos"]
        goal = (
            float(target_qpos)
            if target_qpos is not None
            else start + float(target_fraction) * (hi - lo)
        )
        goal = float(np.clip(goal, lo, hi))
        sign = 1.0 if goal >= start else -1.0
        anchor = self.base_from_world(info["anchor_world"])
        axis = self.base_from_world(info["anchor_world"] + info["axis_world"]) - anchor
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        if abs(float(axis[2])) < 0.8:
            raise RuntimeError(
                f"{joint_name}: hinge axis {np.round(axis, 3).tolist()} is not vertical; "
                "turn_knob pinches from above — use open_articulated/close_articulated"
            )
        if sign * (start - goal) >= 0:
            return {
                "turned": True,
                "qpos": round(start, 4),
                "start_qpos": round(start, 4),
                "target_qpos": round(goal, 4),
                "turns": 0,
                "opening_m": 0.0,
            }
        cloud = self._joint_body_cloud(info["body_id"])
        radius = float(np.linalg.norm(cloud[:, :2] - anchor[:2], axis=1).max())
        if radius > 0.5 * GRIPPER_MAX_OPENING + 0.03:
            raise RuntimeError(
                f"{joint_name}: the hinged body reaches {radius:.2f} m from its axis — a "
                "door or lever, not a knob; use open_articulated/close_articulated "
                "(measured 2026-08-27: a microwave door here burned the whole horizon)"
            )
        top = float(cloud[:, 2].max())
        band = cloud[cloud[:, 2] > top - band_m]
        rel = band[:, :2] - anchor[:2]
        best = None
        for deg in np.arange(0.0, 180.0, 5.0):
            unit = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            width = 2 * float(np.abs(rel @ unit).max())
            if best is None or width < best[0]:
                best = (width, unit)
        width, unit = best
        if width > GRIPPER_MAX_OPENING - 0.005:
            raise RuntimeError(
                f"{joint_name}: the knob is {width:.3f} m across in every direction — "
                "wider than the gripper; push it instead of pinching it"
            )
        close_dir = np.array([unit[0], unit[1], 0.0])
        grip = np.array([anchor[0], anchor[1], top - depth_m])

        def qpos() -> float:
            return float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])

        turns = 0
        opening = 0.0
        for _ in range(int(attempts)):
            quat0 = tool_down_quat_wxyz(close_dir)
            self.open_fingers()
            self.goto(grip + np.array([0.0, 0.0, 0.10]), quat0)
            self.goto(grip, quat0)
            self.close_fingers()
            self.settle(5)
            opening = self.get_gripper_opening()
            rot0 = _quat_wxyz_to_mat(quat0)
            angle = 0.0
            stalled = 0
            if opening < HELD_OPENING:  # closed on air: rotating the wrist turns nothing
                self.open_fingers()
                self.settle(5)
                self.move_by_displacement(z=0.08)
                cloud = self._joint_body_cloud(info["body_id"])
                top = float(cloud[:, 2].max())
                grip = np.array([anchor[0], anchor[1], top - depth_m])
                continue
            while turns < int(max_turns) and sign * (qpos() - goal) < 0:
                if self._steps_left() < 250:  # keep room for the release and lift
                    break
                before = qpos()
                angle += sign * min(float(step_rad), abs(goal - qpos()) + 0.05)
                self.goto(grip, _mat_to_quat_wxyz(_rot_about_axis(axis, angle) @ rot0))
                turns += 1
                if abs(qpos() - before) < 0.2 * step_rad:
                    stalled += 1
                    if stalled >= 2:
                        break
                else:
                    stalled = 0
            if sign * (qpos() - goal) >= 0:
                break
            self.open_fingers()
            self.settle(5)
            self.move_by_displacement(z=0.08)
            cloud = self._joint_body_cloud(info["body_id"])
            top = float(cloud[:, 2].max())
            grip = np.array([anchor[0], anchor[1], top - depth_m])
        self.open_fingers()
        self.settle(5)
        self.move_by_displacement(z=0.10)
        return {
            "turned": bool(sign * (qpos() - goal) >= 0),
            "qpos": round(qpos(), 4),
            "start_qpos": round(start, 4),
            "target_qpos": round(goal, 4),
            "turns": turns,
            "opening_m": round(opening, 4),
        }

    def close_articulated(
        self,
        joint_name: str,
        *,
        target_qpos: float = 0.0,
        stroke: float = 0.05,
        max_strokes: int = 15,
        standoff_m: float = 0.06,
        pitches: Any = None,
        close_dir: Any = None,
        park_after: bool = True,
    ) -> dict[str, Any]:
        """Push a drawer or door shut with the CLOSED gripper (the reverse of
        :meth:`open_articulated`).

        No grasp is needed to close something: the fingers are shut into a rigid stub and
        pressed against the moving part's outer face — the face that protrudes farthest
        along the opening direction — then driven back along the joint in verified strokes.
        LIBERO's ``(Close x)`` wants the joint back past its closed end (WhiteCabinet/
        WoodenCabinet ``qpos > 0.0``, Microwave ``qpos > -0.005``, ShortCabinet/ShortFridge
        ``qpos < 0.0``), so ``target_qpos`` defaults to 0 and ``closed`` is STRICT
        (``qpos`` is returned so you can test the benchmark's own predicate).  ``stroke``
        is really how far past the face the controller target sits, i.e. push force: a
        drawer that stalls at 5 cm strokes goes through at 10 cm (measured 2026-08-27), so
        a stalled stroke doubles it (capped at max(20 cm, stroke)) and re-seats; if that
        still stalls it takes the next steeper wrist pitch, and when nothing is left to
        escalate it stops instead of burning the horizon.  The escalation was measured on
        ONE drawer (the LIBERO white cabinet's bottom level): there a level wrist stalls
        1 mm short of ``qpos > 0`` while 15 deg carries it into the stop — but 60 deg does
        NOT (the steep stub pushes down more than along; libero.10_3 lost 5 of 11 runs
        that way), and pushing the HANDLE BAR at any stroke never crossed zero on the
        same drawer in another scene (skills/knowledge/shared/panda/
        libero-goal-predicates-and-drawer-closing.md).  Pass ``pitches=(15.0,)`` (or
        ``(0.0,)``) to pin the wrist when you know what works; the default ladder is
        :data:`PUSH_PITCHES` from :meth:`reachable_tool_quat`, i.e. the flattest pitch the
        seedless IK probe accepts for the approach and contact points.
        Verified in sim on cabinet drawers and on a microwave door (libero.90_33: closed
        in 4 strokes, reward 1.0).  ``close_dir`` sets the direction the closed fingers
        lie along (default vertical, the certified behaviour): for a face LOW over the
        table pass a horizontal direction perpendicular to the push — with vertical
        fingers the hand body spans 10 cm below the fingertips and stands on the table at
        a bottom drawer's handle height (z 0.03: measured 2026-08-27, three level strokes
        stalled at -0.017 on libero.90_22), with horizontal fingers it is 2 cm tall there.
        The fingers stay on the face between strokes (each stroke simply commands a target
        further along the joint): backing off and re-approaching every stroke hooks the
        closed fingertips behind the handle bar and drags the part open again.  A stroke
        costs ~250 steps; strokes stop at the horizon (``budget_exhausted``).  The joint is
        read AGAIN after the final ``park()`` retreat — the closed fingertips can hook the
        bar back out on the way (0.5-0.9 cm measured) — and ``closed``/``qpos`` are the
        post-retreat values the scorer will see; ``qpos_before_park`` is the other one.
        ``park_after=False`` skips that retreat and leaves the hand PRESSING on the face —
        the robust final state for a task that ends with the close: a drawer pressed to
        +0.0017 relaxed to -0.0065 on the retreat in one process and stayed at +0.0016 in
        another (libero.90_23 / 90_22, same code and seed, 2026-08-27), so a margin of a
        millimetre is a coin flip once the hand lets go.  ``close_dir`` also takes
        ``"horizontal"`` / ``"vertical"`` like ``push_object``; a direction parallel to the
        push raises, and the direction actually used is reported (``finger_dir``).  With
        horizontal fingers the hand is ~20 cm WIDE along the face — check the face is that
        wide (a narrow door beside a wall).  ``parked`` is True only when the retreat
        really happened (a ``park()`` refused for budget leaves the hand on the face).
        For a hinge door the push tangent turns per stroke and the finger direction is
        re-projected each stroke.  Returns ``{"closed", "qpos", "qpos_before_park",
        "start_qpos", "strokes", "stalls", "reach_m", "reopened_m", "pitch_deg",
        "reachable", "parked", "finger_dir", "budget_exhausted"}``.
        """
        info = self._joint_info(joint_name)
        if info["type"] not in (MJ_SLIDE, MJ_HINGE):
            raise ValueError(f"{joint_name} is not a slide or hinge joint")
        lo, hi = (float(v) for v in info["range"])
        start = info["qpos"]
        goal = float(np.clip(target_qpos, lo, hi))
        sign = 1.0 if goal >= start else -1.0  # sign of the CLOSING motion in joint space
        axis = info["axis_world"] / max(np.linalg.norm(info["axis_world"]), 1e-9)
        ladder = tuple(float(p) for p in (PUSH_PITCHES if pitches is None else pitches))
        if not ladder:
            raise ValueError("pitches must not be empty")
        finger_pref = "vertical" if close_dir is None else close_dir
        finger_used = np.array([0.0, 0.0, 1.0])

        def qpos() -> float:
            return float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])

        def push_dir_world() -> np.ndarray:
            if info["type"] == MJ_SLIDE:
                return axis * sign
            face = self._handle_geom(info["body_id"], np.array([0.0, 0.0, 1.0]))
            rel = face["center_world"] - info["anchor_world"]
            radial = rel - axis * (rel @ axis)
            return np.cross(axis, radial / max(np.linalg.norm(radial), 1e-9)) * sign

        self.close_fingers()
        self.settle(3)
        strokes = 0
        stalls = 0
        reopened = 0.0
        pitch_deg: float | None = None
        quat = tool_pitched_quat_wxyz([1.0, 0.0, 0.0])
        reach = float(stroke)  # how far past the face the controller target sits
        reach_cap = max(0.20, float(stroke))
        pitch_floor = ladder[0]  # raised only when a longer reach still stalls
        reachable: bool | None = None
        seat = True  # the first stroke has to reach the face from outside
        while strokes < int(max_strokes) and sign * (qpos() - goal) < 0:
            if self._steps_left() < 260:  # a stroke costs ~250 steps: do not start one
                break
            push = push_dir_world()
            face = self._handle_geom(info["body_id"], -push)  # the face we push on
            contact = self.base_from_world(face["center_world"])
            push_base = self.base_from_world(info["anchor_world"] + push) - self.base_from_world(
                info["anchor_world"]
            )
            push_base = push_base / max(np.linalg.norm(push_base), 1e-9)
            # the fingers lie along finger_dir; a direction parallel to the push would be
            # replaced silently by the frame helper (vertical fingers for a +-y push, NaN
            # for +-x) — refuse it instead
            finger_dir = self._finger_dir(finger_pref, push_base)
            perp = finger_dir - push_base * float(finger_dir @ push_base)
            if float(np.linalg.norm(perp)) < 0.1 * max(float(np.linalg.norm(finger_dir)), 1e-9):
                raise ValueError("close_dir must not be parallel to the push direction")
            finger_used = perp / np.linalg.norm(perp)
            approach = contact - push_base * standoff_m
            press = contact + push_base * reach
            before = qpos()
            if seat:
                # probe the approach and the CONTACT (+2 cm), not the force target: a
                # press point 20 cm inside the cabinet is never meant to be reached and
                # rejecting flatter pitches for it is what drove the wrist to 60 deg
                pose = self.reachable_tool_quat(
                    [approach, contact + push_base * 0.02],
                    push_base,
                    finger_dir,
                    pitches=tuple(p for p in ladder if p >= pitch_floor) or (ladder[-1],),
                )
                pitch_deg = pose["pitch_deg"]
                reachable = bool(pose["reachable"])
                quat = np.asarray(pose["quat_wxyz"], dtype=float)
                # unhook straight up before backing off: closed fingertips that have gone
                # past a handle bar drag the part open again on a straight retreat (3.8 cm
                # per stroke, measured on the white cabinet's bottom drawer)
                self.move_by_displacement(z=0.06, iterations=2, max_iterations=2)
                self.goto(approach, quat)
                reopened = max(reopened, float(sign * (before - qpos())))
                before = qpos()
                seat = False
            self.goto(press, quat)
            strokes += 1
            # "stalled" has to be judged against what is LEFT to travel: within a
            # millimetre of the stop every stroke moves less than 5 mm, and a fixed
            # threshold there reads real progress as a stall and escalates the push and
            # the wrist pitch until the horizon is gone (measured: 8 strokes to -0.0015
            # against 5 strokes to +0.0007 without the escalation)
            if abs(qpos() - before) < min(0.005, 0.5 * abs(goal - before)):
                stalls += 1
                seat = True  # re-seat on the (moved) face before giving up
                # escalate the reach first (the target sits further past the face:
                # 5 -> 10 -> 20 cm carries a drawer through the last centimetre into
                # its stop), the pitch only when a longer reach still stalls — and stop
                # as soon as there is nothing left to escalate (three zero-motion cycles
                # at 20 cm / 60 deg cost 1650 steps, measured)
                used = pitch_deg if pitch_deg is not None else ladder[0]
                if reach < reach_cap:
                    reach = min(reach * 2.0, reach_cap)
                else:
                    steeper = next((p for p in ladder if p > used), None)
                    if steeper is None:
                        break
                    pitch_floor = steeper
                if stalls >= 3:  # three escalating strokes that moved nothing: stuck
                    break
            else:
                stalls = 0
        cut = self._steps_left() < 260
        q_before_park = qpos()
        parked = False
        if park_after:
            park_rep = self.park()
            parked = bool(
                park_rep
                and not park_rep.get("budget_exhausted")
                and float(park_rep.get("retreated_m", 0.0) or 0.0) > 0.0
            )
        final = qpos()
        return {
            "closed": bool(sign * (final - goal) > 0),
            "parked": parked,
            "finger_dir": [round(float(v), 3) for v in finger_used] if strokes else None,
            "qpos": round(final, 4),
            "qpos_before_park": round(q_before_park, 4),
            "start_qpos": round(start, 4),
            "strokes": strokes,
            "stalls": stalls,
            "reach_m": round(reach, 4),
            "reopened_m": round(max(reopened, float(sign * (q_before_park - final))), 4),
            "pitch_deg": pitch_deg,
            "reachable": reachable,
            "budget_exhausted": bool(cut or self._steps_left() < 260),
        }

    def _joint_info(self, joint_name: str) -> dict[str, Any]:
        raw_m, raw_d = self._raw()
        sim = self.sim
        jid = sim.model.joint_name2id(joint_name)
        return {
            "id": jid,
            "type": int(raw_m.jnt_type[jid]),
            "range": np.asarray(raw_m.jnt_range[jid], dtype=float),
            "axis_world": np.asarray(raw_d.xaxis[jid], dtype=float),
            "anchor_world": np.asarray(raw_d.xanchor[jid], dtype=float),
            "body_id": int(raw_m.jnt_bodyid[jid]),
            "qpos": float(np.atleast_1d(sim.data.get_joint_qpos(joint_name))[0]),
        }

    def _handle_geom(self, body_id: int, move_dir_world: np.ndarray) -> dict[str, Any]:
        """The collision geom of ``body_id`` that protrudes farthest along ``move_dir``:
        a drawer/door handle. Returns its world centre and its long axis (world).
        """
        raw_m, raw_d = self._raw()
        best = None
        for g in range(raw_m.ngeom):
            # collision geoms carry contype OR conaffinity (LIBERO's microwave door has
            # contype 0 / conaffinity 1 on every geom)
            if raw_m.geom_bodyid[g] != body_id or not (
                raw_m.geom_contype[g] or raw_m.geom_conaffinity[g]
            ):
                continue
            pos = np.asarray(raw_d.geom_xpos[g], dtype=float)
            proj = float(pos @ move_dir_world)
            if best is None or proj > best[0]:
                best = (proj, g, pos)
        if best is None:
            raise RuntimeError("the articulated body has no collision geoms to grasp")
        _, g, pos = best
        size = np.asarray(raw_m.geom_size[g], dtype=float)
        xmat = np.asarray(raw_d.geom_xmat[g], dtype=float).reshape(3, 3)
        long_axis = xmat[:, int(np.argmax(size))]
        return {"geom": g, "center_world": pos, "long_axis_world": long_axis, "size": size}

    def open_articulated(
        self,
        joint_name: str,
        target_fraction: float = 0.6,
        direction: int | None = None,
        stroke: float = 0.03,
        max_strokes: int = 12,
        pitches: Any = None,
    ) -> dict[str, Any]:
        """Open a drawer (slide joint) or door (hinge joint) to ``target_fraction`` of
        its range by grasping its handle and pulling along the joint axis in short,
        verified strokes; re-grasps when the handle slips.

        The grasping wrist is pitched down as far as the IK needs
        (:meth:`reachable_tool_quat`): a level wrist cannot be posed at the height of a
        bottom drawer's handle at all, and the report says which pitch was used.

        ``direction`` (+1/-1) overrides the guess that "open" is the range end farther
        from the current position.  ``pitches`` pins the wrist-pitch ladder
        (``(0.0,)`` = level): the seedless IK probe rejects a level wrist at some low
        heights that the closed loop reaches (a solver local minimum), and a pitched
        pull on a VERTICAL bar (a microwave handle) pushes the pads down the bar and
        slips — 8 strokes / 3 regrasps for 15 deg at pitch 45 (libero.90_35, measured
        2026-08-27).  ``reachable`` says whether the seedless probe accepted the pitch of
        the LAST grasp or fell back to the least-bad one (each grasp's pitch and probe
        verdict are in ``trace``).  A stroke that moves neither the part nor the HAND
        (``hand_moved_m`` in the trace) is the arm at its reach limit, not a slip: the
        re-grasp then goes through :meth:`park` first (``rehooks``) so the next approach
        starts from a fresh elbow branch.  A hinge door that the arm cannot swing past
        ~45-55 deg by pulling (the microwave: the handle runs out to x 0.77 at -0.97 rad)
        is finished by PUSHING its inner face with :meth:`close_articulated`
        ``(target_qpos=<open value>)`` — a push away from the robot is the reach-friendly
        stroke; libero.90_35 opens 0 -> -0.63 in three level strokes here and -0.63 ->
        -1.48 in that push (measured 2026-08-27).  Returns ``{"opened", "qpos",
        "fraction", "strokes", "regrasps", "rehooks", "pitch_deg", "reachable",
        "target_qpos", "trace"}``.
        """
        info = self._joint_info(joint_name)
        lo, hi = info["range"]
        span = float(hi - lo)
        if span <= 0:
            raise RuntimeError(f"{joint_name} has no range to open")
        q0 = info["qpos"]
        sign = direction if direction in (1, -1) else (1 if abs(hi - q0) > abs(q0 - lo) else -1)
        open_end = hi if sign > 0 else lo
        requested_fraction = float(target_fraction)
        if requested_fraction < 0.0:
            raise ValueError("target_fraction must be non-negative")
        required_fraction = min(requested_fraction, abs(open_end - q0) / span)
        target_q = q0 + sign * required_fraction * span

        def fraction() -> float:
            q = float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])
            return directed_joint_progress(q, q0, sign, span), q

        axis = info["axis_world"] / max(np.linalg.norm(info["axis_world"]), 1e-9)
        if info["type"] == MJ_SLIDE:
            move_world = axis * sign
        elif info["type"] == MJ_HINGE:
            move_world = None  # tangential, computed per stroke
        else:
            raise RuntimeError(f"{joint_name}: only slide and hinge joints are supported")

        def handle_point() -> tuple[np.ndarray, np.ndarray]:
            probe_dir = move_world if move_world is not None else np.array([0.0, 0.0, 1.0])
            if info["type"] == MJ_HINGE:
                # the handle is the geom farthest from the hinge axis
                raw_m, raw_d = self._raw()
                best = None
                for g in range(raw_m.ngeom):
                    if raw_m.geom_bodyid[g] != info["body_id"] or not (
                        raw_m.geom_contype[g] or raw_m.geom_conaffinity[g]
                    ):
                        continue
                    pos = np.asarray(raw_d.geom_xpos[g], dtype=float)
                    rel = pos - info["anchor_world"]
                    radial = rel - axis * (rel @ axis)
                    if best is None or np.linalg.norm(radial) > best[0]:
                        best = (np.linalg.norm(radial), pos, radial)
                if best is None:
                    raise RuntimeError("no handle geom found on the hinged body")
                return best[1], best[2]
            h = self._handle_geom(info["body_id"], probe_dir)
            return h["center_world"], h["long_axis_world"]

        strokes = 0
        regrasps = 0
        rehooks = 0
        pitch_deg: float | None = None
        reachable: bool | None = None
        trace: list[dict[str, Any]] = []
        frac, q = fraction()
        while frac < required_fraction and strokes < max_strokes:
            center_w, aux = handle_point()
            if info["type"] == MJ_SLIDE:
                bar = aux
            else:
                radial = aux / max(np.linalg.norm(aux), 1e-9)
                move_world = np.cross(axis, radial) * sign
                bar = axis  # door handles run parallel to the hinge axis
            # approach from the open side: tool axis points at the part, fingers close
            # across the bar (a horizontal bar -> vertical pinch)
            point_dir = -move_world
            close_dir = np.cross(point_dir, bar)
            if np.linalg.norm(close_dir) < 1e-6:
                close_dir = np.array([0.0, 0.0, 1.0])
            origin = self.base_from_world(center_w)
            ahead = self.base_from_world(center_w + move_world * 0.10) - origin
            pose = self.reachable_tool_quat(
                [origin, origin + ahead],
                point_dir,
                close_dir,
                pitches=PUSH_PITCHES if pitches is None else pitches,
            )
            quat = np.asarray(pose["quat_wxyz"], dtype=float)
            pitch_deg = pose["pitch_deg"]
            reachable = bool(pose["reachable"])
            self.open_fingers()
            self.goto(origin + ahead, quat)
            self.goto(origin, quat)
            self.close_fingers()
            opening = self.get_gripper_opening()
            trace.append(
                {
                    "grasp_opening": round(opening, 4),
                    "tcp": np.round(self.tcp(), 3).tolist(),
                    "pitch_deg": pitch_deg,
                    "reachable": reachable,
                }
            )
            slipped = False
            arm_stalled = False
            while frac < required_fraction and strokes < max_strokes:
                before = q
                tcp_before = self.tcp()
                step_base = self.base_from_world(center_w + move_world * stroke) - origin
                self.move_by_displacement(*step_base, iterations=3, max_iterations=3)
                strokes += 1
                frac, q = fraction()
                hand_moved = float(np.linalg.norm(self.tcp() - tcp_before))
                trace.append(
                    {
                        "stroke": strokes,
                        "dq": round(q - before, 4),
                        "tcp": np.round(self.tcp(), 3).tolist(),
                        "hand_moved_m": round(hand_moved, 4),
                    }
                )
                if sign * (q - before) < 0.25 * stroke:
                    slipped = True
                    # the part did not move AND the hand did not either: the arm is at its
                    # reach limit in this posture (not a slipped pinch) — re-grasping from
                    # the same posture would stall again
                    arm_stalled = hand_moved < 0.3 * stroke
                    break
                center_w, _ = handle_point()  # the handle moved with the part
                origin = self.base_from_world(center_w)
            if frac >= required_fraction or not slipped:
                break
            regrasps += 1
            self.open_fingers()
            if arm_stalled:
                # re-hook: park (a fresh elbow branch for the next approach) instead of
                # backing off along the pull, which the stalled arm cannot do either
                rehooks += 1
                trace.append({"rehook": rehooks, "park": self.park()})
            else:
                self.move_by_displacement(
                    *(
                        self.base_from_world(center_w + move_world * 0.08)
                        - self.base_from_world(center_w)
                    )
                )
            if regrasps >= 3:
                break
        self.open_fingers()
        self.settle(5)
        if move_world is not None:
            self.move_by_displacement(
                *(self.base_from_world(move_world * 0.08) - self.base_from_world([0, 0, 0]))
            )
        frac, q = fraction()
        return {
            "opened": frac >= required_fraction,
            "qpos": q,
            "fraction": round(frac, 3),
            "strokes": strokes,
            "regrasps": regrasps,
            "rehooks": rehooks,
            "pitch_deg": pitch_deg,
            "reachable": reachable,
            "target_qpos": float(target_q),
            "requested_fraction": requested_fraction,
            "required_fraction": round(required_fraction, 6),
            "trace": trace,
        }

    # ------------------------------------------------- hand pose and verified pinches
    # Promoted 2026-08-27 from the helpers 34 certified libero-90 solutions re-implemented
    # in their preludes (rim_pick, box_pick, top_pick, grasp_at, lift_check, ee/tcp, put_on,
    # travel_z, recentre_on, push_close, press_face, _outward/drawer_axis).

    def get_hand_pose(self) -> dict[str, Any]:
        """Where the hand is, in the robot-base frame: ``{"wrist", "quat_wxyz", "tcp",
        "fingertips"}``.

        ``tcp`` is the API's TOOL POINT (``wrist + R(quat) @ TCP_OFFSET_HAND``, 10 cm
        along the tool axis): the point ``goto_pose``, ``goto_pose_closed`` and
        ``move_by_displacement`` aim at and the point every height in this API refers to.
        It is NOT the end of the fingers: the finger tips extend ``FINGERTIP_BELOW_TCP``
        (0.016 m) past it along the tool axis (measured 2026-08-27: the tip bodies +0.018,
        robosuite's grip site +0.007), so the lowest safe ``tcp`` over a surface is the
        surface + 0.016 — ``fingertips`` is that point.  ``wrist`` is what
        ``get_observation()["robot_cartesian_pos"][:3]`` reports: the cap-x flange frame
        (``gripper0_eef`` pulled back 0.107 m along the tool and yawed 90 deg), 1 cm behind
        the MuJoCo hand body — not the ``panda_hand`` body itself.  Mixing ``wrist`` and
        ``tcp`` puts every absolute height 10 cm out.  ``quat_wxyz`` is the hand
        orientation; ``np.asarray(quat_wxyz_to_mat(quat))[:, 2]`` is the tool axis and
        ``[:, 1]`` the finger-closing axis.
        """
        pos, quat = self.hand_pose()
        rot = _quat_wxyz_to_mat(quat)
        tcp = self.tcp()
        return {
            "wrist": [float(v) for v in pos],
            "quat_wxyz": [float(v) for v in quat],
            "tcp": [float(v) for v in tcp],
            "fingertips": [float(v) for v in tcp + rot @ np.array([0.0, 0.0, FINGERTIP_BELOW_TCP])],
        }

    #: a lift that raises the object by less than this fraction of the hand's rise is a slip
    HOLD_FRACTION = 0.6

    def lift_check(self, object_name: str, lift_m: float = 0.15) -> dict[str, Any]:
        """Prove a grasp physically: raise the fingertips ``lift_m`` and report how far the
        object rose with them.

        The finger gap is not the test — a bowl held by its 5 mm rim reads 0.004-0.005 m,
        a moka pot wedged between OPEN fingers reads 0.078 m and still rises — so every
        verified pinch here judges ``rise_m`` against the hand's own rise: ``held`` is
        ``rise_m > 0.6 * hand_rise_m`` with the hand having risen at least
        ``max(2 cm, half of lift_m)``, so ``lift_m`` must be at least 0.03 (a 1 cm lift
        with perfect tracking used to read "not held").  ``reason`` says WHY a check
        failed: ``"hand_did_not_rise"`` (blocked under a shelf, an IK stall — the object
        may well be held, do not open the fingers on it), ``"object_stayed"`` (a real
        slip), ``"partial_slip"``.  ``object_z`` is the BODY ORIGIN (a bottle's base, a
        ketchup's centre) and ``object_name`` resolves like ``body_position`` (a bare type
        name is the ``_1`` instance).  The hand stays up afterwards (the object is at carry
        height); a failed check leaves the gripper closed for the caller to open.  Cost
        ~30-100 steps (up to ~250 when fighting a contact; measured 21-42 for 1-5 cm, 96
        for 15 cm).  Returns ``{"rise_m", "hand_rise_m", "requested_m", "held", "reason",
        "opening_m", "object_z", "budget_exhausted"}``.
        """
        lift = float(lift_m)
        if not np.isfinite(lift) or lift < 0.03:
            raise ValueError(f"lift_m must be a finite lift of at least 0.03 m, got {lift_m!r}")
        before = self.body_position(object_name)
        tcp0 = self.tcp()
        left0 = self._steps_left()
        self.move_by_displacement(z=lift)
        after = self.body_position(object_name)
        hand_rise = float(self.tcp()[2] - tcp0[2])
        rise = float(after[2] - before[2])
        floor = max(0.02, 0.5 * lift)
        held = bool(hand_rise > floor and rise > self.HOLD_FRACTION * hand_rise)
        if held:
            reason = ""
        elif hand_rise <= floor:
            reason = "hand_did_not_rise"
        elif rise <= 0.005:
            reason = "object_stayed"
        else:
            reason = "partial_slip"
        return {
            "rise_m": round(rise, 4),
            "hand_rise_m": round(hand_rise, 4),
            "requested_m": lift,
            "held": held,
            "reason": reason,
            "opening_m": round(self.get_gripper_opening(), 4),
            "object_z": float(after[2]),
            "budget_exhausted": self._steps_left() <= 1 or (left0 - self._steps_left()) < 1,
        }

    #: a jaw gap at or below this after a close means the fingers met on nothing
    EMPTY_OPENING = 0.002

    def grasp_at(
        self,
        point: Any,
        quat_wxyz: Any,
        approach_m: float = 0.12,
        *,
        object_name: str | None = None,
        lift_m: float = 0.0,
        max_descent_error_m: float | None = None,
        preopen_m: float | None = None,
    ) -> dict[str, Any]:
        """Pinch at a measured base-frame point with hand orientation ``quat_wxyz``: open,
        hover ``approach_m`` above the point, descend straight down, close.

        The universal "pinch here" step behind ``rim_pick``/``box_pick``/``top_pick``, for
        when you have measured the point yourself (``get_object_extents``, ``plan_pinch``).
        Both legs are closed-loop (``goto``); when the hand starts BELOW the hover it rises
        to the hover height first (from above it goes straight to the hover), and
        ``approach_error_m``/``grasp_error_m`` report how close the hand got.  The descent
        (:meth:`_descend_to`) goes down in 5 cm legs and treats a leg that drifts more than
        1 cm sideways, or hardly moves, as a stall: it lifts 3 cm and re-seeds the IK once
        (``reseeded``), then gives up (``descent_reason`` "stalled"); the drift is reported
        (``xy_drift_m``) — a descent that sweeps sideways knocks a tall object over (the
        wine bottle of libero.90_26: 7.6 cm of drift, measured 2026-08-27).  With
        ``max_descent_error_m`` the jaws are NOT closed when the descent ends farther than
        that from the point (a finger on a lid, a wall): ``closed_on_material`` is then
        False and ``descent_reason`` "stalled".  The hover is ``approach_m`` above the
        PINCH POINT, not above the object: on a tall object pinched low the fingertips are
        already below its top at the hover.  With ``object_name`` and ``lift_m`` > 0
        (>= 0.03) the grasp is verified by :meth:`lift_check`; ``held``/``rise_m`` are
        None otherwise (do not read ``not held`` as a failure then).  A close that reads
        empty (gap <= 2 mm) skips the lift and reports ``held`` False.  ``preopen_m``
        opens the jaws only that far instead of fully (:meth:`open_fingers`): the open jaw
        spans ``gap + 0.053`` m across its closing axis and it is that envelope, not the
        pinch point, that a narrow slot rejects (libero.90_6).  With
        ``object_name`` the object's origin is compared before and after the descent
        (``disturbed_m``): a knocked object is visible even when the jaws then close on
        the right point.  ``closed_on_material`` is the finger gap after closing (> 4 mm);
        trust ``held`` over it.  A failed lift leaves the gripper CLOSED for the caller.
        Refused outright (``reason``) with fewer than ``PICK_MIN_STEPS`` steps left; a
        descent cut by the horizon sets ``budget_exhausted``.  Cost ~200-450 steps, +30-100
        for the lift.  Returns ``{"opening_m", "closed_on_material", "approach_error_m",
        "grasp_error_m", "descent_reason", "xy_drift_m", "reseeded", "disturbed_m",
        "held", "rise_m", "tcp", "reason", "preopen_m", "budget_exhausted"}``.
        """
        target = np.asarray(point, dtype=float).reshape(3)
        quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
        hover = target + np.array([0.0, 0.0, float(approach_m)])
        report: dict[str, Any] = {
            "opening_m": round(float(self.get_gripper_opening()), 4),
            "closed_on_material": False,
            "approach_error_m": None,
            "grasp_error_m": None,
            "descent_reason": "",
            "xy_drift_m": 0.0,
            "reseeded": False,
            "disturbed_m": None,
            "held": None,
            "rise_m": None,
            "tcp": [round(float(v), 4) for v in self.tcp()],
            "reason": "",
            "preopen_m": None if preopen_m is None else round(float(preopen_m), 4),
            "budget_exhausted": self._steps_left() <= 1,
        }
        if self._steps_left() < self.PICK_MIN_STEPS:
            report["reason"] = f"fewer than {self.PICK_MIN_STEPS} steps left: grasp refused"
            report["budget_exhausted"] = True
            return report
        before_obj = self.body_position(object_name) if object_name is not None else None
        self.open_fingers(preopen_m)
        report["preopen_m"] = None if preopen_m is None else round(float(preopen_m), 4)
        here = self.tcp()
        if float(np.linalg.norm(here[:2] - target[:2])) > 0.03 and here[2] < hover[2] - 0.01:
            self.goto(np.array([here[0], here[1], hover[2]]), quat, iterations=2, max_iterations=4)
        approach_err = self.goto(hover, quat)
        grasp_err, descent = self._descend_to(target, quat)
        report.update(
            {
                "approach_error_m": round(float(approach_err), 4),
                "grasp_error_m": round(float(grasp_err), 4),
                "descent_reason": descent["reason"],
                "xy_drift_m": descent["xy_drift_m"],
                "reseeded": descent["reseeded"],
            }
        )
        if before_obj is not None:
            after_obj = self.body_position(object_name)
            report["disturbed_m"] = round(float(np.linalg.norm(after_obj - before_obj)), 4)
        if max_descent_error_m is not None and grasp_err > float(max_descent_error_m):
            report["descent_reason"] = "stalled"
            report["reason"] = (
                f"descent ended {grasp_err:.3f} m from the point (> {max_descent_error_m}): "
                "jaws not closed"
            )
            report["held"] = False if object_name is not None and float(lift_m) > 0.0 else None
            report["tcp"] = [round(float(v), 4) for v in self.tcp()]
            report["budget_exhausted"] = self._steps_left() <= 1 or descent["reason"] == "budget"
            return report
        closed = self.close_fingers()  # refused at the horizon: the gap then reads fully open
        self.settle(3)
        opening = self.get_gripper_opening()
        report.update(
            {
                "opening_m": round(opening, 4),
                "closed_on_material": bool(closed and opening > HELD_OPENING),
                "tcp": [round(float(v), 4) for v in self.tcp()],
                "budget_exhausted": self._steps_left() <= 1 or descent["reason"] == "budget",
            }
        )
        if object_name is not None and float(lift_m) > 0.0:
            if not closed or opening <= self.EMPTY_OPENING:
                # nothing between the pads: the lift would only prove what the gap says
                report.update({"held": False, "rise_m": 0.0, "reason": "closed on nothing"})
                return report
            lift = self.lift_check(object_name, float(lift_m))
            report.update(
                {
                    "held": lift["held"],
                    "rise_m": lift["rise_m"],
                    "reason": lift["reason"],
                    "tcp": [round(float(v), 4) for v in self.tcp()],
                    "budget_exhausted": bool(
                        report["budget_exhausted"] or lift["budget_exhausted"]
                    ),
                }
            )
        return report

    #: a grasp attempt (open, hover, descend, close, lift) costs about this many steps;
    #: below it a pick primitive stops trying and reports ``budget_exhausted``
    PICK_MIN_STEPS = 450

    #: a joint-space jump larger than this on any joint means the seedless IK found a
    #: different branch (a wrist flip): keep the seeded posture instead
    RESEED_MAX_JUMP_RAD = 1.2
    #: a descent leg that ends farther than this from the goal xy is a sideways sweep
    DESCENT_MAX_XY_DRIFT = 0.01

    def _descend_to(
        self, target: Any, quat: Any, *, step_m: float = 0.05
    ) -> tuple[float, dict[str, Any]]:
        """Lower the fingertips straight down onto ``target`` in ``step_m`` legs and return
        ``(final position error, {"reason", "xy_drift_m", "reseeded", "legs"})``.

        One long closed-loop ``goto`` from a hover 15 cm up ends 3-8 cm short at the far
        side of the table (x 0.72: the velocity-cost IK creeps and the loop gives up while
        the error still shrinks — measured 2026-08-27 on the SCENE3 moka pot, 4.0 cm short
        and 11 deg off, and 8.2 cm on the SCENE4 wine bottle) where the same descent in
        5 cm legs converges.  A leg counts as dead when it moves less than 40 % of its
        length OR ends more than ``DESCENT_MAX_XY_DRIFT`` sideways of the goal (a leg
        accepted at 60 % of its travel with 2.8 cm of xy error swept the open finger
        through the wine bottle of libero.90_26 and knocked it over): the first dead leg
        lifts 3 cm and re-seeds the IK, taking the seedless solution only when no joint
        jumps more than ``RESEED_MAX_JUMP_RAD`` (a wrist-flipped branch would drive
        through the scene; the seeded posture is kept otherwise); two dead legs in a row
        end the descent (``reason`` "stalled"; "budget" when the horizon cut it, "reached"
        otherwise).
        """
        goal = np.asarray(target, dtype=float).reshape(3)
        q = np.asarray(quat, dtype=float).reshape(4)
        dead = 0
        retried = False
        reseeded = False
        legs = 0
        drift = 0.0
        reason = "reached"
        # bounded: a descent that cannot converge must not eat the horizon (one did: a
        # neck pinch whose palm sat on the bottle's neck top burned 3500 steps)
        max_legs = int(np.ceil(max(self.tcp()[2] - goal[2], 0.0) / float(step_m))) + 3
        while self.tcp()[2] > goal[2] + 0.004:
            if self._steps_left() <= 300:
                reason = "budget"
                break
            if legs >= max_legs:
                reason = "stalled"
                break
            legs += 1
            here = self.tcp()
            nxt = max(float(goal[2]), float(here[2]) - float(step_m))
            leg = float(here[2]) - nxt
            self.goto(np.array([goal[0], goal[1], nxt]), q, iterations=2, max_iterations=4)
            now = self.tcp()
            moved = float(here[2] - now[2])
            xy_err = float(np.linalg.norm(now[:2] - goal[:2]))
            drift = max(drift, xy_err)
            if moved < 0.4 * leg or xy_err > self.DESCENT_MAX_XY_DRIFT:
                if not retried:
                    # the velocity-cost solve is stuck in a posture that cannot lower the
                    # hand: lift a little and let the next solve start seedless (pyroki's
                    # from-zero solve), which picks the branch that reaches the point —
                    # unless that branch is a wrist flip away from the current posture
                    retried = True
                    self.goto(self.tcp() + np.array([0.0, 0.0, 0.03]), q, iterations=2)
                    if self._reseed_is_safe(np.array([goal[0], goal[1], nxt]), q):
                        self.api.cfg = None
                        reseeded = True
                    continue
                dead += 1
                if dead >= 2:
                    reason = "stalled"
                    break
            else:
                dead = 0
                retried = False
        err = float(np.linalg.norm(goal - self.tcp()))
        if reason == "reached" and err > 0.02:
            reason = "stalled"
        return err, {
            "reason": reason,
            "xy_drift_m": round(drift, 4),
            "reseeded": reseeded,
            "legs": legs,
        }

    def _reseed_is_safe(self, position: np.ndarray, quat: np.ndarray) -> bool:
        """Whether a seedless IK solve for ``position``/``quat`` stays on the current
        joint-space branch (no joint jumps more than ``RESEED_MAX_JUMP_RAD``).
        """
        api = self.api
        prev = api.cfg
        if prev is None:
            return True
        hand_target = np.asarray(position, dtype=float) + _quat_wxyz_to_mat(quat) @ (
            -TCP_OFFSET_HAND
        )
        try:
            fresh = api._pks.solve_ik(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_wxyz=np.asarray(quat, dtype=float),
                target_position=hand_target,
            )
        except Exception:
            return False
        jump = np.abs(np.asarray(fresh[:7], dtype=float) - np.asarray(prev[:7], dtype=float))
        return bool(np.max(jump) <= self.RESEED_MAX_JUMP_RAD)

    def rim_pick(
        self,
        object_name: str,
        depth_m: float = 0.012,
        approach_m: float = 0.12,
        *,
        radial_offsets: Any = (0.0, 0.006, -0.008, 0.014),
        sides: Any = ("-x", "+x", "-y", "+y"),
        lift_m: float = 0.15,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Top-down pinch of a bowl-, cup- or plate-like object's RIM WALL, measured from
        its collision box, verified by lifting.

        A bowl is wider than the 7.7 cm jaw, so a pinch at its centre closes on air; the
        fingers must straddle the wall — one inside the cavity, one outside — ``depth_m``
        below the rim (``top_z``).  The wall is placed at ``centre +- (half of the smaller
        horizontal extent + offset)`` along the chosen side, with the fingers closing along
        that axis — so for a HANDLED mug the box centre is shifted toward the handle and
        the ``+-y`` points sit off the cavity: try the axis perpendicular to the longer
        extent first.  Each ``radial_offsets`` value (0 first: the box edge; +6 mm
        outward; -8 mm inward for a flared rim; +14 mm) is tried on each of ``sides`` in
        order (``"-x"`` = the wall nearest the robot, then ``"+x"``, ``"-y"``, ``"+y"``);
        a side whose point the IK probe rejects is skipped (the probed point is reported),
        a pinch that closes on nothing (gap <= 2 mm: no lift is spent on it) or does not
        lift the object (:meth:`lift_check`) is released and the next is tried — set back
        down first when it rose partway.  ``depth_m`` is capped like ``box_pick``'s so the
        finger tips (16 mm past the tool point) stay off the support on a thin plate.  The
        object's box is re-measured before every attempt (a failed pinch nudges it).
        A ONE-SIDED rim pinch tilts a flat object by rim-thickness/diameter: a 1.9 cm
        plate hung vertical, a bowl 17 deg, a mug 9 deg (measured 2026-08-27) — the
        object's tilt is reported (``tilt_deg``) and a pinch that leaves it more than 45
        deg over is not counted as held.  Cost ~400-700 steps per attempt (a descent that
        stalls on the support ~1150); the primitive stops when fewer than ~450 steps are
        left (``reason`` "budget").  The hand ends ``lift_m`` above the grasp with the
        object hanging: the bowl's centre sits ~half a bowl from the fingertips, so carry
        it with ``carry_to(..., object_name=)`` or ``place_*`` (they centre the OBJECT).
        Measured 2026-08-27 (KITCHEN_SCENE1/2/4/5/7/10 bowls, plates, white bowls): the
        first candidate held in most scenes; a candidate that holds in one process can be
        the third in another, so the report says which.

        **Cap the cost when the caller has other work to do.**  Sixteen candidates at
        ~1100 steps each is more than the whole 4000-step horizon, and a run that spends
        it all still reports ``grasped`` True while nothing can be done with the object:
        on libero.90_8's vla-then-code attempt three attempts cost 3380 steps and the
        placement that followed had nothing left (2026-08-28).  ``max_steps`` bounds the
        steps THIS call may spend — no new attempt starts once the spend plus one attempt
        would exceed it (``reason`` "max_steps") — and every report now carries
        ``steps_used`` and ``steps_left``.  A grasp that succeeds with fewer than
        ``PICK_MIN_STEPS`` steps left says so in ``reason`` ("held, but only N steps
        left"): read that before acting on ``grasped``.

        Returns ``{"grasped",
        "grasp_point", "quat_wxyz", "side", "close_dir", "radial_offset_m", "depth_m",
        "opening_m", "rise_m", "tilt_deg", "attempts", "tried", "reason", "steps_used",
        "steps_left", "budget_exhausted"}``.
        """
        axis_of = {"-x": (0, -1.0), "+x": (0, 1.0), "-y": (1, -1.0), "+y": (1, 1.0)}
        order = [str(s) for s in sides]
        for s in order:
            if s not in axis_of:
                raise ValueError(f"sides must be among {sorted(axis_of)}; got {s!r}")
        tried: list[dict[str, Any]] = []
        attempts = 0
        skipped_all = True
        started_with = self._steps_left()

        def spent() -> int:
            return int(started_with - self._steps_left())

        for offset in (float(o) for o in radial_offsets):
            for side in order:
                axis, sign = axis_of[side]
                ext = self.get_object_extents(object_name)
                centre = np.asarray(ext["center"], dtype=float)
                size = np.asarray(ext["size"], dtype=float)
                radius = 0.5 * float(min(size[0], size[1])) + offset
                # the finger tips reach FINGERTIP_BELOW_TCP past the tool point: on a flat
                # plate the pinch depth is capped so they stay off the support
                depth = min(float(depth_m), max(float(size[2]) - FINGERTIP_BELOW_TCP, 0.003))
                point = np.array([centre[0], centre[1], float(ext["top_z"]) - depth])
                point[axis] += sign * radius
                close_dir = np.zeros(3)
                close_dir[axis] = 1.0
                quat = tool_down_quat_wxyz(close_dir)
                probe = self.check_reachable(point, quat)
                if not probe.get("reachable"):
                    tried.append(
                        {
                            "side": side,
                            "offset_m": offset,
                            "point": [round(float(v), 4) for v in point],
                            "skipped": "ik probe",
                        }
                    )
                    continue
                skipped_all = False
                if self._steps_left() < self.PICK_MIN_STEPS:
                    return self._pick_report(
                        tried, attempts, reason="budget", budget=True, steps_used=spent()
                    )
                if max_steps is not None and spent() + self.PICK_MIN_STEPS > int(max_steps):
                    return self._pick_report(
                        tried,
                        attempts,
                        reason=f"max_steps: {spent()} of {int(max_steps)} spent",
                        budget=False,
                        steps_used=spent(),
                    )
                attempts += 1
                up0 = self._object_up_axis(object_name)
                g = self.grasp_at(point, quat, approach_m, object_name=object_name, lift_m=lift_m)
                tilt = float(
                    np.degrees(
                        np.arccos(np.clip(up0 @ self._object_up_axis(object_name), -1.0, 1.0))
                    )
                )
                entry = {
                    "side": side,
                    "offset_m": offset,
                    "depth_m": round(depth, 4),
                    "opening_m": g["opening_m"],
                    "rise_m": g["rise_m"],
                    "grasp_error_m": g["grasp_error_m"],
                    "tilt_deg": round(tilt, 1),
                    "reason": g["reason"],
                }
                tried.append(entry)
                if g["held"] and g["opening_m"] > self.EMPTY_OPENING and tilt <= 45.0:
                    left = self._steps_left()
                    return {
                        "grasped": True,
                        "grasp_point": [round(float(v), 4) for v in point],
                        "quat_wxyz": [float(v) for v in quat],
                        "side": side,
                        "close_dir": [float(v) for v in close_dir],
                        "radial_offset_m": offset,
                        "depth_m": round(depth, 4),
                        "opening_m": g["opening_m"],
                        "rise_m": g["rise_m"],
                        "tilt_deg": round(tilt, 1),
                        "attempts": attempts,
                        "tried": tried,
                        # a hold nothing can be done with is not a clean success: say so
                        "reason": (
                            ""
                            if left >= self.PICK_MIN_STEPS
                            else f"held, but only {left} steps left"
                        ),
                        "steps_used": spent(),
                        "steps_left": int(left),
                        "budget_exhausted": g["budget_exhausted"],
                    }
                if g["held"] and tilt > 45.0:
                    entry["reason"] = f"held but tilted {tilt:.0f} deg"
                # set a partly lifted object back down before letting go: a bowl dragged
                # halfway up and dropped from carry height tumbles
                rise = float(g["rise_m"] or 0.0)
                if rise > 0.005 and self._steps_left() > 100:
                    self.move_by_displacement(z=-rise, iterations=2, max_iterations=3)
                self.open_fingers()
                self.settle(5)
                if g["budget_exhausted"]:
                    return self._pick_report(
                        tried, attempts, reason="budget", budget=True, steps_used=spent()
                    )
        reason = "all sides unreachable" if skipped_all else "no candidate held"
        return self._pick_report(
            tried, attempts, reason=reason, budget=self._steps_left() <= 1, steps_used=spent()
        )

    def _pick_report(
        self,
        tried: list[dict[str, Any]],
        attempts: int,
        *,
        reason: str,
        budget: bool,
        steps_used: int | None = None,
    ) -> dict[str, Any]:
        return {
            "grasped": False,
            "grasp_point": None,
            "quat_wxyz": None,
            "side": None,
            "close_dir": None,
            "radial_offset_m": None,
            "depth_m": tried[-1].get("depth_m") if tried else None,
            "opening_m": tried[-1].get("opening_m") if tried else None,
            "rise_m": tried[-1].get("rise_m") if tried else None,
            "tilt_deg": tried[-1].get("tilt_deg") if tried else None,
            "attempts": attempts,
            "tried": tried,
            "reason": reason,
            "steps_used": steps_used,
            "steps_left": int(self._steps_left()),
            "budget_exhausted": budget,
        }

    def _horizontal_obb(self, object_name: str) -> dict[str, Any]:
        """The object's own horizontal oriented box: ``{"long", "short"`` (base-frame
        unit vectors), ``"long_m", "short_m", "centre_xy", "yaw_deg", "round"}``.

        The principal axes of the collision cloud's xy projection; snapped to the base
        axes when within 5 deg of them (or when the cloud is round/square within 10 %),
        so an axis-aligned box reads exactly as its AABB.
        """
        cloud = self.object_cloud(object_name)
        xy = cloud[:, :2]
        rel = xy - xy.mean(axis=0)
        w, v = np.linalg.eigh(rel.T @ rel / max(len(rel), 1))
        long_u = v[:, 1]
        ratio = float(np.sqrt(max(w[0], 0.0) / max(w[1], 1e-12)))
        is_round = ratio > 0.9
        ang = float(np.degrees(np.arctan2(long_u[1], long_u[0]))) % 180.0
        snapped = is_round
        for base_ang, base in ((0.0, np.array([1.0, 0.0])), (90.0, np.array([0.0, 1.0]))):
            if min(abs(ang - base_ang), 180.0 - abs(ang - base_ang)) < 5.0:
                long_u = base
                snapped = True
        if is_round:
            ext = xy.max(axis=0) - xy.min(axis=0)
            long_u = np.array([1.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0])
        short_u = np.array([-long_u[1], long_u[0]])
        pl, ps = xy @ long_u, xy @ short_u
        centre = long_u * (pl.min() + pl.max()) / 2.0 + short_u * (ps.min() + ps.max()) / 2.0
        return {
            "long": np.array([long_u[0], long_u[1], 0.0]),
            "short": np.array([short_u[0], short_u[1], 0.0]),
            "long_m": float(pl.max() - pl.min()),
            "short_m": float(ps.max() - ps.min()),
            "centre_xy": centre,
            "yaw_deg": 0.0 if snapped else round(ang, 1),
            "round": is_round,
        }

    def box_pick(
        self,
        object_name: str,
        depth_m: float = 0.02,
        approach_m: float = 0.12,
        *,
        attempts: int = 3,
        lift_m: float = 0.14,
        close_dir: Any = None,
    ) -> dict[str, Any]:
        """Top-down pinch of a small box- or can-shaped object across its OWN narrower
        horizontal side, at the centre of its footprint, verified by lifting.

        The plain pick for butter, pudding, cream cheese, cans and cartons whose whole
        body fits the jaw: the fingers close along the short edge of the object's own
        horizontal box (its principal axes, so a yawed box is pinched across its real
        short side and not across a diagonal of its world-axis AABB — a 45-degree cream
        cheese used to be refused as 8.8 cm wide; ``yaw_deg`` is reported) or along
        ``close_dir`` (must have a horizontal component), ``depth_m`` below the top face
        on the first attempt and deeper by ``depth_m`` per retry (never below 80 % of the
        object's height), so a pinch that slipped off a rounded top bites lower next
        time; when the cap leaves the depth unchanged the pinch is shifted 5 mm along the
        long edge instead of being repeated.  The depth is at least 3 mm below the top:
        on an object thinner than ``FINGERTIP_BELOW_TCP`` + 5 mm the finger tips reach the
        support and the descent's 4 mm tolerance is what keeps them off it.  The pinch is
        at the footprint centre with no reachability probe — the staged closed-loop
        descent reaches points ``check_reachable`` refuses (measured at x 0.82).  Each
        attempt is :meth:`grasp_at` (jaws not closed when the descent ends > 1.5 cm short:
        ``skipped``) + :meth:`lift_check`; a failure opens and retries from a re-measured
        box.  Refuses (``reason``) an object whose narrower side does not fit the jaw —
        use :meth:`rim_pick` for bowls and :meth:`top_pick` for tall bottles.  Cost
        ~400-700 steps per attempt.  Returns ``{"grasped", "grasp_point", "quat_wxyz",
        "close_dir", "yaw_deg", "object_side_m", "depth_m", "opening_m", "rise_m",
        "attempts", "tried", "reason", "budget_exhausted"}``.
        """
        if int(attempts) < 1:
            raise ValueError("attempts must be >= 1")
        if close_dir is not None:
            cd = np.asarray(close_dir, dtype=float).reshape(3)
            if np.linalg.norm(cd[:2]) < 1e-6:
                raise ValueError("close_dir must have a horizontal component")
        tried: list[dict[str, Any]] = []
        prev_depth: float | None = None
        for k in range(int(attempts)):
            ext = self.get_object_extents(object_name)
            size = np.asarray(ext["size"], dtype=float)
            obb = self._horizontal_obb(object_name)
            if close_dir is None:
                unit = obb["short"]
                width = float(obb["short_m"])
                long_u = obb["long"]
            else:
                unit = np.asarray(close_dir, dtype=float)
                unit = np.array([unit[0], unit[1], 0.0]) / max(np.linalg.norm(unit[:2]), 1e-9)
                cloud = self.object_cloud(object_name)
                proj = cloud[:, :2] @ unit[:2]
                width = float(proj.max() - proj.min())
                long_u = np.array([-unit[1], unit[0], 0.0])
            if width > GRIPPER_MAX_OPENING - 0.003:
                return {
                    "grasped": False,
                    "grasp_point": None,
                    "quat_wxyz": None,
                    "close_dir": [float(v) for v in unit],
                    "yaw_deg": obb["yaw_deg"],
                    "object_side_m": round(width, 4),
                    "depth_m": None,
                    "opening_m": None,
                    "rise_m": None,
                    "attempts": k,
                    "tried": tried,
                    "reason": f"{width:.3f} m across the closing axis: wider than the jaw",
                    "budget_exhausted": False,
                }
            if self._steps_left() < self.PICK_MIN_STEPS:
                break
            # the finger tips reach FINGERTIP_BELOW_TCP past the midpoint: on a thin object
            # the pinch depth is capped (at least 3 mm below the top)
            depth = min(
                float(depth_m) * (k + 1),
                0.8 * float(size[2]),
                max(float(size[2]) - FINGERTIP_BELOW_TCP - 0.002, 0.003),
            )
            shift = 0.0
            if prev_depth is not None and abs(depth - prev_depth) < 1e-6:
                # the cap left the depth where it was: vary the pinch instead of repeating it
                shift = 0.005 if k % 2 == 1 else -0.005
            prev_depth = depth
            centre_xy = obb["centre_xy"] + long_u[:2] * shift
            point = np.array([centre_xy[0], centre_xy[1], float(ext["top_z"]) - depth])
            quat = tool_down_quat_wxyz(unit)
            g = self.grasp_at(
                point,
                quat,
                approach_m,
                object_name=object_name,
                lift_m=lift_m,
                max_descent_error_m=0.015,
            )
            entry = {
                "depth_m": round(depth, 4),
                "shift_m": shift,
                "opening_m": g["opening_m"],
                "rise_m": g["rise_m"],
                "grasp_error_m": g["grasp_error_m"],
                "reason": g["reason"],
            }
            if g["descent_reason"] == "stalled" and not g["closed_on_material"]:
                entry["skipped"] = "descent stalled"
            tried.append(entry)
            if g["closed_on_material"] and g["held"]:
                return {
                    "grasped": True,
                    "grasp_point": [round(float(v), 4) for v in point],
                    "quat_wxyz": [float(v) for v in quat],
                    "close_dir": [float(v) for v in unit],
                    "yaw_deg": obb["yaw_deg"],
                    "object_side_m": round(width, 4),
                    "depth_m": round(depth, 4),
                    "opening_m": g["opening_m"],
                    "rise_m": g["rise_m"],
                    "attempts": k + 1,
                    "tried": tried,
                    "reason": "",
                    "budget_exhausted": g["budget_exhausted"],
                }
            rise = float(g["rise_m"] or 0.0)
            if rise > 0.005 and self._steps_left() > 100:
                self.move_by_displacement(z=-rise, iterations=2, max_iterations=3)
            self.open_fingers()
            self.settle(5)
            if g["budget_exhausted"]:
                break
        return {
            "grasped": False,
            "grasp_point": None,
            "quat_wxyz": None,
            "close_dir": None,
            "yaw_deg": None,
            "object_side_m": None,
            "depth_m": None,
            "opening_m": tried[-1]["opening_m"] if tried else None,
            "rise_m": tried[-1]["rise_m"] if tried else None,
            "attempts": len(tried),
            "tried": tried,
            "reason": "no attempt held" if tried else "no budget for an attempt",
            "budget_exhausted": self._steps_left() < self.PICK_MIN_STEPS,
        }

    def palm_clearance(
        self, point: Any, quat_wxyz: Any, object_name: str, *, margin_m: float = 0.004
    ) -> dict[str, Any]:
        """Would the HAND BODY land on ``object_name`` if the tool point were pinched at
        ``point`` (BASE frame, as :meth:`get_hand_pose` reports it) with orientation
        ``quat_wxyz``?  Nothing moves.

        What is tested is the object's COLLISION-GEOM CORNER CLOUD (:meth:`object_cloud`:
        the eight box-hull corners of a primitive geom, the vertices of a mesh), not a
        dense sample of its material, so the verdict is PERMISSIVE for a wide primitive
        geom: a box more than ~2.9 cm across the hand's narrow axis can span the palm
        footprint with all eight corners outside it and read ``clear``.  ``margin_m``
        widens the across-footprint as well as the retreat.  (OPEN: sample the palm box
        against the geoms the way ``clearance_many`` does.)

        The palm begins ``HAND_BOX_ABOVE_TCP`` (2.4 cm) behind the tool point and spans
        +-10.4 cm along the finger-closing axis and +-1 cm across it: any corner of the
        object farther back along the tool than that, inside that footprint, is in the
        palm's way — a top-down pinch on a 15.7 cm bottle 4 cm above its base stalls 11 cm
        short, exactly where the palm meets the bottle's top (measured 2026-08-27, libero
        90_26; the same descent with the bottle removed reaches the point).  ``retreat_m``
        is how far the point must move BACK along the tool axis (up, for a tool-down pinch)
        for the palm to clear the highest blocking material by ``margin_m``; ``clear`` is
        True when nothing blocks.  ``top_pick`` applies it (``palm_check``).  Returns
        ``{"clear", "blocking_points", "deepest_blocker_m" (behind the tool point along
        the tool), "retreat_m"}``.
        """
        p0 = np.asarray(point, dtype=float).reshape(3)
        rot = _quat_wxyz_to_mat(np.asarray(quat_wxyz, dtype=float))
        rel = self.object_cloud(object_name) - p0
        back = rel @ (-rot[:, 2])
        along = np.abs(rel @ rot[:, 1])
        across = np.abs(rel @ rot[:, 0])
        blocking = (
            (back > HAND_BOX_ABOVE_TCP)
            & (along < HAND_HALF_ALONG_CLOSE)
            & (across < HAND_HALF_ACROSS + float(margin_m))
        )
        n = int(np.count_nonzero(blocking))
        deepest = float(back[blocking].max()) if n else 0.0
        retreat = max(deepest - HAND_BOX_ABOVE_TCP + float(margin_m), 0.0) if n else 0.0
        return {
            "clear": n == 0,
            "blocking_points": n,
            "deepest_blocker_m": round(deepest, 4),
            "retreat_m": round(retreat, 4),
        }

    def top_pick(
        self,
        object_name: str,
        fractions: Any = (0.30, 0.50, 0.15),
        *,
        close_dirs: Any = None,
        hover_m: float = 0.15,
        lift_m: float = 0.12,
        xy: Any = None,
        tilt_retry_deg: float | None = None,
        palm_check: bool = True,
    ) -> dict[str, Any]:
        """Top-down pinch of a TALL object (moka pot, bottle, mug) at its BODY ORIGIN, at a
        fraction of its height below its top face, verified by lifting.

        Two things it fixes over a coarse grasp.  A tall object's bounding-box top is its
        lid, spout or handle, so a pinch just under ``top_z`` closes on nothing (moka pot:
        gap 0.078 = fully open, yet the pot rides up wedged between the fingers) — gripping
        down the body lands on it.  And a one-shot approach knocks a tall object over
        before the fingers are near it — so the hand hovers EXACTLY above the pinch point
        first (closed loop, ``hover_m`` up), then drops straight down in 5 cm legs; a
        drop that stalls more than 1.5 cm short (a finger on the lid, the spout) is
        abandoned before the jaws close (``skipped``).  For each of ``fractions`` (in
        order) the close axis is chosen from the object's PARTS at that height
        (``get_object_parts``: the geoms spanning the pinch height, measured across x and
        y): the narrower axis that fits the jaw wins, and a height at which NEITHER fits
        is refused with a reason instead of being attempted (the moka pot's body is
        81 mm across x and 66 mm across y — the AABB rule used to close across x, wider
        than the 77 mm jaw; measured 2026-08-27).  Both wrist-roll twins of that pinch
        are tried (they are one grasp for the symmetric gripper but two IK branches, and
        far out on the table only one reaches; ``close_dirs`` overrides, duplicates are
        dropped).  The object is re-measured before EVERY attempt (a failed descent can
        knock it over) and its displacement/tilt is reported per attempt
        (``object_moved_m``, ``object_tilted``) — after a move of more than 2 cm the
        remaining attempts are abandoned (``reason``).  ``xy`` overrides the pinch xy
        (default: the body origin, which on a pot is the axis).  ``tilt_retry_deg``
        (opt-in) retries a stalled descent once with the tool pitched that far away from
        the robot — it did NOT rescue the far-side moka pot in four runs (x 0.72, the
        wrist's joint-6 limit) and the review found it makes near-side stalls worse, so it
        is off by default.  Verified by :meth:`lift_check`, never by the finger gap.
        ``palm_check`` (default True, 2026-08-27) runs :meth:`palm_clearance` on every
        pinch point: a pinch deeper than 2.4 cm under the object's highest material within
        the palm's footprint is RAISED to the deepest clear height (``palm_raised_m`` in
        ``tried``) instead of stalling 11 cm short on the palm (the wine bottle: every
        fraction of its 15.7 cm height beyond 0.13 did), and a fraction with no clear
        height under the top is skipped.  Cost ~450-700 steps per attempt with a clean
        descent, 1400-2400 when a descent stalls (each stall costs the staged legs plus
        the re-seed).  Returns ``{"grasped", "fraction", "grasp_point", "quat_wxyz",
        "close_dir", "opening_m", "rise_m", "descent_error_m", "hover_error_m",
        "attempts", "tried", "reason", "budget_exhausted"}``.
        """
        tried: list[dict[str, Any]] = []
        attempts = 0
        palm_skipped = 0
        reason = ""

        def refusal(why: str, budget: bool) -> dict[str, Any]:
            last = tried[-1] if tried else {}
            return {
                "grasped": False,
                "fraction": None,
                "grasp_point": None,
                "quat_wxyz": None,
                "close_dir": None,
                "opening_m": last.get("opening_m"),
                "rise_m": last.get("rise_m"),
                "descent_error_m": last.get("descent_error_m"),
                "hover_error_m": last.get("hover_error_m"),
                "attempts": attempts,
                "tried": tried,
                "reason": why,
                "budget_exhausted": budget,
            }

        for frac in (float(f) for f in fractions):
            ext = self.get_object_extents(object_name)
            size = np.asarray(ext["size"], dtype=float)
            z = float(ext["top_z"]) - frac * float(size[2])
            if close_dirs is None:
                widths = self._widths_at_height(object_name, z)
                fitting = [
                    (w, ax)
                    for ax, w in sorted(widths.items(), key=lambda kv: kv[1])
                    if w <= GRIPPER_MAX_OPENING - 0.003
                ]
                if not fitting:
                    tried.append(
                        {
                            "fraction": frac,
                            "skipped": f"{min(widths.values()):.3f} m across at z {z:.3f}: "
                            "wider than the jaw on both axes",
                            "widths_m": {k: round(v, 4) for k, v in widths.items()},
                        }
                    )
                    continue
                a = [1.0, 0.0, 0.0] if fitting[0][1] == "x" else [0.0, 1.0, 0.0]
                quats = [tool_down_quat_wxyz(a)]
                # the real second wrist branch: the same frame rotated 180 deg about the
                # tool axis (for a y-closing pinch the negated close_dir is NOT a twin —
                # the branch rule maps it back to the same frame)
                rot = _quat_wxyz_to_mat(quats[0]) @ np.diag([-1.0, -1.0, 1.0])
                twin = _mat_to_quat_wxyz(rot)
                if not (
                    np.allclose(twin, quats[0], atol=1e-6)
                    or np.allclose(-twin, quats[0], atol=1e-6)
                ):
                    quats.append(twin)
                dirs = [_quat_wxyz_to_mat(q)[:, 1] for q in quats]
            else:
                dirs = [np.asarray(d, dtype=float).reshape(3) for d in close_dirs]
                quats = [tool_down_quat_wxyz(d) for d in dirs]
                dedup: list[np.ndarray] = []
                keep_dirs: list[np.ndarray] = []
                for q, d in zip(quats, dirs, strict=True):
                    if any(
                        np.allclose(q, k, atol=1e-6) or np.allclose(-q, k, atol=1e-6) for k in dedup
                    ):
                        continue
                    dedup.append(q)
                    keep_dirs.append(d)
                quats, dirs = dedup, keep_dirs
            for quat, cd in zip(quats, dirs, strict=True):
                if self._steps_left() < self.PICK_MIN_STEPS:
                    return refusal("budget", True)
                attempts += 1
                # re-measure: a failed attempt can knock the object over or shove it
                origin0 = self.body_position(object_name) if xy is None else np.asarray(xy, float)
                ext = self.get_object_extents(object_name)
                z = float(ext["top_z"]) - frac * float(ext["size"][2])
                up0 = self._object_up_axis(object_name)
                point = np.array([float(origin0[0]), float(origin0[1]), z])
                palm_raised = 0.0
                if palm_check:
                    palm = self.palm_clearance(point, quat, object_name)
                    if not palm["clear"]:
                        palm_raised = float(palm["retreat_m"])
                        point = point + _quat_wxyz_to_mat(quat)[:, 2] * (-palm_raised)
                        # NOTE: the close axis and the jaw-fit gate were measured at the
                        # ORIGINAL height; the raise moves the pinch toward the top face,
                        # where a flared body (a rim, a lid) is wider.  OPEN: re-measure
                        # _widths_at_height at the raised z and skip the fraction when the
                        # chosen axis no longer fits.
                        if point[2] > float(ext["top_z"]) - 0.003:
                            attempts -= 1
                            palm_skipped += 1
                            tried.append(
                                {
                                    "fraction": frac,
                                    "close_dir": [round(float(v), 3) for v in cd],
                                    "skipped": "no pinch under the top clears the palm "
                                    f"(retreat {palm_raised:.3f} m)",
                                }
                            )
                            continue
                g = self.grasp_at(
                    point,
                    quat,
                    float(hover_m),
                    object_name=object_name,
                    lift_m=float(lift_m),
                    max_descent_error_m=0.015,
                )
                tilted = False
                if g["descent_reason"] == "stalled" and tilt_retry_deg is not None:
                    away = np.array(origin0[:2] - 0.0)
                    away = np.array([away[0], away[1], 0.0])
                    away = away / max(np.linalg.norm(away), 1e-9)
                    quat_t = tool_pitched_quat_wxyz(away, 90.0 - float(tilt_retry_deg), cd)
                    self.move_by_displacement(z=0.03, iterations=2, max_iterations=3)
                    g = self.grasp_at(
                        point,
                        quat_t,
                        0.03,
                        object_name=object_name,
                        lift_m=float(lift_m),
                        max_descent_error_m=0.015,
                    )
                    tilted = True
                    quat = quat_t
                moved = float(np.linalg.norm(self.body_position(object_name) - origin0))
                tilt = float(
                    np.degrees(
                        np.arccos(np.clip(up0 @ self._object_up_axis(object_name), -1.0, 1.0))
                    )
                )
                entry: dict[str, Any] = {
                    "fraction": frac,
                    "close_dir": [round(float(v), 3) for v in cd],
                    "palm_raised_m": round(palm_raised, 4),
                    "descent_error_m": g["grasp_error_m"],
                    "hover_error_m": g["approach_error_m"],
                    "tilted": tilted,
                    "opening_m": g["opening_m"],
                    "rise_m": g["rise_m"],
                    "object_moved_m": round(moved, 4),
                    "object_tilted": bool(tilt > 20.0),
                    "reason": g["reason"],
                }
                tried.append(entry)
                if g["closed_on_material"] and g["held"]:
                    return {
                        "grasped": True,
                        "fraction": frac,
                        "grasp_point": [round(float(v), 4) for v in point],
                        "quat_wxyz": [float(v) for v in quat],
                        "close_dir": [round(float(v), 3) for v in cd],
                        "opening_m": g["opening_m"],
                        "rise_m": g["rise_m"],
                        "descent_error_m": g["grasp_error_m"],
                        "hover_error_m": g["approach_error_m"],
                        "attempts": attempts,
                        "palm_raised_m": round(palm_raised, 4),
                        "fraction_achieved": round(
                            (float(ext["top_z"]) - float(point[2]))
                            / max(float(ext["size"][2]), 1e-9),
                            3,
                        ),
                        "tried": tried,
                        "reason": "",
                        "budget_exhausted": g["budget_exhausted"],
                    }
                if g["descent_reason"] == "stalled" and not g["closed_on_material"]:
                    entry["skipped"] = "descent stalled"
                    self.move_by_displacement(z=float(hover_m), iterations=2, max_iterations=3)
                else:
                    rise = float(g["rise_m"] or 0.0)
                    if rise > 0.005 and self._steps_left() > 100:
                        self.move_by_displacement(z=-rise, iterations=2, max_iterations=3)
                    self.open_fingers()
                    self.settle(5)
                if g["budget_exhausted"]:
                    return refusal("budget", True)
                if moved > 0.02 or tilt > 20.0:
                    reason = (
                        f"the object moved {moved:.3f} m / tilted {tilt:.0f} deg during the "
                        "attempt: remaining attempts abandoned (re-measure it)"
                    )
                    return refusal(reason, self._steps_left() < self.PICK_MIN_STEPS)
        if not reason:
            if attempts:
                reason = "no attempt held"
            elif palm_skipped:
                reason = "no pinch under the top clears the palm at any fraction"
            else:
                reason = "no height fits the jaw"
        return refusal(reason, self._steps_left() < self.PICK_MIN_STEPS)

    def _widths_at_height(self, object_name: str, z: float) -> dict[str, float]:
        """Extent of the object's collision parts across base x and y at height ``z``
        (the parts whose z range spans it; the whole box when none does).
        """
        parts = [p for p in self.get_object_parts(object_name) if p["lo"][2] <= z <= p["hi"][2]]
        if not parts:
            ext = self.get_object_extents(object_name)
            return {"x": float(ext["size"][0]), "y": float(ext["size"][1])}
        lo = np.min([p["lo"] for p in parts], axis=0)
        hi = np.max([p["hi"] for p in parts], axis=0)
        return {"x": float(hi[0] - lo[0]), "y": float(hi[1] - lo[1])}

    # ------------------------------------------------------ carrying and placing

    def travel_height(
        self, object_name: str, clear_z: float, clearance_m: float = 0.05
    ) -> dict[str, Any]:
        """Fingertip height at which the HELD object's measured bottom clears ``clear_z``
        by ``clearance_m``: ``clear_z + clearance_m + (tcp_z - bottom_z)``.

        The rule every carry needs and every fingertip-based guess gets wrong: a rim-pinched
        bowl hangs 3.6-6.5 cm below the fingertips, and a travel height chosen from the
        fingertips drags it into a drawer's front wall (libero.goal_3, three drops of three).
        ``clear_z`` is the top of whatever the object must pass over — a support's
        ``top_z``, a region's ``top_z`` (a drawer region reaches its wall tops).  Re-read
        it after a carry: the load settles in the jaws.  ``held`` says whether the object
        is in the jaws at all — with the hand open (or the object far from the fingertips)
        ``tcp_z``/``hang_m`` are meaningless numbers (a parked hand over an object on the
        table reads a 25 cm hang), so read ``held`` first.  Returns ``{"tcp_z", "hang_m"
        (tool point above the object's bottom), "clear_z", "clearance_m", "held",
        "reason"}``.
        """
        tcp_z = float(self.tcp()[2])
        bottom = float(self.get_object_extents(object_name)["bottom_z"])
        why = self._not_held(object_name)
        return {
            "tcp_z": round(float(clear_z) + float(clearance_m) + (tcp_z - bottom), 4),
            "hang_m": round(tcp_z - bottom, 4),
            "clear_z": float(clear_z),
            "clearance_m": float(clearance_m),
            "held": why is None,
            "reason": why or "",
        }

    def place_at(
        self,
        object_name: str,
        target_xy: Any,
        support_top_z: float,
        *,
        clearance_m: float = 0.05,
        step_m: float = 0.015,
        max_descent_m: float | None = None,
        watch_name: str | None = None,
    ) -> dict[str, Any]:
        """Carry the HELD object over ``target_xy`` (base frame) at a height where its
        bottom clears ``support_top_z``, then lower it until it rests and let go.

        The composed "put it down there" for a measured spot — a support's centre and
        ``top_z``, a region centre and its ``top_z``, an object's own pre-pick
        ``bottom_z`` (table patches report ``top_z`` ~2 cm below where objects rest, so
        use the latter for a table).  ``carry_to(..., object_name=)`` centres the OBJECT
        at the travel height or the current height, whichever is higher (a pick leaves the
        hand 10-15 cm above the travel height); once over the target the hand comes down
        FAST to the travel height re-measured for the settled load (``pre_descent_m``),
        and only then :meth:`place_gently` descends in ``step_m`` steps (``target_xy``
        re-centring on the way) and releases the moment the object stops following the
        hand; ``watch_name`` aborts the descent if that body is shoved.  ~500-900 steps
        for a 5-10 cm clearance (the gentle descent used to start from the pick height:
        1900 steps and a 14 cm creep, measured 2026-08-27).  ``max_descent_m`` (default
        ``clearance_m`` + 5 cm) bounds the gentle descent from the travel height; when it
        runs out the object is released where it is and ``rested`` is False even if it
        landed — check ``in_contact`` / the region.  Returns the ``place_gently`` report
        plus ``travel_z`` (the tool height actually reached before the gentle descent),
        ``pre_descent_m`` and ``xy_error_m`` (object centre to ``target_xy`` after
        release).
        """
        xy = np.asarray(target_xy, dtype=float)[:2]
        if (why := self._not_held(object_name)) is not None:
            obj = self.body_position(object_name)
            return {
                "rested": False,
                "held": False,
                "released": True,
                "descent_m": 0.0,
                "object_z": float(obj[2]),
                "pushed_support": False,
                "xy_error_m": round(float(np.linalg.norm(obj[:2] - xy)), 4),
                "travel_z": None,
                "reason": why,
                "budget_exhausted": self._steps_left() <= 1,
            }
        travel = self.travel_height(object_name, float(support_top_z), float(clearance_m))
        self.carry_to(float(xy[0]), float(xy[1]), height_m=travel["tcp_z"], object_name=object_name)
        # over the target now: come down fast to the travel height, re-measured for the
        # load as it hangs after the carry (a rim-pinched bowl hung 1.5 cm lower)
        travel = self.travel_height(object_name, float(support_top_z), float(clearance_m))
        here = self.tcp()
        pre_descent = 0.0
        if here[2] > travel["tcp_z"] + float(step_m) and self._steps_left() > 200:
            _, quat = self.hand_pose()
            self.goto(np.array([here[0], here[1], travel["tcp_z"]]), quat, iterations=2)
            pre_descent = float(here[2] - self.tcp()[2])
        descent = float(clearance_m) + 0.05 if max_descent_m is None else float(max_descent_m)
        start_z = float(self.tcp()[2])  # where the gentle descent really starts
        report = self.place_gently(
            object_name,
            max_descent=descent,
            step=float(step_m),
            watch_name=watch_name,
            target_xy=xy,
        )
        report["travel_z"] = round(start_z, 4)
        report["pre_descent_m"] = round(pre_descent, 4)
        return report

    def recentre_on(
        self,
        object_name: str,
        support_name: str,
        *,
        tolerance_m: float = 0.015,
        tries: int = 2,
        regrasp: str = "rim",
        scored_name: str | None = None,
    ) -> dict[str, Any]:
        """After a placement, close the residual between the object's centre and the
        support's centre: re-pinch the object, hop it by the residual vector, set it down.

        LIBERO's ``(On a b)`` allows 3 cm between body centres and a first placement lands
        0.9-2.9 cm off run to run — a coin flip across processes (libero.spatial_7 was
        decertified at 32 mm).  Nothing happens when the residual is already within
        ``tolerance_m``; otherwise the object is re-grasped with :meth:`rim_pick`
        (``regrasp="rim"``: bowls, plates, cups) or :meth:`box_pick` (``"box"``), moved by
        the measured residual and released with :meth:`place_gently`, up to ``tries``
        times.  ``scored_name`` is the body the PREDICATE reads when it is not the one
        the fingers take (a sub-body workaround puts them 4.9 cm apart): the residual is
        then measured on that body while the pinch still goes to ``object_name``.
        Cost 0 when not needed, ~900-1400 steps per hop.  Returns ``{"ok", "offset_m",
        "hops", "reason", "budget_exhausted"}``.
        """
        if regrasp not in ("rim", "box"):
            raise ValueError("regrasp must be 'rim' or 'box'")
        judged = scored_name or object_name

        def residual() -> np.ndarray:
            return self.body_position(support_name)[:2] - self.body_position(judged)[:2]

        hops = 0
        for _ in range(int(tries)):
            d = residual()
            if float(np.linalg.norm(d)) <= float(tolerance_m):
                break
            if self._steps_left() < self.PICK_MIN_STEPS + 500:
                return {
                    "ok": False,
                    "offset_m": round(float(np.linalg.norm(d)), 4),
                    "hops": hops,
                    "reason": "no budget for a hop",
                    "budget_exhausted": True,
                }
            pick = (
                self.rim_pick(object_name, radial_offsets=(0.0, 0.006))
                if regrasp == "rim"
                else self.box_pick(object_name, attempts=2)
            )
            if not pick["grasped"]:
                return {
                    "ok": False,
                    "offset_m": round(float(np.linalg.norm(d)), 4),
                    "hops": hops,
                    "reason": "regrasp failed",
                    "budget_exhausted": bool(pick["budget_exhausted"]),
                }
            d = residual()  # the pick itself can move the object
            self.move_by_displacement(float(d[0]), float(d[1]), 0.0)
            self.place_gently(object_name, max_descent=0.20)
            hops += 1
        d = residual()
        off = float(np.linalg.norm(d))
        return {
            "ok": bool(off <= float(tolerance_m)),
            "offset_m": round(off, 4),
            "hops": hops,
            "reason": "" if off <= float(tolerance_m) else "residual still over tolerance",
            "budget_exhausted": self._steps_left() <= 1,
        }

    def wait(self, steps: int = 20) -> dict[str, Any]:
        """Let the physics run ``steps`` control steps (the budget unit) with the arm
        target and the gripper command held.

        For an object that may still be moving: an object whose init state places it
        above its support is still falling right after reset (a bottle spawned 0.55 m up
        is still at z 0.63 after the reset's own settle and lands by control step ~15 —
        measured 2026-08-27, LIVING_ROOM_SCENE1/2/3; a pinch attempted at once closes on
        air), or a released object that needs a few steps to come to rest.  Poll
        ``body_position`` between short waits and stop when it no longer changes rather
        than waiting a fixed count.  Never steps past the horizon: the request is
        truncated to the steps left minus one and reports ``budget_exhausted`` — after
        which every primitive refuses.  Negative requests are clamped to 0.  Returns
        ``{"steps"`` (actually run), ``"budget_exhausted"}``.
        """
        asked = max(int(steps), 0)
        n = min(asked, max(self._steps_left() - 1, 0))
        self.settle(n)
        return {"steps": n, "budget_exhausted": n < asked}

    def get_object_parts(self, object_name: str) -> list[dict[str, Any]]:
        """Every collision geom of an object AND of its sub-bodies as a measured part, in
        the robot-base frame: ``[{"geom", "body", "joint", "type", "center", "half_size",
        "axes", "lo", "hi", "bound_radius"}]``.

        What an object CONSISTS of, for grasps the whole-body box cannot express: a frying
        pan's handle (a 2.5 x 1.2 cm bar, 13 cm long), a bottle's neck against its body,
        a cabinet's handle bar against its panel, a microwave's door and its handle bars
        (on the door sub-body, which the resolved body's own geoms used to hide).
        ``body`` names the body each part belongs to and ``joint`` the joint that moves
        that body (None for a rigid part), so a fixture's moving parts can be told apart.
        ``half_size`` is ALWAYS three half extents along ``axes`` — the geom's local x/y/z
        as base-frame unit vectors (row i is axis i): box: its own; sphere ``[r, r, r]``;
        cylinder ``[r, r, h]``; capsule ``[r, r, h + r]`` (the caps included); mesh: the
        vertex box in the geom's frame (a 6 mm burner plate used to read as an 18 cm
        bounding sphere) — so the part's thickness across any pinch direction is
        ``2 * sum(|axes_i . dir| * half_size_i)`` for every type; ``center`` is the centre
        of that box; ``bound_radius`` is the bounding sphere; ``lo``/``hi`` is the part's
        axis-aligned box in the base frame.  A body with no collision geoms of its own
        (nor in its sub-bodies) lists its visual ones.  Unnamed geoms are reported by their
        numeric id as a string.  Feed a point on a part to ``grasp_at``.
        """
        body, _bodies, geoms, eff = self._object_geoms(object_name, True)
        raw_m, raw_d = self._raw()
        # also every sub-body's geoms (an articulated fixture keeps its door, handle and
        # drawer geoms on child bodies)
        body_id = self.sim.model.body_name2id(body)
        parent = raw_m.body_parentid
        seen = set(geoms)
        for b in range(int(raw_m.nbody)):
            if b != body_id and self._is_descendant(int(b), body_id, parent):
                for g in self._body_geoms(b, eff):
                    if g not in seen:
                        seen.add(g)
                        geoms.append(g)
        names = {MJ_SPHERE: "sphere", MJ_CAPSULE: "capsule", MJ_CYLINDER: "cylinder"}
        names.update({MJ_BOX: "box", MJ_MESH: "mesh", MJ_PLANE: "plane"})
        base_rot = _quat_wxyz_to_mat(np.asarray(self.lle.base_link_wxyz_xyz, dtype=float)[:4])
        joint_of: dict[int, str | None] = {}
        for j in range(int(raw_m.njnt)):
            joint_of.setdefault(int(raw_m.jnt_bodyid[j]), self.sim.model.joint_id2name(j))
        out = []
        for g in geoms:
            corners = np.array([self.base_from_world(p) for p in self._geom_corners(g)])
            xmat = np.asarray(raw_d.geom_xmat[g], dtype=float).reshape(3, 3)
            rot = base_rot.T @ xmat
            size = np.asarray(raw_m.geom_size[g], dtype=float)
            kind = int(raw_m.geom_type[g])
            centre_w = np.array(raw_d.geom_xpos[g], dtype=float)
            if kind == MJ_BOX:
                half = size[:3].copy()
            elif kind == MJ_SPHERE:
                half = np.array([size[0]] * 3)
            elif kind == MJ_CYLINDER:
                half = np.array([size[0], size[0], size[1]])
            elif kind == MJ_CAPSULE:
                half = np.array([size[0], size[0], size[1] + size[0]])
            elif kind == MJ_MESH:
                mesh = int(raw_m.geom_dataid[g])
                start = int(raw_m.mesh_vertadr[mesh])
                count = int(raw_m.mesh_vertnum[mesh])
                verts = np.asarray(raw_m.mesh_vert[start : start + count], dtype=float)
                verts = verts.reshape(-1, 3)
                lo_l, hi_l = verts.min(axis=0), verts.max(axis=0)
                half = (hi_l - lo_l) / 2.0
                centre_w = centre_w + xmat @ ((hi_l + lo_l) / 2.0)
            else:
                half = np.array([float(raw_m.geom_rbound[g])] * 3)
            gb = int(raw_m.geom_bodyid[g])
            out.append(
                {
                    "geom": self.sim.model.geom_id2name(g) or str(g),
                    "body": self.sim.model.body_id2name(gb),
                    "joint": joint_of.get(gb),
                    "type": names.get(kind, str(kind)),
                    "center": [round(float(v), 4) for v in self.base_from_world(centre_w)],
                    "half_size": [round(float(v), 4) for v in half],
                    "axes": [[round(float(v), 3) for v in rot[:, i]] for i in range(3)],
                    "lo": [round(float(v), 4) for v in corners.min(axis=0)],
                    "hi": [round(float(v), 4) for v in corners.max(axis=0)],
                    "bound_radius": round(float(raw_m.geom_rbound[g]), 4),
                }
            )
        return out

    def tilt_held(
        self, angle_deg: float, *, axis: Any = None, object_name: str | None = None
    ) -> dict[str, Any]:
        """Rotate the hand ``angle_deg`` about a HORIZONTAL base-frame axis through the
        fingertips, keeping the fingertips where they are, so a held object tilts with it
        — lay a bottle down in a drawer, lean one along a wine rack, stand a pan on edge.

        ``axis`` defaults to the finger-closing axis (hand y, flattened to horizontal):
        the tool axis then swings in the plane perpendicular to the fingers and the pinch
        stays symmetric.  A given ``axis`` is flattened to horizontal too (a vertical one
        raises: that is a yaw, use ``turn_held``), and the call is refused when the tool
        axis is within ~18 deg of it (the turn would not be a tilt).  Positive =
        right-hand rule about the axis (about hand y with the tool down, +90 deg swings the
        tool from -z toward the hand's +x — which world direction that is depends on the
        grasp's wrist branch: read ``get_hand_pose()["quat_wxyz"]`` first or pass ``axis``
        explicitly, as the certified programs do; ``tool_axis_before`` is reported).
        |angle_deg| <= 180.  The rotation is VERIFIED like ``turn_held``: the hand's
        rotation about ``axis`` (the component of the full relative rotation, not a tool-
        column projection that read noise for a level tool) is measured and the residual
        re-commanded up to twice, stopping early when an attempt advanced less than 2 deg;
        with ``object_name`` the object's own turn about ``axis`` (``turned_deg``) and
        ``slip_deg`` (about ``axis`` only — a spin in the pinch about another axis is not
        counted) are reported.  Do it in free air with room around: the object swings
        about the FINGERTIPS, not about its centre (a 15 cm bottle pinched near its cap
        sweeps 15 cm), and the wrist comes down on the far side of the tilt — check
        ``get_hand_pose`` afterwards before descending.  Cost ~150-400 steps per attempt,
        up to ~750 with two re-commands.  Returns ``{"commanded_deg", "hand_turned_deg",
        "turned_deg", "slip_deg", "reached", "position_error_m", "axis", "tool_axis",
        "tool_axis_before", "attempts", "budget_exhausted"}``.
        """
        angle = float(angle_deg)
        if not np.isfinite(angle) or abs(angle) > 180.0:
            raise ValueError("angle_deg must be within -180..180")
        _, quat0 = self.hand_pose()
        rot0 = _quat_wxyz_to_mat(quat0)
        if axis is None:
            ax = rot0[:, 1].copy()
            ax[2] = 0.0
            if np.linalg.norm(ax) < 0.2:  # fingers close vertically: use the across axis
                ax = rot0[:, 0].copy()
                ax[2] = 0.0
        else:
            ax = np.asarray(axis, dtype=float).reshape(3)
            if abs(float(ax[2])) > 0.1 * max(np.linalg.norm(ax), 1e-9):
                raise ValueError(
                    "axis must be HORIZONTAL (a vertical axis is a yaw: use turn_held)"
                )
            ax[2] = 0.0
        if np.linalg.norm(ax) < 1e-6:
            raise ValueError("axis must be a non-zero horizontal base-frame vector")
        ax = ax / np.linalg.norm(ax)
        tool0 = rot0[:, 2]
        if abs(float(ax @ tool0)) > 0.95:
            raise ValueError(
                "the tool axis is (nearly) parallel to the tilt axis: a rotation about it "
                "is a roll, not a tilt — pass a different axis"
            )

        def turn_about(rot_before: np.ndarray, rot_after: np.ndarray) -> float:
            # the component about ``ax`` of the full relative rotation
            rel = rot_after @ rot_before.T
            vee = 0.5 * np.array(
                [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
            )
            return float(np.degrees(np.arctan2(float(ax @ vee), 0.5 * (np.trace(rel) - 1.0))))

        def object_rot() -> np.ndarray | None:
            if object_name is None:
                return None
            ext = self.get_object_extents(object_name)
            return _quat_wxyz_to_mat(np.asarray(self.sim.data.get_body_xquat(ext["body"]), float))

        obj0 = object_rot()
        target = self.tcp()
        remaining = angle
        turned_hand = 0.0
        error = float("inf")
        attempts = 0
        refused = False
        for _attempt in range(3):
            if self._steps_left() < 120:
                refused = True
                break
            attempts += 1
            prev_turned = turned_hand
            _, quat = self.hand_pose()
            spin = _rot_about_axis(ax, np.radians(remaining))
            error = self.goto(target, _mat_to_quat_wxyz(spin @ _quat_wxyz_to_mat(quat)))
            _, quat1 = self.hand_pose()
            turned_hand = turn_about(rot0, _quat_wxyz_to_mat(quat1))
            remaining = ((angle - turned_hand + 180.0) % 360.0) - 180.0
            if abs(remaining) < 3.0:
                break
            if abs(turned_hand - prev_turned) < 2.0:
                break  # the IK does not get there: re-commanding only burns steps
        obj1 = object_rot()
        turned = slip = None
        if obj0 is not None and obj1 is not None:
            turned = round(turn_about(obj0, obj1), 1)
            slip = round(turned_hand - turned, 1)
        _, quat_end = self.hand_pose()
        return {
            "commanded_deg": angle,
            "hand_turned_deg": round(turned_hand, 1),
            "turned_deg": turned,
            "slip_deg": slip,
            "reached": bool(abs(remaining) < 3.0),
            "position_error_m": round(float(error), 4) if np.isfinite(error) else None,
            "axis": [round(float(v), 4) for v in ax],
            "tool_axis": [round(float(v), 4) for v in _quat_wxyz_to_mat(quat_end)[:, 2]],
            "tool_axis_before": [round(float(v), 4) for v in tool0],
            "attempts": attempts,
            "budget_exhausted": bool(refused or self._steps_left() <= 1),
        }

    # ------------------------------------------------------- drawer faces and pushes

    def articulation_outward(self, joint_name: str) -> dict[str, Any]:
        """Which way a sliding part (drawer) travels when it OPENS, and where its outer
        face is — all from the joint and the geometry, no names.

        The joint's slide axis gives the line; the sign comes from the part's own
        collision cloud PROJECTED ON THE AXIS: a drawer's handle makes the cloud's
        mid-point along the axis lean OUTWARD of the part's body origin (7.5 mm on a
        LIBERO cabinet drawer, open or shut — the projection is rotation-invariant, where
        the axis-aligned box centre used to flip the answer on the wooden cabinet yawed
        -117 deg in the libero_spatial scenes; measured 2026-08-27).  The margin is small
        — the front panel sits at -78 mm and the back wall at +94 mm, only the bar at
        -109 mm tips it — so ``lean_m`` is reported: below ~2 mm the fixture's own cloud
        is the fallback (``method``), which is unreliable (another open drawer shifts it
        7 cm).  ``direction`` is the unit base-frame vector the part moves along when
        opening (horizontal for a drawer); ``face`` is the centre of the part's outer face
        band — the point of the cloud farthest along ``direction``, centred across the
        band of points within 1 cm of it — the point ``push_close``/``press_face`` press
        on; ``open_end``/``closed_end`` are the joint RANGE ends on that convention
        (LIBERO cabinets: closed at the +0.01 end, open at -0.16; the scorer's ``(Close)``
        is ``qpos > 0`` and the drawer physically rests at ~+0.002, so do not push toward
        ``closed_end`` itself); ``limited`` is False for an unlimited slide (both ends
        then read 0).  Hinge joints raise ValueError: use ``close_articulated``/
        ``open_articulated``, which compute the tangent per stroke; so does an unknown or
        a robot joint.  Returns ``{"direction", "axis", "open_sign", "face", "center",
        "lo", "hi", "part_body", "parent_body", "qpos", "open_end", "closed_end",
        "lean_m", "method", "limited"}``.
        """
        if joint_name not in self.sim.model.joint_names:
            raise ValueError(f"unknown joint {joint_name!r}; see get_joint_states()")
        if joint_name.startswith(("robot", "gripper")):
            raise ValueError(f"{joint_name} is a robot joint, not an articulated part")
        info = self._joint_info(joint_name)
        if info["type"] != MJ_SLIDE:
            raise ValueError(
                f"{joint_name} is not a slide joint; use close_articulated/open_articulated"
            )
        raw_m, _ = self._raw()
        part = info["body_id"]
        parent = int(raw_m.body_parentid[part])
        anchor = self.base_from_world(info["anchor_world"])
        axis = self.base_from_world(info["anchor_world"] + info["axis_world"]) - anchor
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        part_cloud = self._joint_body_cloud(part)
        part_c = (part_cloud.min(axis=0) + part_cloud.max(axis=0)) / 2.0
        # the part's cloud leans toward its handle relative to its OWN body origin (a
        # LIBERO drawer's origin is its box centre carried along the joint): 7.5 mm on the
        # white cabinet, the same whether the drawer is open or shut.  The fixture's box
        # centre is NOT a reference — another open drawer moves it 7 cm and flipped the
        # sign (libero.90_23: the "outer face" came out at the back of the cabinet)
        part_origin = self.base_from_world(self.sim.data.xpos[part])
        proj = part_cloud @ axis
        lean = float(0.5 * (proj.max() + proj.min()) - part_origin @ axis)
        method = "part cloud vs body origin (projected on the axis)"
        if abs(lean) < 0.002:  # no handle asymmetry: fall back to the fixture's cloud
            fixture_cloud = self._fixture_cloud(parent, exclude=part)
            fproj = fixture_cloud @ axis
            lean = float(0.5 * (proj.max() + proj.min()) - 0.5 * (fproj.max() + fproj.min()))
            method = "part vs fixture cloud (fallback, unreliable with another part open)"
        sign = 1.0 if lean >= 0.0 else -1.0
        direction = axis * sign
        lo, hi = part_cloud.min(axis=0), part_cloud.max(axis=0)
        # the outer face: the cloud's extreme along the opening direction, centred across
        # the band of points within 1 cm of it (the AABB half-extent is only right for an
        # axis-aligned part: 87 mm off on the yawed wooden cabinet)
        along = part_cloud @ direction
        band = part_cloud[along >= along.max() - 0.01]
        band_c = (band.min(axis=0) + band.max(axis=0)) / 2.0
        face = band_c + direction * float(along.max() - band_c @ direction)
        # in joint space, moving along +axis raises qpos: the open end is the range end
        # reached by travelling along ``direction``
        rlo, rhi = (float(v) for v in info["range"])
        limited = bool(rhi > rlo)
        open_end, closed_end = (rhi, rlo) if sign > 0 else (rlo, rhi)
        return {
            "direction": [round(float(v), 4) for v in direction],
            "axis": [round(float(v), 4) for v in axis],
            "open_sign": sign,
            "face": [round(float(v), 4) for v in face],
            "center": [round(float(v), 4) for v in part_c],
            "lo": [round(float(v), 4) for v in lo],
            "hi": [round(float(v), 4) for v in hi],
            "part_body": self.sim.model.body_id2name(part),
            "parent_body": self.sim.model.body_id2name(parent),
            "qpos": round(float(info["qpos"]), 4),
            "open_end": open_end,
            "closed_end": closed_end,
            "lean_m": round(lean, 4),
            "method": method,
            "limited": limited,
        }

    def _fixture_cloud(self, root: int, *, exclude: int) -> np.ndarray:
        """Base-frame corner cloud of a fixture body and its subtree, minus one part."""
        raw_m, _ = self._raw()
        parent = raw_m.body_parentid
        bodies = [root] + [
            b
            for b in range(int(raw_m.nbody))
            if b != root and self._is_descendant(int(b), root, parent)
        ]
        bodies = [b for b in bodies if b != exclude and not self._is_descendant(b, exclude, parent)]
        clouds = []
        for b in bodies:
            geoms = self._body_geoms(b, True) or self._body_geoms(b, False)
            for g in geoms:
                clouds.append(np.array([self.base_from_world(p) for p in self._geom_corners(g)]))
        if not clouds:
            raise RuntimeError("the fixture has no geoms to measure")
        return np.concatenate(clouds)

    def push_close(
        self,
        joint_name: str,
        step_m: float = 0.03,
        *,
        max_steps: int = 4,
        standoff_m: float = 0.10,
        height_frac: float = 0.5,
        stall_m: float = 0.002,
        close_dir: Any = "horizontal",
        target_qpos: float | None = None,
    ) -> dict[str, Any]:
        """Close a drawer by pushing its measured outer face with the CLOSED gripper in
        verified ``step_m`` increments until the slide joint stalls twice.

        The tool is held HORIZONTAL, pointing at the face, with the closed fingers lying
        along ``close_dir`` — ``"horizontal"`` (default: across the face, so the hand is
        only ~2 cm tall and fits a bottom drawer's handle height, where a hand with
        vertical fingers stands on the table), ``"vertical"``, or a base-frame direction;
        the stub is seated ``standoff_m`` in front of the face at ``height_frac`` of the part's
        height, then ``move_by_displacement`` drives it inward step by step, reading the
        joint after every step; two steps in a row that move the joint less than
        ``stall_m`` end it (``reason`` "stalled") — within a millimetre of the stop every
        step moves less than 2 mm, so waiting for the drawer to "move first" used to run
        all ``max_steps`` at 200-335 blocked steps each.  ``target_qpos`` (default None:
        push until stalled) stops the push as soon as the joint is past it in the closing
        sense.  The seat in front of the face is verified: a seat missed by more than 2 cm
        aborts with ``reason`` "approach missed" before any push.  The hand is left
        PRESSING on the face (a legal final state — the scorer reads the joint; retreating
        with closed fingertips can hook the handle bar back out).  This is the finishing
        move after ``close_articulated`` for the last millimetres; for the bulk of the
        travel use ``close_articulated`` (goto-based strokes) — against a resisting drawer
        ``move_by_displacement`` costs ~240 steps per BLOCKED step (3669 steps for 0.0 m
        measured), so ``max_steps`` (default 4) bounds the damage.  ``qpos`` is the final
        joint value; LIBERO's ``(Close)`` on a cabinet wants ``qpos > 0``.  Returns
        ``{"qpos", "start_qpos", "moved_m", "steps", "stalls", "face", "seat_error_m",
        "fingers_closed", "reason", "budget_exhausted"}``.
        """
        out = self.articulation_outward(joint_name)
        n = np.asarray(out["direction"], dtype=float)
        lo, hi = np.asarray(out["lo"], dtype=float), np.asarray(out["hi"], dtype=float)
        face = np.asarray(out["face"], dtype=float)
        z = float(lo[2] + np.clip(float(height_frac), 0.0, 1.0) * (hi[2] - lo[2]))
        quat = tool_horizontal_quat_wxyz(-n, self._finger_dir(close_dir, n))
        closed = self.close_fingers()
        start = float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])
        seat = np.array([face[0] + n[0] * standoff_m, face[1] + n[1] * standoff_m, z])
        seat_err = self._approach_from_front(seat, n, quat)
        q = start
        stalls = 0
        steps = 0
        reason = ""
        closing_sign = -float(out["open_sign"])  # qpos moves this way when closing

        def past_target(qv: float) -> bool:
            return target_qpos is not None and closing_sign * (qv - float(target_qpos)) > 0

        if seat_err > 0.02:
            reason = "approach missed"
        elif past_target(q):
            reason = "already past target_qpos"
        else:
            for _ in range(int(max_steps)):
                if self._steps_left() < 250:
                    reason = "budget"
                    break
                self.move_by_displacement(
                    -n[0] * float(step_m),
                    -n[1] * float(step_m),
                    0.0,
                    iterations=2,
                    max_iterations=2,
                )
                steps += 1
                q_new = float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])
                if abs(q_new - q) < float(stall_m):
                    stalls += 1
                else:
                    stalls = 0
                q = q_new
                if past_target(q):
                    reason = "past target_qpos"
                    break
                if stalls >= 2:
                    reason = "stalled"
                    break
        return {
            "qpos": round(q, 4),
            "start_qpos": round(start, 4),
            "moved_m": round(abs(q - start), 4),
            "steps": steps,
            "stalls": stalls,
            "face": out["face"],
            "seat_error_m": round(float(seat_err), 4),
            "fingers_closed": bool(closed),
            "reason": reason,
            "budget_exhausted": self._steps_left() < 250,
        }

    def _approach_from_front(self, seat: np.ndarray, outward: np.ndarray, quat: Any) -> float:
        """Reach ``seat`` (a point in front of a face) from the OUTSIDE and return the
        final goto error: when the hand is more than 5 cm above the seat, behind the seat's
        plane (a press leaves it there), or more than 3 cm off to the side, first move
        10 cm further out along ``outward`` at the current height, descend there, then
        slide in.  A straight descent onto the seat of a closed bottom drawer swept the
        arm through the OPEN top drawer above it and shut it (libero.90_23, twice, measured
        2026-08-27); a diagonal slide from behind the plane drags the handle bar.
        """
        out = np.asarray(outward, dtype=float)
        here = self.tcp()
        rel = here - seat
        lateral = rel - out * float(rel @ out)
        lateral[2] = 0.0
        detour = (
            here[2] > seat[2] + 0.05
            or float(rel @ out) < -0.01
            or float(np.linalg.norm(lateral)) > 0.03
        )
        if detour:
            front = seat + out * 0.10
            self.goto(np.array([front[0], front[1], here[2]]), quat, iterations=2, max_iterations=3)
            self.goto(front, quat, iterations=2, max_iterations=4)
        return float(self.goto(seat, quat))

    @staticmethod
    def _finger_dir(close_dir: Any, push: np.ndarray) -> np.ndarray:
        """Base-frame direction the closed fingers lie along for a push along ``push``."""
        if isinstance(close_dir, str):
            if close_dir == "vertical":
                return np.array([0.0, 0.0, 1.0])
            if close_dir == "horizontal":
                return np.array([-push[1], push[0], 0.0])
            raise ValueError("close_dir must be 'horizontal', 'vertical' or a direction")
        d = np.asarray(close_dir, dtype=float).reshape(3)
        if np.linalg.norm(d) < 1e-6:
            raise ValueError("close_dir must be a non-zero direction")
        return d

    def press_face(
        self,
        joint_name: str,
        depth_m: float = 0.12,
        *,
        tries: int = 3,
        height_frac: float = 0.5,
        pitches: Any = (0.0, 20.0, 40.0),
        target_qpos: float = 0.0,
        standoff_m: float = 0.05,
        close_dir: Any = "horizontal",
    ) -> dict[str, Any]:
        """Drive a nearly closed drawer home with single DEEP one-shot ``goto_pose``
        commands past its measured outer face, at escalating wrist pitches.

        Encodes a measured step-cost fact (2026-08-27): one deep one-shot IK command
        ``depth_m`` past the face costs ~280 steps and pushes with the controller's full
        authority, where ``move_by_displacement`` against a resisting drawer costs ~240
        steps per BLOCKED step and moves it nothing.  Use it to finish the last
        centimetre after ``close_articulated`` (whose ``park()`` retreat tends to hook the
        handle bar back out) when ``qpos`` must cross ``target_qpos`` (LIBERO cabinets:
        ``> 0``, judged in the closing sense the joint's geometry gives — not a fixed
        sign).  Each try uses the next pitch of ``pitches`` (the tool tilted down that
        far, the fingers along ``close_dir``); it stops as soon as the joint is past
        ``target_qpos`` and leaves the hand pressing (the scorer reads the joint).  The
        face is re-measured before every try — it moves — and a try starts by lifting 6 cm
        and re-seating ``standoff_m`` in FRONT of the face whenever the hand is not already
        within 2 cm of that seat (the first try used to skip the re-seat when the hand was
        merely far enough out, and dove diagonally from 26 cm up through the open drawer
        above): a diagonal approach from wherever the hand was hooks the handle bar and
        pulls the drawer OUT (measured 2026-08-27: -0.017 to -0.042 on libero.90_22).
        ``close_dir`` as in :meth:`push_close` (default horizontal fingers: the hand stays
        2 cm tall).  An empty ``pitches`` raises.  Returns ``{"qpos", "start_qpos",
        "tries", "pitch_deg", "crossed", "budget_exhausted"}``.
        """
        ladder = tuple(float(p) for p in pitches)
        if not ladder:
            raise ValueError("pitches must not be empty")
        self.close_fingers()

        def qpos() -> float:
            return float(np.atleast_1d(self.sim.data.get_joint_qpos(joint_name))[0])

        out = self.articulation_outward(joint_name)
        closing_sign = -float(out["open_sign"])

        def past(qv: float) -> bool:
            return closing_sign * (qv - float(target_qpos)) > 0

        start = qpos()
        used = 0
        pitch = None
        for k in range(int(tries)):
            if past(qpos()):
                break
            if self._steps_left() < 300:
                break
            out = self.articulation_outward(joint_name)
            n = np.asarray(out["direction"], dtype=float)
            lo, hi = np.asarray(out["lo"], dtype=float), np.asarray(out["hi"], dtype=float)
            face = np.asarray(out["face"], dtype=float)
            z = float(lo[2] + np.clip(float(height_frac), 0.0, 1.0) * (hi[2] - lo[2]))
            pitch = ladder[min(k, len(ladder) - 1)]
            quat = tool_pitched_quat_wxyz(-n, pitch, self._finger_dir(close_dir, -n))
            target = np.array([face[0] - n[0] * depth_m, face[1] - n[1] * depth_m, z])
            seat = np.array([face[0] + n[0] * standoff_m, face[1] + n[1] * standoff_m, z])
            here = self.tcp()
            if k > 0 or float(np.linalg.norm(here - seat)) > 0.02:
                self.move_by_displacement(z=0.06, iterations=2, max_iterations=2)
                self._approach_from_front(seat, n, quat)
            self.goto(target, quat, iterations=1, max_iterations=1, max_steps=120)
            used += 1
        final = qpos()
        return {
            "qpos": round(final, 4),
            "start_qpos": round(start, 4),
            "tries": used,
            "pitch_deg": pitch,
            "crossed": bool(past(final)),
            "budget_exhausted": self._steps_left() < 300,
        }

    # ------------------------------------------------ flat insertion and site frames
    # Promoted 2026-08-27 (round 4) from the flat-insert helpers of libero.90_86/88 and the
    # open requests of the round-3 sweeps.

    def _open_sides(self, region: dict[str, Any], ignore_bodies: Any = ()) -> list[dict[str, Any]]:
        """Which vertical faces of a region box are free of fixture material.

        For each of +-x / +-y a 3 x 3 grid of probe points on that face (inset 1.5 cm from
        its edges) is tested with :meth:`clearance_many`; a face whose probes all clear
        1 cm is ``open``.  Entries are ``{"direction"`` (outward unit vector), ``"face"``
        (the face centre, base frame), ``"clearance_m"``, ``"open"``, ``"toward_robot"``
        (cosine between the outward direction and the direction to the robot base)}``,
        sorted with the face nearest the robot first.  Mesh geoms are measured by their
        bounding sphere (conservative), so a wall of mesh geoms can read as "closed" from
        further away than it is.
        """
        lo = np.asarray(region["lo"], dtype=float)
        hi = np.asarray(region["hi"], dtype=float)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out: list[dict[str, Any]] = []
        for d in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
            d = np.asarray(d, dtype=float)
            axis = int(np.argmax(np.abs(d)))
            other = 1 - axis
            face = centre.copy()
            face[axis] = centre[axis] + d[axis] * half[axis]
            inset_o = max(float(half[other]) - 0.015, 0.0)
            inset_z = max(float(half[2]) - 0.015, 0.0)
            probes = []
            for fo in (-1.0, 0.0, 1.0):
                for fz in (-1.0, 0.0, 1.0):
                    p = face.copy()
                    p[other] += fo * inset_o
                    p[2] += fz * inset_z
                    probes.append(p)
            clear = self.clearance_many(probes, ignore_bodies=ignore_bodies, limit=0.05)
            c = float(clear.min())
            toward = float(-d[:2] @ centre[:2]) / max(float(np.linalg.norm(centre[:2])), 1e-9)
            out.append(
                {
                    "direction": d,
                    "face": face,
                    "clearance_m": round(c, 4),
                    "open": bool(c > 0.01),
                    "toward_robot": round(toward, 3),
                }
            )
        out.sort(key=lambda e: -e["toward_robot"])
        return out

    #: an insertion needs a verified pick (~450), the wrist turn and two low moves (~600)
    INSERT_MIN_STEPS = 1100

    def insert_flat_into_region(
        self,
        object_name: str,
        region_name: str,
        *,
        entry_dir: Any = None,
        depth_m: float = 0.03,
        grasp_below_top_m: float = 0.02,
        floor_clearance_m: float = 0.006,
        pitch_deg: float = 0.0,
        rotate_height_m: float | None = None,
        standoff_m: float = 0.05,
        servo_rounds: int = 2,
        lift_clear_m: float = 0.04,
        scored_name: str | None = None,
    ) -> dict[str, Any]:
        """Slide a flat, STANDING object (a book) sideways into a compartment that is open
        only at the front — a shelf level, a caddy slot approached from the side, a
        microwave cavity — and let go inside, with the hand never backing out.

        The manoeuvre (certified as hand-written programs on libero.90_86/88, 2026-08-27):
        (1) a top-down pinch across the object's OWN thin side ``grasp_below_top_m`` under
        its top edge, verified by a short lift, so most of the object hangs below the
        fingers; (2) ONE wrist turn to "tool along the entry direction, fingers closing
        vertically" at ``rotate_height_m`` (default: MEASURED with
        :meth:`travel_height` from the compartment ceiling and the load hanging in the
        jaws, reported as ``rotate_z``) — the pinch fixed the object's thickness normal
        to the finger axis, so the flat pose comes out of the wrist whatever the object's
        yaw was; (3) a descent OUTSIDE the compartment to the carry height, where the
        object's body origin is MEASURED relative to the tool point (the low level-wrist
        moves end 8-32 deg off the commanded orientation while ``check_reachable`` reports
        0.0 — measuring after the descent absorbs that); (4) one move that puts the ORIGIN
        (the point ``In`` tests; on these books the leading edge once flat) ``depth_m``
        inside the region's open face, corrected ``servo_rounds`` times; (5) release and a
        vertical lift of ``lift_clear_m`` — NO retreat: released, the object settles between
        the pads and a straight retreat dragged a book 14 cm back out onto the table
        (90_88) and a mug out of a microwave; an open hand over the compartment is a legal
        final state.

        ``entry_dir`` (base frame, horizontal, pointing INTO the compartment) defaults to
        the open face nearest the robot found by :meth:`_open_sides` (probes on each face
        of the box against the parent fixture's geoms).  The carry height is computed from
        the box FLOOR (``lo[2]``), the object's half thickness and the finger geometry:
        when the fingertips will be inside the compartment the lower finger's outer face
        (``FINGER_THICKNESS_M`` below the pad) must clear the floor by
        ``floor_clearance_m`` — a 2.7 cm book carried with its mid-plane 1 cm above a shelf
        floor put the lower pad 1 cm BELOW that floor and the fingertip hooked the plate
        edge (measured); when the object is long enough for the fingers to stay outside
        (``hand_inside`` False: a 13.5 cm book pinched 2 cm from its edge reaches 11.5 cm
        ahead of the tool point) the object plus ``PAD_BELOW_TOOL_M`` (the 1 cm the lower
        pad's outer face still dips below the tool point) must clear it.  The upper
        finger / the object's top is checked against the box ceiling; a compartment that
        cannot take the hand (or the object) is refused with ``reason`` before anything
        moves.  ``pitch_deg`` tilts the carry tool DOWN that far — and the object with it
        (leading edge down: a parallel pinch across the thickness keeps the object's plane
        perpendicular to the fingers, so a pitched wrist can NOT carry it level; a level
        wrist cannot be held below ~8 cm on this arm, which is what makes a 7 cm bottom
        compartment out of reach for a flat carry — measured 90_89) — ``object_pitch_deg``
        reports what the object did.  ``scored_name`` names the body the predicate is
        judged on when it differs from the grip body.  Cost ~1200-1700 steps end to end
        (measured on the two certificates); refused below ``INSERT_MIN_STEPS``.  Returns
        ``{"inside", "object_pos", "region", "entry_dir", "open_sides", "hand_inside",
        "carry_z", "grasp", "rotate_dev_deg", "insert_error_m", "object_pitch_deg",
        "released", "reason", "budget_exhausted"}``.
        """
        region = self._region(region_name)
        lo = np.asarray(region["lo"], dtype=float)
        hi = np.asarray(region["hi"], dtype=float)
        centre = 0.5 * (lo + hi)
        judge = scored_name or object_name
        report: dict[str, Any] = {
            "inside": False,
            "object_pos": [float(v) for v in self.body_position(judge)],
            "region": region_name,
            "entry_dir": None,
            "open_sides": [],
            "hand_inside": None,
            "carry_z": None,
            "grasp": None,
            "rotate_dev_deg": None,
            "insert_error_m": None,
            "object_pitch_deg": None,
            "released": False,
            "reason": "",
            "budget_exhausted": self._steps_left() <= 1,
        }
        if self._steps_left() < self.INSERT_MIN_STEPS:
            report["reason"] = f"fewer than {self.INSERT_MIN_STEPS} steps left: insertion refused"
            report["budget_exhausted"] = True
            return report
        ext = self.get_object_extents(object_name)
        sides = self._open_sides(region, ignore_bodies=[ext["body"]])
        report["open_sides"] = [
            {
                k: ([round(float(x), 4) for x in v] if isinstance(v, np.ndarray) else v)
                for k, v in s.items()
            }
            for s in sides
        ]
        if entry_dir is None:
            open_sides = [s for s in sides if s["open"]]
            if not open_sides:
                report["reason"] = "no open face on the region box (pass entry_dir)"
                return report
            chosen = open_sides[0]
            e_in = -chosen["direction"]
            face_c = chosen["face"]
        else:
            e_in = np.asarray(entry_dir, dtype=float).reshape(3)
            e_in[2] = 0.0
            if np.linalg.norm(e_in) < 1e-6:
                raise ValueError("entry_dir must be a horizontal, non-zero base-frame direction")
            e_in = e_in / np.linalg.norm(e_in)
            axis = int(np.argmax(np.abs(e_in)))
            face_c = centre.copy()
            face_c[axis] = centre[axis] - np.sign(e_in[axis]) * 0.5 * (hi[axis] - lo[axis])
        report["entry_dir"] = [round(float(v), 4) for v in e_in]
        # the object's own thin side is what the fingers close across
        obb = self._horizontal_obb(object_name)
        thin = float(obb["short_m"])
        if thin > GRIPPER_MAX_OPENING - 0.003:
            report["reason"] = f"{thin:.3f} m thick: wider than the jaw"
            return report
        height = float(ext["size"][2])
        reach_ahead = height - float(grasp_below_top_m)  # how far the object extends past the tool
        hand_inside = bool(float(depth_m) - reach_ahead > -0.01)
        half_t = 0.5 * thin
        floor = float(lo[2])
        ceiling = float(hi[2])
        if hand_inside:
            carry_z = floor + half_t + FINGER_THICKNESS_M + float(floor_clearance_m)
            top_needed = carry_z + half_t + FINGER_THICKNESS_M + 0.004
        else:
            # even with the fingers outside the compartment the lower pad's outer face
            # dips ~1 cm below the tool point: the sim test on the shelf fails without
            # it (carry_z 0.0647 against a 0.0743 floor+finger bound), so it is the
            # pad allowance, not a duplicate of floor_clearance_m
            carry_z = floor + half_t + PAD_BELOW_TOOL_M + float(floor_clearance_m)
            top_needed = carry_z + half_t + 0.005
        report["hand_inside"] = hand_inside
        report["carry_z"] = round(carry_z, 4)
        if top_needed > ceiling:
            report["reason"] = (
                f"compartment too low: {'the hand' if hand_inside else 'the object'} needs the "
                f"tool at z {carry_z:.3f} and reaches z {top_needed:.3f} against a ceiling of "
                f"{ceiling:.3f}"
            )
            return report
        # 1. pinch across the thin side near the top edge, verified by a short lift
        point = np.array(
            [
                obb["centre_xy"][0],
                obb["centre_xy"][1],
                float(ext["top_z"]) - float(grasp_below_top_m),
            ]
        )
        q_grasp = tool_down_quat_wxyz(obb["short"])
        g = self.grasp_at(
            point, q_grasp, 0.10, object_name=object_name, lift_m=0.03, max_descent_error_m=0.015
        )
        report["grasp"] = {
            k: g[k]
            for k in ("opening_m", "held", "rise_m", "grasp_error_m", "descent_reason", "reason")
        }
        if not g["held"]:
            self.open_fingers()
            report["reason"] = "grasp failed: " + (g["reason"] or g["descent_reason"] or "not held")
            report["budget_exhausted"] = bool(g["budget_exhausted"])
            return report
        # 2. straight up, then ONE wrist turn at the stand-off point outside the face
        # the turn height is MEASURED from the compartment it has to clear and the load
        # hanging in the jaws (travel_height), not a base-frame number written here
        if rotate_height_m is None:
            rotate_z = float(self.travel_height(object_name, float(ceiling))["tcp_z"])
        else:
            rotate_z = float(rotate_height_m)
        report["rotate_z"] = round(rotate_z, 4)
        here = self.tcp()
        self.goto(
            np.array([here[0], here[1], rotate_z]),
            q_grasp,
            iterations=2,
            max_iterations=4,
        )
        q_carry = tool_pitched_quat_wxyz(e_in, float(pitch_deg), [0.0, 0.0, 1.0])
        standoff = face_c - e_in * (reach_ahead + float(standoff_m))
        self.goto(np.array([standoff[0], standoff[1], rotate_z]), q_carry)
        _, quat_now = self.hand_pose()
        dev = _quat_wxyz_to_mat(quat_now).T @ _quat_wxyz_to_mat(q_carry)
        report["rotate_dev_deg"] = round(
            float(np.degrees(np.arccos(np.clip((np.trace(dev) - 1.0) / 2.0, -1.0, 1.0)))), 2
        )
        # 3. descend outside, measure where the origin hangs relative to the tool point
        self.goto(np.array([standoff[0], standoff[1], carry_z]), q_carry)
        target = face_c + e_in * float(depth_m)
        err = float("inf")
        for _ in range(max(int(servo_rounds), 1)):
            if self._steps_left() < 200:
                break
            off = self.body_position(object_name) - self.tcp()
            aim = np.array([target[0] - off[0], target[1] - off[1], carry_z])
            self.goto(aim, q_carry)
            now = self.body_position(object_name)
            err = float(np.linalg.norm(now[:2] - target[:2]))
            if err < 0.006:
                break
        report["insert_error_m"] = round(err, 4) if np.isfinite(err) else None
        # the object's standing-up axis is its LENGTH axis once flat: level = horizontal,
        # positive = the leading edge (ahead of the fingers) down
        up1 = self._object_up_axis(object_name)
        report["object_pitch_deg"] = round(
            float(np.degrees(np.arcsin(np.clip(float(up1[2]), -1.0, 1.0)))), 1
        )
        # 4. release and lift the open pads clear; never back out
        released = self.open_fingers()
        self.settle(10)
        report["released"] = bool(released)
        if released and self._steps_left() > 60:
            lift = float(lift_clear_m)
            if hand_inside:
                lift = min(
                    lift, max(ceiling - 0.005 - (self.tcp()[2] + half_t + FINGER_THICKNESS_M), 0.0)
                )
            if lift > 0.003:
                _, quat_now = self.hand_pose()
                self.goto(
                    self.tcp() + np.array([0.0, 0.0, lift]),
                    quat_now,
                    iterations=2,
                    max_iterations=3,
                )
        self.settle(5)
        region = self._region(region_name)
        rest = self.body_position(judge)
        lo2 = np.asarray(region["lo"], dtype=float) - np.array([0.0, 0.0, 0.01])
        hi2 = np.asarray(region["hi"], dtype=float)
        report["object_pos"] = [float(v) for v in rest]
        report["inside"] = bool(released and np.all(rest > lo2) and np.all(rest < hi2))
        report["budget_exhausted"] = self._steps_left() <= 1 or not released
        return report

    def align_held_to_region(
        self,
        object_name: str,
        region_name: str,
        *,
        object_axis: Any = (0.0, 0.0, 1.0),
        site_axis: Any = (0.0, 0.0, 1.0),
        tolerance_deg: float = 5.0,
    ) -> dict[str, Any]:
        """Tilt the HELD object so its own ``object_axis`` (body frame) points along the
        region site's ``site_axis`` (site frame, ``site_mat`` from :meth:`get_region_boxes`)
        — the wine rack's cradle (``libero.goal_9``: the ``top_region`` site is ROLLED
        62 deg) scores ``(On bottle region)`` in the SITE frame, so a standing bottle must
        LIE along the cradle before it is set down.

        Nothing is guessed: both axes are read from MuJoCo and reported in the BASE frame
        (``site_axis_base``, ``object_axis_before``), the turn is the angle between them
        about their cross product, and the motion is :meth:`tilt_held` (verified,
        ~150-400 steps).  The axes are re-read AFTERWARDS, so ``angle_after_deg`` and
        ``aligned`` are measurements, not the commanded turn — ``reached`` is
        ``tilt_held``'s own verdict (the WRIST reached the commanded angle), which is not
        the same thing.  Within ``tolerance_deg`` nothing moves (a bottle pinched by its
        neck settles a few degrees in the pads).  Two inputs are REFUSED without moving,
        with a ``reason``: an empty hand, and a pair of axes whose difference is a pure YAW
        (their cross product is vertical) — a tilt cannot yaw an object, and rolling it
        about the finger axis instead used to be a large wrong motion reported as success.
        Returns the ``tilt_held`` report plus ``{"angle_deg", "angle_after_deg", "aligned",
        "align_axis", "yaw_residual_deg", "site_axis_base", "object_axis_before"}``.
        """
        why = self._not_held(object_name)
        region = self._region(region_name)
        mat = np.asarray(region["site_mat"], dtype=float).reshape(3, 3)
        s_world = mat @ np.asarray(site_axis, dtype=float).reshape(3)
        s_world = self.base_from_world(s_world) - self.base_from_world([0.0, 0.0, 0.0])
        s_world = s_world / max(np.linalg.norm(s_world), 1e-9)
        ext = self.get_object_extents(object_name)
        quat = np.asarray(self.sim.data.get_body_xquat(ext["body"]), dtype=float)
        o_world = _quat_wxyz_to_mat(quat) @ np.asarray(object_axis, dtype=float).reshape(3)
        o_world = self.base_from_world(o_world) - self.base_from_world([0.0, 0.0, 0.0])
        o_world = o_world / max(np.linalg.norm(o_world), 1e-9)
        angle = float(np.degrees(np.arccos(np.clip(o_world @ s_world, -1.0, 1.0))))
        full_axis = np.cross(o_world, s_world)
        yaw_residual = (
            float(np.degrees(np.arcsin(np.clip(abs(full_axis[2]), 0.0, 1.0))))
            if angle > 1e-3
            else 0.0
        )
        axis = full_axis.copy()
        axis[2] = 0.0
        base = {
            "angle_deg": round(angle, 1),
            "align_axis": [round(float(v), 4) for v in (axis / max(np.linalg.norm(axis), 1e-9))],
            "yaw_residual_deg": round(yaw_residual, 1),
            "site_axis_base": [round(float(v), 4) for v in s_world],
            "object_axis_before": [round(float(v), 4) for v in o_world],
        }
        idle = {
            "reached": False,
            "aligned": False,
            "angle_after_deg": round(angle, 1),
            "hand_turned_deg": 0.0,
            "turned_deg": 0.0,
            "slip_deg": 0.0,
            "commanded_deg": 0.0,
            "attempts": 0,
            "budget_exhausted": False,
        }
        if why is not None:
            return {**base, **idle, "reason": why}
        if angle < float(tolerance_deg):
            return {**base, **idle, "reached": True, "aligned": True, "reason": ""}
        if np.linalg.norm(full_axis) > 1e-6 and np.linalg.norm(axis) < 1e-6:
            # the cross product is VERTICAL: the two axes differ by a yaw only, which a
            # tilt cannot do.  Rolling about the finger axis instead (what this used to
            # do) is a large motion in the wrong degree of freedom.
            return {
                **base,
                **idle,
                "reason": f"the two axes differ by a {angle:.0f} deg YAW, which tilt_held "
                "cannot do (turn the wrist, or re-grip)",
            }
        if np.linalg.norm(axis) < 1e-6:
            # the two axes lie in one vertical line (a 180 deg flip): tilt about the finger axis
            _, hq = self.hand_pose()
            axis = _quat_wxyz_to_mat(hq)[:, 1].copy()
            axis[2] = 0.0
        rep = self.tilt_held(angle, axis=axis, object_name=object_name)
        # measure the alignment that was asked for, not the wrist turn that was commanded
        quat_after = np.asarray(self.sim.data.get_body_xquat(ext["body"]), dtype=float)
        o_after = _quat_wxyz_to_mat(quat_after) @ np.asarray(object_axis, dtype=float).reshape(3)
        o_after = self.base_from_world(o_after) - self.base_from_world([0.0, 0.0, 0.0])
        o_after = o_after / max(np.linalg.norm(o_after), 1e-9)
        after = float(np.degrees(np.arccos(np.clip(o_after @ s_world, -1.0, 1.0))))
        return {
            **base,
            **rep,
            "angle_after_deg": round(after, 1),
            "aligned": bool(after < float(tolerance_deg)),
            "object_axis_after": [round(float(v), 4) for v in o_after],
        }

    # ------------------------------------------------------------ registry

    def tools(self) -> dict[str, Callable[..., Any]]:
        """Name → callable for :meth:`LiberoAdapter.extra_tools`."""
        return {
            "get_gripper_opening": self.get_gripper_opening,
            "get_step_budget": self.get_step_budget,
            "move_by_displacement": self.move_by_displacement,
            "check_reachable": self.check_reachable,
            "probe_goto": self.probe_goto,
            "joint_positions": self.joint_positions,
            "reachable_tool_quat": self.reachable_tool_quat,
            "park": self.park,
            "checkpoint": self.checkpoint,
            "restore": self.restore,
            "place_gently": self.place_gently,
            "descend_over": self.descend_over,
            "open_articulated": self.open_articulated,
            "body_position": self.body_position,
            "get_object_extents": self.get_object_extents,
            "get_region_boxes": self.get_region_boxes,
            "in_contact": self.in_contact,
            "plan_pinch": self.plan_pinch,
            "clearance_at": self.clearance_at,
            "clearance_many": self.clearance_many,
            "grasp_object": self.grasp_object,
            "carry_to": self.carry_to,
            "release_over": self.release_over,
            "place_in_region": self.place_in_region,
            "object_cloud": self.object_cloud,
            "fit_yaw": self.fit_yaw,
            "turn_held": self.turn_held,
            "place_on": self.place_on,
            "push_object": self.push_object,
            "place_on_region": self.place_on_region,
            "get_joint_info": self.get_joint_info,
            "turn_knob": self.turn_knob,
            "close_articulated": self.close_articulated,
            # promoted 2026-08-27 from the libero-90 solution preludes
            "get_hand_pose": self.get_hand_pose,
            "tcp": self.tcp,
            "lift_check": self.lift_check,
            "grasp_at": self.grasp_at,
            "rim_pick": self.rim_pick,
            "box_pick": self.box_pick,
            "top_pick": self.top_pick,
            "travel_height": self.travel_height,
            "place_at": self.place_at,
            "recentre_on": self.recentre_on,
            "articulation_outward": self.articulation_outward,
            "push_close": self.push_close,
            "press_face": self.press_face,
            "wait": self.wait,
            "get_object_parts": self.get_object_parts,
            "tilt_held": self.tilt_held,
            # round 4 (2026-08-27): flat insertion, site-frame alignment
            "insert_flat_into_region": self.insert_flat_into_region,
            "align_held_to_region": self.align_held_to_region,
            "palm_clearance": self.palm_clearance,
            # orientation helpers (pure functions, exposed so a program need not copy them)
            "frame_quat_wxyz": frame_quat_wxyz,
            "tool_down_quat_wxyz": tool_down_quat_wxyz,
            "tool_horizontal_quat_wxyz": tool_horizontal_quat_wxyz,
            "tool_pitched_quat_wxyz": tool_pitched_quat_wxyz,
            "quat_wxyz_to_mat": quat_wxyz_to_mat,
            "mat_to_quat_wxyz": mat_to_quat_wxyz,
        }
