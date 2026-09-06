"""Simulator-independent held-object alignment control.

The controller in this module deliberately knows nothing about RoboDojo tasks,
scorers, object labels, or simulator state.  Callers provide public/current hand and
object poses plus two narrow callbacks: a plan-only IK probe and an executor that
returns fresh poses after one bounded command.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

ArrayLike: TypeAlias = Sequence[float] | np.ndarray

AlignmentStatus = Literal[
    "success",
    "already_aligned",
    "invalid_input",
    "ik_unreachable",
    "probe_failed",
    "returned_wrist_rejected",
    "branch_jump",
    "wound_joint",
    "execution_failed",
    "held_lost",
    "no_progress",
    "step_limit",
]

PoseTuple = tuple[float, float, float, float, float, float, float]
QuaternionTuple = tuple[float, float, float, float]
VectorTuple = tuple[float, float, float]


@dataclass(frozen=True)
class HeldAlignmentConfig:
    """Safety and convergence limits for :func:`align_held_current_posture`.

    ``acceptance_tolerance_deg`` is supplied by the consumer; the helper never reads a
    scorer.  The default ten degrees is useful for common axis-alignment consumers,
    while the legacy ``align_held`` wrapper should pass two degrees to preserve its
    current contract.
    """

    max_object_step_deg: float = 10.0
    max_steps: int = 18
    acceptance_tolerance_deg: float = 10.0
    returned_wrist_tolerance_deg: float = 10.0
    max_branch_jump_rad: float = 1.2
    wound_margin_rad: float = 1.2
    roll_joint_index: int = -1
    joint_limits: tuple[tuple[float, float], ...] | None = None
    anchor_position_tolerance_m: float = 0.02
    anchor_orientation_tolerance_deg: float = 8.0
    min_progress_deg: float = 0.25
    max_non_progress_steps: int = 2
    max_staging_translation_m: float = 0.12


@dataclass(frozen=True)
class IKProbeRequest:
    """One current-branch, plan-only IK request."""

    step_index: int
    hand_pose: PoseTuple
    desired_object_quaternion: QuaternionTuple
    seed_joints: tuple[float, ...]


@dataclass(frozen=True)
class IKCandidate:
    """Typed result returned by the plan-only IK callback.

    ``returned_hand_quaternion`` is the wrist orientation the solver would actually
    execute.  It may be a small tilt fallback rather than the requested orientation.
    """

    reachable: bool
    joints: tuple[float, ...] | None = None
    returned_hand_quaternion: QuaternionTuple | None = None
    reason: str = ""


@dataclass(frozen=True)
class AlignmentCommand:
    """One accepted, bounded command passed to the executor callback."""

    step_index: int
    hand_pose: PoseTuple
    requested_hand_quaternion: QuaternionTuple
    desired_object_quaternion: QuaternionTuple
    predicted_object_quaternion: QuaternionTuple
    joints: tuple[float, ...]
    used_returned_wrist: bool


@dataclass(frozen=True)
class ExecutionObservation:
    """Fresh public state returned after executing one alignment command."""

    ok: bool
    hand_pose: PoseTuple
    object_pose: PoseTuple
    joints: tuple[float, ...]
    reason: str = ""


@dataclass(frozen=True)
class AlignmentStep:
    """Auditable measurements for one attempted wrist step."""

    step_index: int
    residual_before_deg: float
    residual_after_deg: float
    requested_step_deg: float
    returned_wrist_deviation_deg: float
    branch_distance_rad: float
    anchor_position_error_m: float
    anchor_orientation_error_deg: float
    used_returned_wrist: bool


@dataclass(frozen=True)
class WorkspaceStagingSuggestion:
    """Translation-only staging pose that preserves the current wrist and hold anchor."""

    hand_pose: PoseTuple
    predicted_object_pose: PoseTuple
    translation_m: float
    reason: str


@dataclass(frozen=True)
class HeldAlignmentOutcome:
    """Typed terminal outcome; no exception or scorer result is used as control flow."""

    status: AlignmentStatus
    success: bool
    held: bool
    reason: str
    steps_executed: int
    object_residual_deg: float | None
    anchor_position_error_m: float | None
    anchor_orientation_error_deg: float | None
    used_returned_wrist: bool
    final_hand_pose: PoseTuple | None
    final_object_pose: PoseTuple | None
    final_joints: tuple[float, ...] | None
    history: tuple[AlignmentStep, ...]
    staging: WorkspaceStagingSuggestion | None


IKProbeCallback = Callable[[IKProbeRequest], IKCandidate]
ExecuteCallback = Callable[[AlignmentCommand], ExecutionObservation]


def _normalize_quaternion(value: ArrayLike, name: str) -> np.ndarray:
    quat: np.ndarray = np.asarray(value, dtype=float).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        raise ValueError(f"{name} must contain four finite WXYZ values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError(f"{name} is degenerate")
    return quat / norm


def _pose(value: ArrayLike, name: str) -> tuple[np.ndarray, np.ndarray]:
    pose: np.ndarray = np.asarray(value, dtype=float).reshape(-1)
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain seven finite XYZ+WXYZ values")
    return pose[:3].copy(), _normalize_quaternion(pose[3:7], f"{name} quaternion")


def _joints(value: ArrayLike, name: str) -> np.ndarray:
    joints: np.ndarray = np.asarray(value, dtype=float).reshape(-1)
    if joints.size == 0 or not np.all(np.isfinite(joints)):
        raise ValueError(f"{name} must contain at least one finite joint value")
    return joints


def _quat_mul(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj(quat: ArrayLike) -> np.ndarray:
    value: np.ndarray = np.asarray(quat, dtype=float)
    return np.array([value[0], -value[1], -value[2], -value[3]]) / float(value @ value)


def _quat_angle_deg(left: ArrayLike, right: ArrayLike) -> float:
    a = _normalize_quaternion(left, "left quaternion")
    b = _normalize_quaternion(right, "right quaternion")
    dot = min(1.0, abs(float(a @ b)))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _slerp(start: ArrayLike, target: ArrayLike, fraction: float) -> np.ndarray:
    q0 = _normalize_quaternion(start, "slerp start")
    q1 = _normalize_quaternion(target, "slerp target")
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        value = q0 + float(fraction) * (q1 - q0)
        return value / np.linalg.norm(value)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    value = (
        np.sin((1.0 - float(fraction)) * theta) * q0
        + np.sin(float(fraction) * theta) * q1
    ) / np.sin(theta)
    return value / np.linalg.norm(value)


def _quat_to_matrix(quat: ArrayLike) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quat, "matrix quaternion")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _as_quaternion_tuple(value: ArrayLike) -> QuaternionTuple:
    quat = _normalize_quaternion(value, "quaternion")
    return tuple(float(item) for item in quat)  # type: ignore[return-value]


def _as_vector_tuple(value: ArrayLike) -> VectorTuple:
    vec: np.ndarray = np.asarray(value, dtype=float).reshape(3)
    return tuple(float(item) for item in vec)  # type: ignore[return-value]


def _as_pose_tuple(position: ArrayLike, quaternion: ArrayLike) -> PoseTuple:
    values = (*_as_vector_tuple(position), *_as_quaternion_tuple(quaternion))
    return values


def _relative_anchor(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = _quat_to_matrix(hand_quaternion).T @ (object_position - hand_position)
    quaternion = _quat_mul(_quat_conj(hand_quaternion), object_quaternion)
    return position, quaternion / np.linalg.norm(quaternion)


def _anchor_error(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    anchor_position: np.ndarray,
    anchor_quaternion: np.ndarray,
) -> tuple[float, float]:
    current_position, current_quaternion = _relative_anchor(
        hand_position,
        hand_quaternion,
        object_position,
        object_quaternion,
    )
    return (
        float(np.linalg.norm(current_position - anchor_position)),
        _quat_angle_deg(current_quaternion, anchor_quaternion),
    )


def _roll_index(index: int, dof: int) -> int:
    resolved = int(index) if int(index) >= 0 else dof + int(index)
    if not 0 <= resolved < dof:
        raise ValueError(f"roll_joint_index {index} is outside {dof} joints")
    return resolved


def _branch_distance(to_joints: np.ndarray, from_joints: np.ndarray, roll: int) -> float:
    if to_joints.shape != from_joints.shape:
        raise ValueError("IK and current joint vectors have different lengths")
    delta = np.arctan2(np.sin(to_joints - from_joints), np.cos(to_joints - from_joints))
    if delta.size > 1:
        delta = np.delete(delta, roll)
    return float(np.max(np.abs(delta)))


def _principal_range_ok(joints: np.ndarray, config: HeldAlignmentConfig, roll: int) -> bool:
    limits: np.ndarray | None = None
    if config.joint_limits is not None:
        limits = np.asarray(config.joint_limits, dtype=float).reshape(-1, 2)
        if limits.shape[0] != joints.size or not np.all(np.isfinite(limits)):
            raise ValueError("joint_limits must be finite and match the joint vector")
    threshold = np.pi + float(config.wound_margin_rad)
    for index, value in enumerate(joints):
        if index == roll or abs(float(value)) <= threshold:
            continue
        if limits is None:
            return False
        low, high = (float(item) for item in limits[index])
        for turns in (-2, -1, 1, 2):
            alias = float(value) - 2.0 * np.pi * turns
            if low <= alias <= high and abs(alias) < abs(float(value)):
                return False
    return True


def _validate_config(config: HeldAlignmentConfig) -> None:
    positive = {
        "max_object_step_deg": config.max_object_step_deg,
        "max_branch_jump_rad": config.max_branch_jump_rad,
        "wound_margin_rad": config.wound_margin_rad,
        "anchor_position_tolerance_m": config.anchor_position_tolerance_m,
        "anchor_orientation_tolerance_deg": config.anchor_orientation_tolerance_deg,
        "max_staging_translation_m": config.max_staging_translation_m,
    }
    if any(not np.isfinite(value) or float(value) <= 0.0 for value in positive.values()):
        raise ValueError("alignment distance, angle, and branch limits must be finite and positive")
    nonnegative = (
        config.acceptance_tolerance_deg,
        config.returned_wrist_tolerance_deg,
        config.min_progress_deg,
    )
    if any(not np.isfinite(value) or float(value) < 0.0 for value in nonnegative):
        raise ValueError("alignment tolerances must be finite and non-negative")
    if int(config.max_steps) < 1 or int(config.max_non_progress_steps) < 1:
        raise ValueError("max_steps and max_non_progress_steps must be positive")


def _staging_suggestion(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
    anchor_position: np.ndarray,
    anchor_quaternion: np.ndarray,
    workspace_center: Sequence[float] | None,
    config: HeldAlignmentConfig,
) -> WorkspaceStagingSuggestion | None:
    if workspace_center is None:
        return None
    center: np.ndarray = np.asarray(workspace_center, dtype=float).reshape(-1)
    if center.size != 3 or not np.all(np.isfinite(center)):
        return None
    delta = center - hand_position
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return None
    translation = min(distance, float(config.max_staging_translation_m))
    staged_hand = hand_position + delta / distance * translation
    staged_object = staged_hand + _quat_to_matrix(hand_quaternion) @ anchor_position
    staged_object_quaternion = _quat_mul(hand_quaternion, anchor_quaternion)
    return WorkspaceStagingSuggestion(
        hand_pose=_as_pose_tuple(staged_hand, hand_quaternion),
        predicted_object_pose=_as_pose_tuple(staged_object, staged_object_quaternion),
        translation_m=translation,
        reason=(
            "translation-only move toward the caller-provided workspace center; "
            "preserve the current wrist and re-probe before executing"
        ),
    )


def align_held_current_posture(
    current_hand_pose: Sequence[float],
    current_object_pose: Sequence[float],
    target_object_quaternion: Sequence[float],
    current_joints: Sequence[float],
    probe_ik: IKProbeCallback,
    execute: ExecuteCallback,
    *,
    config: HeldAlignmentConfig | None = None,
    workspace_center: Sequence[float] | None = None,
) -> HeldAlignmentOutcome:
    """Align a held object with bounded current-posture wrist commands.

    The hand position is held at its fresh value while object orientation advances by
    at most ``max_object_step_deg`` of commanded slerp per iteration.  Each IK result is
    checked for winding and current-branch continuity before execution.  A returned
    wrist fallback is accepted only when its predicted object frame remains within the
    caller's explicit, scorer-independent tolerance.  Fresh hand/object poses are then
    checked against the initial SE(3) hold anchor after every command.

    All expected failures return :class:`HeldAlignmentOutcome`; callback exceptions are
    contained and reported as typed ``probe_failed`` or ``execution_failed`` outcomes.
    """
    cfg = config or HeldAlignmentConfig()
    history: list[AlignmentStep] = []
    used_returned_wrist = False
    hand_position: np.ndarray | None = None
    hand_quaternion: np.ndarray | None = None
    object_position: np.ndarray | None = None
    object_quaternion: np.ndarray | None = None
    joints: np.ndarray | None = None
    anchor_position: np.ndarray | None = None
    anchor_quaternion: np.ndarray | None = None
    residual: float | None = None
    anchor_position_error: float | None = None
    anchor_orientation_error: float | None = None

    def finish(
        status: AlignmentStatus,
        reason: str,
        *,
        held: bool = True,
        suggest_stage: bool = False,
    ) -> HeldAlignmentOutcome:
        staging = None
        if (
            suggest_stage
            and hand_position is not None
            and hand_quaternion is not None
            and anchor_position is not None
            and anchor_quaternion is not None
        ):
            staging = _staging_suggestion(
                hand_position,
                hand_quaternion,
                anchor_position,
                anchor_quaternion,
                workspace_center,
                cfg,
            )
        return HeldAlignmentOutcome(
            status=status,
            success=status in ("success", "already_aligned"),
            held=held,
            reason=reason,
            steps_executed=len(history),
            object_residual_deg=residual,
            anchor_position_error_m=anchor_position_error,
            anchor_orientation_error_deg=anchor_orientation_error,
            used_returned_wrist=used_returned_wrist,
            final_hand_pose=(
                None
                if hand_position is None or hand_quaternion is None
                else _as_pose_tuple(hand_position, hand_quaternion)
            ),
            final_object_pose=(
                None
                if object_position is None or object_quaternion is None
                else _as_pose_tuple(object_position, object_quaternion)
            ),
            final_joints=(
                None if joints is None else tuple(float(value) for value in joints)
            ),
            history=tuple(history),
            staging=staging,
        )

    try:
        _validate_config(cfg)
        hand_position, hand_quaternion = _pose(current_hand_pose, "current_hand_pose")
        object_position, object_quaternion = _pose(current_object_pose, "current_object_pose")
        target_quaternion = _normalize_quaternion(
            target_object_quaternion,
            "target_object_quaternion",
        )
        joints = _joints(current_joints, "current_joints")
        roll = _roll_index(cfg.roll_joint_index, joints.size)
        anchor_position, anchor_quaternion = _relative_anchor(
            hand_position,
            hand_quaternion,
            object_position,
            object_quaternion,
        )
        residual = _quat_angle_deg(object_quaternion, target_quaternion)
    except (TypeError, ValueError) as exc:
        return finish("invalid_input", str(exc), held=False)

    if residual <= float(cfg.acceptance_tolerance_deg):
        return finish("already_aligned", "object orientation is already within tolerance")

    non_progress_steps = 0
    for step_index in range(1, int(cfg.max_steps) + 1):
        residual_before = _quat_angle_deg(object_quaternion, target_quaternion)
        fraction = min(1.0, float(cfg.max_object_step_deg) / max(residual_before, 1e-9))
        desired_object = _slerp(object_quaternion, target_quaternion, fraction)
        requested_hand = _quat_mul(desired_object, _quat_conj(anchor_quaternion))
        requested_hand /= np.linalg.norm(requested_hand)
        requested_step = _quat_angle_deg(object_quaternion, desired_object)
        request = IKProbeRequest(
            step_index=step_index,
            hand_pose=_as_pose_tuple(hand_position, requested_hand),
            desired_object_quaternion=_as_quaternion_tuple(desired_object),
            seed_joints=tuple(float(value) for value in joints),
        )
        try:
            candidate = probe_ik(request)
        except Exception as exc:  # callback boundary: convert to a typed outcome
            return finish(
                "probe_failed",
                f"IK probe raised {type(exc).__name__}: {exc}",
                suggest_stage=True,
            )
        if not isinstance(candidate, IKCandidate):
            return finish(
                "probe_failed",
                "IK probe must return IKCandidate",
                suggest_stage=True,
            )
        if not candidate.reachable:
            return finish(
                "ik_unreachable",
                candidate.reason or "current-posture IK probe found no solution",
                suggest_stage=True,
            )
        try:
            if candidate.joints is None:
                raise ValueError("reachable IK candidate omitted joints")
            candidate_joints = _joints(candidate.joints, "IK candidate joints")
            if candidate_joints.size != joints.size:
                raise ValueError("IK candidate and current joint vectors have different lengths")
            if not _principal_range_ok(candidate_joints, cfg, roll):
                return finish(
                    "wound_joint",
                    "IK candidate has a non-roll joint with a closer in-limit 2pi alias",
                    suggest_stage=True,
                )
            branch_distance = _branch_distance(candidate_joints, joints, roll)
        except (TypeError, ValueError) as exc:
            return finish("probe_failed", str(exc), suggest_stage=True)
        if branch_distance > float(cfg.max_branch_jump_rad):
            return finish(
                "branch_jump",
                (
                    f"IK candidate branch distance {branch_distance:.3f} rad exceeds "
                    f"{cfg.max_branch_jump_rad:.3f} rad"
                ),
                suggest_stage=True,
            )

        try:
            returned_hand = (
                requested_hand
                if candidate.returned_hand_quaternion is None
                else _normalize_quaternion(
                    candidate.returned_hand_quaternion,
                    "returned_hand_quaternion",
                )
            )
        except (TypeError, ValueError) as exc:
            return finish("probe_failed", str(exc), suggest_stage=True)
        returned_deviation = _quat_angle_deg(returned_hand, requested_hand)
        predicted_object = _quat_mul(returned_hand, anchor_quaternion)
        predicted_object /= np.linalg.norm(predicted_object)
        predicted_waypoint_error = _quat_angle_deg(predicted_object, desired_object)
        if (
            returned_deviation > float(cfg.returned_wrist_tolerance_deg) + 1e-7
            or predicted_waypoint_error > float(cfg.returned_wrist_tolerance_deg) + 1e-7
        ):
            return finish(
                "returned_wrist_rejected",
                (
                    f"returned wrist changes the predicted object waypoint by "
                    f"{predicted_waypoint_error:.3f} deg"
                ),
                suggest_stage=True,
            )
        returned_used_this_step = returned_deviation > 1e-4
        used_returned_wrist = used_returned_wrist or returned_used_this_step
        command = AlignmentCommand(
            step_index=step_index,
            hand_pose=_as_pose_tuple(hand_position, returned_hand),
            requested_hand_quaternion=_as_quaternion_tuple(requested_hand),
            desired_object_quaternion=_as_quaternion_tuple(desired_object),
            predicted_object_quaternion=_as_quaternion_tuple(predicted_object),
            joints=tuple(float(value) for value in candidate_joints),
            used_returned_wrist=returned_used_this_step,
        )
        try:
            observation = execute(command)
        except Exception as exc:  # callback boundary: convert to a typed outcome
            return finish(
                "execution_failed",
                f"executor raised {type(exc).__name__}: {exc}",
                suggest_stage=True,
            )
        if not isinstance(observation, ExecutionObservation):
            return finish(
                "execution_failed",
                "executor must return ExecutionObservation",
                suggest_stage=True,
            )
        if not observation.ok:
            return finish(
                "execution_failed",
                observation.reason or "executor reported failure",
                suggest_stage=True,
            )
        try:
            next_hand_position, next_hand_quaternion = _pose(
                observation.hand_pose,
                "executed hand pose",
            )
            next_object_position, next_object_quaternion = _pose(
                observation.object_pose,
                "executed object pose",
            )
            next_joints = _joints(observation.joints, "executed joints")
            if next_joints.size != joints.size:
                raise ValueError("executed and current joint vectors have different lengths")
            actual_branch_distance = _branch_distance(next_joints, joints, roll)
            actual_wound = not _principal_range_ok(next_joints, cfg, roll)
            anchor_position_error, anchor_orientation_error = _anchor_error(
                next_hand_position,
                next_hand_quaternion,
                next_object_position,
                next_object_quaternion,
                anchor_position,
                anchor_quaternion,
            )
            residual_after = _quat_angle_deg(next_object_quaternion, target_quaternion)
        except (TypeError, ValueError) as exc:
            return finish("execution_failed", str(exc), suggest_stage=True)

        hand_position, hand_quaternion = next_hand_position, next_hand_quaternion
        object_position, object_quaternion = next_object_position, next_object_quaternion
        joints = next_joints
        residual = residual_after
        history.append(
            AlignmentStep(
                step_index=step_index,
                residual_before_deg=residual_before,
                residual_after_deg=residual_after,
                requested_step_deg=requested_step,
                returned_wrist_deviation_deg=returned_deviation,
                branch_distance_rad=actual_branch_distance,
                anchor_position_error_m=anchor_position_error,
                anchor_orientation_error_deg=anchor_orientation_error,
                used_returned_wrist=returned_used_this_step,
            )
        )
        if (
            anchor_position_error > float(cfg.anchor_position_tolerance_m)
            or anchor_orientation_error > float(cfg.anchor_orientation_tolerance_deg)
        ):
            return finish(
                "held_lost",
                (
                    f"object-hand anchor drifted {anchor_position_error:.4f} m and "
                    f"{anchor_orientation_error:.2f} deg"
                ),
                held=False,
            )
        if actual_wound:
            return finish(
                "wound_joint",
                "executed state contains a wound non-roll joint",
                suggest_stage=True,
            )
        if actual_branch_distance > float(cfg.max_branch_jump_rad):
            return finish(
                "branch_jump",
                (
                    f"executed branch distance {actual_branch_distance:.3f} rad exceeds "
                    f"{cfg.max_branch_jump_rad:.3f} rad"
                ),
                suggest_stage=True,
            )
        if residual_after <= float(cfg.acceptance_tolerance_deg):
            return finish("success", "object orientation reached the explicit tolerance")
        progress = residual_before - residual_after
        non_progress_steps = (
            non_progress_steps + 1
            if progress < float(cfg.min_progress_deg)
            else 0
        )
        if non_progress_steps >= int(cfg.max_non_progress_steps):
            return finish(
                "no_progress",
                (
                    f"orientation contracted by less than {cfg.min_progress_deg:.3f} deg "
                    f"for {non_progress_steps} consecutive steps"
                ),
                suggest_stage=True,
            )

    return finish(
        "step_limit",
        f"alignment did not converge within {cfg.max_steps} bounded steps",
        suggest_stage=True,
    )


__all__ = [
    "AlignmentCommand",
    "AlignmentStatus",
    "AlignmentStep",
    "ExecuteCallback",
    "ExecutionObservation",
    "HeldAlignmentConfig",
    "HeldAlignmentOutcome",
    "IKCandidate",
    "IKProbeCallback",
    "IKProbeRequest",
    "WorkspaceStagingSuggestion",
    "align_held_current_posture",
]
