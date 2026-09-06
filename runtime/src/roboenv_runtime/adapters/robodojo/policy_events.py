"""Allowlisted public-pose event detection for continuous RoboDojo policy plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

MAX_EVENT_ACTIONS = 110


def _number(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in {low}..{high}")
    result = float(value)
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be a finite number in {low}..{high}")
    return result


def _labels(value: Any, name: str, live: set[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of public object labels")
    if len(value) > 32:
        raise ValueError(f"{name} accepts at most 32 labels")
    result: list[str] = []
    for label in value:
        if not isinstance(label, str) or not label or label.strip() != label:
            raise ValueError(f"{name} entries must be non-empty exact strings")
        if label not in live:
            raise ValueError(f"{name} contains non-public or unknown label {label!r}")
        if label in result:
            raise ValueError(f"{name} contains duplicate label {label!r}")
        result.append(label)
    return tuple(result)


def validate_event_config(
    *,
    n_actions: Any,
    candidate_labels: Any,
    protected_labels: Any,
    live_labels: set[str],
    owner_rise_m: Any,
    owner_near_m: Any,
    owner_width_m: Any,
    owner_arm_margin_m: Any,
    owner_stable_actions: Any,
    owner_position_drift_m: Any,
    owner_rotation_drift_deg: Any,
    protected_displacement_m: Any,
    protected_rotation_deg: Any,
) -> dict[str, Any]:
    """Validate the complete and deliberately small event-condition allowlist."""
    if isinstance(n_actions, bool) or not isinstance(n_actions, int):
        raise ValueError(f"n_actions must be an integer in 1..{MAX_EVENT_ACTIONS}")
    if not 1 <= n_actions <= MAX_EVENT_ACTIONS:
        raise ValueError(f"n_actions must be an integer in 1..{MAX_EVENT_ACTIONS}")
    candidates = _labels(candidate_labels, "candidate_labels", live_labels)
    protected = _labels(protected_labels, "protected_labels", live_labels)
    if not candidates and not protected:
        raise ValueError("at least one candidate_labels or protected_labels entry is required")
    overlap = sorted(set(candidates) & set(protected))
    if overlap:
        raise ValueError(f"candidate_labels and protected_labels must be disjoint: {overlap}")
    if (
        isinstance(owner_stable_actions, bool)
        or not isinstance(owner_stable_actions, int)
        or not 1 <= owner_stable_actions <= 20
    ):
        raise ValueError("owner_stable_actions must be an integer in 1..20")
    return {
        "n_actions": n_actions,
        "candidate_labels": candidates,
        "protected_labels": protected,
        "owner_rise_m": _number("owner_rise_m", owner_rise_m, 0.0, 1.0),
        "owner_near_m": _number("owner_near_m", owner_near_m, 0.005, 1.0),
        "owner_width_m": _number("owner_width_m", owner_width_m, 0.001, 0.25),
        "owner_arm_margin_m": _number("owner_arm_margin_m", owner_arm_margin_m, 0.0, 0.5),
        "owner_stable_actions": owner_stable_actions,
        "owner_position_drift_m": _number(
            "owner_position_drift_m", owner_position_drift_m, 0.0, 0.25
        ),
        "owner_rotation_drift_deg": _number(
            "owner_rotation_drift_deg", owner_rotation_drift_deg, 0.0, 180.0
        ),
        "protected_displacement_m": _number(
            "protected_displacement_m", protected_displacement_m, 0.0, 1.0
        ),
        "protected_rotation_deg": _number(
            "protected_rotation_deg", protected_rotation_deg, 0.0, 180.0
        ),
    }


def pose_snapshot(
    get_pose: Callable[[str], Any], labels: tuple[str, ...]
) -> dict[str, dict[str, list[float]]]:
    """Best-effort finite public object poses; missing later poses become events."""
    result: dict[str, dict[str, list[float]]] = {}
    for label in labels:
        try:
            position, quaternion = get_pose(label)
            pos = np.asarray(position, dtype=float).reshape(-1)
            quat = np.asarray(quaternion, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(quat))
            if (
                pos.size != 3
                or quat.size != 4
                or not np.all(np.isfinite(pos))
                or not np.all(np.isfinite(quat))
                or norm < 1e-9
            ):
                continue
            result[label] = {
                "position": pos.tolist(),
                "quaternion": (quat / norm).tolist(),
            }
        except Exception:
            continue
    return result


def gripper_snapshot(
    get_pose: Callable[[str], Any], get_state: Callable[[str], Any]
) -> dict[str, dict[str, Any]]:
    """Finite public fingertip pose and jaw width for both canonical arms."""
    result: dict[str, dict[str, Any]] = {}
    for arm in ("left", "right"):
        try:
            state = get_state(arm)
            raw_pose = state.get("tip_pose") if isinstance(state, Mapping) else None
            vector = np.asarray(raw_pose if raw_pose is not None else get_pose(arm), dtype=float)
            vector = vector.reshape(-1)
            width = float(state["width"])
            quaternion = vector[3:]
            norm = float(np.linalg.norm(quaternion))
            if (
                vector.size != 7
                or not np.all(np.isfinite(vector))
                or not np.isfinite(width)
                or norm < 1e-9
            ):
                continue
            result[arm] = {
                "position": vector[:3].tolist(),
                "quaternion": (quaternion / norm).tolist(),
                "width": width,
            }
        except Exception:
            continue
    return result


def quaternion_angle(first: Any, second: Any) -> float:
    """Return shortest angular distance between finite wxyz quaternions."""
    lhs = np.asarray(first, dtype=float)
    rhs = np.asarray(second, dtype=float)
    dot = float(np.dot(lhs / np.linalg.norm(lhs), rhs / np.linalg.norm(rhs)))
    return float(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0)))


def _multiply(first: Any, second: Any) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(first, dtype=float)
    w2, x2, y2, z2 = np.asarray(second, dtype=float)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotation(quaternion: Any) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _object_in_hand(
    object_pose: Mapping[str, Any], hand_pose: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    hand_q = np.asarray(hand_pose["quaternion"], dtype=float)
    relative_position = _rotation(hand_q).T @ (
        np.asarray(object_pose["position"], dtype=float)
        - np.asarray(hand_pose["position"], dtype=float)
    )
    relative_q = _multiply(
        hand_q * np.asarray([1.0, -1.0, -1.0, -1.0]), object_pose["quaternion"]
    )
    relative_q /= max(float(np.linalg.norm(relative_q)), 1e-9)
    return relative_position, relative_q


class PolicyEventTracker:
    """Pure public-pose owner/protection detector evaluated after every action."""

    def __init__(
        self,
        config: Mapping[str, Any],
        object_poses: dict[str, dict[str, list[float]]],
        grippers: dict[str, dict[str, Any]],
    ) -> None:
        required = set(config["candidate_labels"]) | set(config["protected_labels"])
        missing = sorted(required - set(object_poses))
        if missing:
            raise ValueError(f"event labels lack a finite public pose: {missing}")
        if config["candidate_labels"] and set(grippers) != {"left", "right"}:
            raise RuntimeError("owner detection requires finite public state for both grippers")
        self.config = dict(config)
        self.initial = object_poses
        self.previous = object_poses
        self.previous_grippers = grippers
        self.streaks: dict[tuple[str, str], int] = {}

    def observe(
        self,
        object_poses: dict[str, dict[str, list[float]]],
        grippers: dict[str, dict[str, Any]],
        *,
        action_index: int,
    ) -> dict[str, Any] | None:
        """Return the first protected violation or stable owner witness."""
        for label in self.config["protected_labels"]:
            if label not in object_poses:
                return {
                    "kind": "protected_pose_unavailable",
                    "label": label,
                    "action_index": action_index,
                }
            initial, current = self.initial[label], object_poses[label]
            displacement = float(
                np.linalg.norm(
                    np.asarray(current["position"]) - np.asarray(initial["position"])
                )
            )
            rotation_deg = float(
                np.degrees(quaternion_angle(initial["quaternion"], current["quaternion"]))
            )
            if (
                displacement > self.config["protected_displacement_m"]
                or rotation_deg > self.config["protected_rotation_deg"]
            ):
                return {
                    "kind": "protected_violation",
                    "label": label,
                    "action_index": action_index,
                    "displacement_m": round(displacement, 6),
                    "rotation_deg": round(rotation_deg, 4),
                    "displacement_limit_m": self.config["protected_displacement_m"],
                    "rotation_limit_deg": self.config["protected_rotation_deg"],
                }

        for label in self.config["candidate_labels"]:
            current, previous = object_poses.get(label), self.previous.get(label)
            if current is None or previous is None or set(grippers) != {"left", "right"}:
                self.streaks = {
                    key: value for key, value in self.streaks.items() if key[0] != label
                }
                continue
            position = np.asarray(current["position"])
            distances = sorted(
                (float(np.linalg.norm(position - np.asarray(grippers[arm]["position"]))), arm)
                for arm in ("left", "right")
            )
            distance, arm = distances[0]
            arm_margin = distances[1][0] - distance
            rise = float(position[2] - self.initial[label]["position"][2])
            previous_hand = self.previous_grippers.get(arm)
            position_drift = rotation_drift = float("inf")
            if previous_hand is not None:
                old_relative = _object_in_hand(previous, previous_hand)
                new_relative = _object_in_hand(current, grippers[arm])
                position_drift = float(np.linalg.norm(new_relative[0] - old_relative[0]))
                rotation_drift = float(
                    np.degrees(quaternion_angle(old_relative[1], new_relative[1]))
                )
            passes = bool(
                rise >= self.config["owner_rise_m"]
                and distance <= self.config["owner_near_m"]
                and grippers[arm]["width"] <= self.config["owner_width_m"]
                and arm_margin >= self.config["owner_arm_margin_m"]
                and position_drift <= self.config["owner_position_drift_m"]
                and rotation_drift <= self.config["owner_rotation_drift_deg"]
            )
            for key in [key for key in self.streaks if key[0] == label and key[1] != arm]:
                self.streaks.pop(key, None)
            key = (label, arm)
            self.streaks[key] = self.streaks.get(key, 0) + 1 if passes else 0
            if self.streaks[key] >= self.config["owner_stable_actions"]:
                return {
                    "kind": "owner_witness",
                    "label": label,
                    "arm": arm,
                    "action_index": action_index,
                    "stable_actions": self.streaks[key],
                    "rise_m": round(rise, 6),
                    "distance_m": round(distance, 6),
                    "width_m": round(float(grippers[arm]["width"]), 6),
                    "arm_margin_m": round(arm_margin, 6),
                    "transform_position_drift_m": round(position_drift, 6),
                    "transform_rotation_drift_deg": round(rotation_drift, 4),
                }
        self.previous = object_poses
        self.previous_grippers = grippers
        return None
