"""Joint-action validation and durable journalling for RoboDojo device writes.

These helpers were shared with a policy-rollout manager that no longer exists;
they are ordinary action handling and stay with the backend that uses them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

# Upper bound on one bounded hold request, so a mistake cannot consume an episode.
MAX_HOLD_ACTIONS = 2_000

JOINT_ACTION_FIELDS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


def _validated_joint_action(action: Any) -> dict[str, np.ndarray]:
    """Normalize finite flat or structured dual-X5 joint actions.

    The native EvalEnv mapping and one packed 14-vector are both accepted; Isaac
    always receives the canonical mapping.
    """

    if isinstance(action, Mapping):
        expected = {name for name, _ in JOINT_ACTION_FIELDS}
        actual = set(action)
        if actual != expected:
            raise RuntimeError(
                "structured joint action keys must be exactly "
                f"{sorted(expected)}, received {sorted(map(str, actual))}"
            )
        normalized: dict[str, np.ndarray] = {}
        for name, dimension in JOINT_ACTION_FIELDS:
            try:
                value = np.asarray(action[name], dtype=float)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"joint action field {name!r} must be numeric") from exc
            if value.shape != (dimension,):
                raise RuntimeError(
                    f"joint action field {name!r} must have shape "
                    f"({dimension},), received {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"joint action field {name!r} contains NaN or infinity")
            normalized[name] = value
        return normalized

    try:
        vector = np.asarray(action, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "joint action must be a packed 14-vector or native joint mapping"
        ) from exc
    if vector.shape != (14,):
        raise RuntimeError(f"packed joint action must have shape (14,), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("packed joint action contains NaN or infinity")
    normalized = {}
    offset = 0
    for name, dimension in JOINT_ACTION_FIELDS:
        normalized[name] = vector[offset : offset + dimension]
        offset += dimension
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_actions_sha256(actions: list[Any]) -> str:
    """Hash canonical finite action JSON for durable inline provenance."""

    payload = json.dumps(
        _jsonable(actions),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
