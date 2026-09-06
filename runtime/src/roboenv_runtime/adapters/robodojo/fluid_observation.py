"""Public, scorer-agnostic observation helpers for RoboDojo fluids.

RoboDojo stores live particle coordinates on a USD PointInstancer in the fluid
object's local frame.  This module resolves a public fluid label and converts that
local cloud into the environment-relative world frame used by the rest of roboenv's
RoboDojo public observations.  It deliberately exposes neither particle identifiers
nor any containment or reward semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_PARTICLE_CAP = 4096
DEFAULT_PARTICLE_CAP = 512
_ORIGIN_ATOL = 1.0e-6

__all__ = [
    "DEFAULT_PARTICLE_CAP",
    "MAX_PARTICLE_CAP",
    "fluid_particle_observation",
    "observe_fluid_particles",
]


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors and array-like values without retaining device state."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    """Return a finite float64 vector with one exact shape."""
    try:
        array = _as_numpy(value).astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _particle_array(value: Any) -> np.ndarray:
    """Validate a non-empty finite ``(N, 3)`` local particle array."""
    try:
        array = _as_numpy(value).astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("particle positions must be numeric") from exc
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"particle positions must have shape (N, 3), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("particle positions must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("particle positions must contain only finite values")
    return array


def _validate_cap(cap: Any) -> int:
    """Validate the public output cap and reject ambiguous coercions."""
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)):
        raise TypeError("cap must be an integer")
    cap_int = int(cap)
    if cap_int <= 0:
        raise ValueError("cap must be positive")
    if cap_int > MAX_PARTICLE_CAP:
        raise ValueError(f"cap must not exceed {MAX_PARTICLE_CAP}")
    return cap_int


def _rotation_wxyz(quaternion: Any) -> np.ndarray:
    """Convert one finite non-zero WXYZ quaternion to a rotation matrix."""
    quat = _finite_vector(quaternion, name="initial quaternion", size=4)
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("initial quaternion must have non-zero norm")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _deterministic_sample(points: np.ndarray, cap: int) -> np.ndarray:
    """Return an input-order-invariant, evenly spread sample of a point cloud."""
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    if len(ordered) <= cap:
        return ordered

    # Midpoints of equal-rank bins avoid the endpoint bias of linspace and are
    # unique whenever cap < len(ordered).
    ranks = np.floor((np.arange(cap, dtype=np.float64) + 0.5) * len(ordered) / cap)
    return ordered[ranks.astype(np.int64)]


def fluid_particle_observation(
    local_positions: Any,
    initial_position: Any,
    initial_quaternion_wxyz: Any,
    env_origin: Any,
    *,
    cap: int = DEFAULT_PARTICLE_CAP,
) -> dict[str, Any]:
    """Build a bounded public observation from live local fluid particles.

    The returned ``points`` are in RoboDojo's public ``env_world`` frame: axes match
    Isaac world axes, while the selected environment's origin is subtracted.  Summary
    statistics are computed over the full validated cloud, not only the sample.

    Args:
        local_positions: Live PointInstancer positions with shape ``(N, 3)``.
        initial_position: Fluid PointInstancer translation in environment coordinates.
        initial_quaternion_wxyz: Fluid PointInstancer orientation in WXYZ order.
        env_origin: Selected Isaac environment's stage-world origin.
        cap: Maximum number of public points, bounded by ``MAX_PARTICLE_CAP``.

    Returns:
        A JSON-compatible observation with a deterministic point cloud and full-cloud
        count, centroid, and axis-aligned bounding box.

    Raises:
        TypeError: If numeric inputs or ``cap`` have invalid types.
        ValueError: If shapes, finiteness, quaternion norm, or ``cap`` are invalid.
    """
    particles = _particle_array(local_positions)
    initial = _finite_vector(initial_position, name="initial position", size=3)
    origin = _finite_vector(env_origin, name="environment origin", size=3)
    cap_int = _validate_cap(cap)
    rotation = _rotation_wxyz(initial_quaternion_wxyz)

    environment_points = particles @ rotation.T + initial

    # Spell out the Isaac stage-world conversion and the public env-origin removal.
    # This is intentionally algebraically neutral: it makes the frame boundary
    # explicit and keeps the public coordinates consistent with object/TCP queries.
    stage_world_points = environment_points + origin
    public_points = stage_world_points - origin
    if not np.isfinite(public_points).all():
        raise ValueError("transformed particle positions must be finite")

    sample = _deterministic_sample(public_points, cap_int)
    minimum = np.min(public_points, axis=0)
    maximum = np.max(public_points, axis=0)
    centroid = np.mean(public_points, axis=0)
    count = len(public_points)
    return {
        "frame": "env_world",
        "count": count,
        "sample_count": len(sample),
        "downsampled": count > cap_int,
        "points": sample.tolist(),
        "centroid": centroid.tolist(),
        "aabb": {"min": minimum.tolist(), "max": maximum.tolist()},
    }


def _validate_env_index(scene_manager: Any, env_idx: Any) -> int:
    """Validate an environment index against the live origins registry."""
    if isinstance(env_idx, (bool, np.bool_)) or not isinstance(env_idx, (int, np.integer)):
        raise TypeError("env_idx must be an integer")
    index = int(env_idx)
    if index < 0:
        raise ValueError("env_idx must be non-negative")
    origins = getattr(scene_manager, "env_origins", None)
    if origins is None:
        raise RuntimeError("scene manager does not expose environment origins")
    try:
        _ = origins[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise IndexError(f"env_idx {index} is unavailable") from exc
    return index


def observe_fluid_particles(
    scene_manager: Any,
    label: str,
    *,
    env_idx: int = 0,
    cap: int = DEFAULT_PARTICLE_CAP,
) -> dict[str, Any]:
    """Resolve a public RoboDojo fluid label and sample its live particle cloud.

    The helper accepts only a public layout label.  It checks the layout's declared
    instance type before accessing the scene object and refuses missing or mismatched
    environment origins.  No internal instance name or PointInstancer path is returned.

    Args:
        scene_manager: Live RoboDojo ``SceneManager``-compatible object.
        label: Non-empty public label declared by the task layout.
        env_idx: Environment index to observe.
        cap: Maximum number of returned points.

    Returns:
        The JSON-compatible dictionary from :func:`fluid_particle_observation`.
    """
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if not label.strip():
        raise ValueError("label must not be empty")
    cap_int = _validate_cap(cap)
    index = _validate_env_index(scene_manager, env_idx)

    layout = getattr(scene_manager, "layout_manager", None)
    if layout is None:
        raise RuntimeError("scene manager does not expose a layout manager")
    try:
        instance_name = layout.get_instance_name(label=label, env_idx=index)
    except KeyError as exc:
        raise KeyError(f"unknown public label {label!r} in environment {index}") from exc
    if not instance_name:
        raise KeyError(f"unknown public label {label!r} in environment {index}")

    try:
        instance_type = layout.instance_type_by_env[index][instance_name]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("layout instance type registry is unavailable") from exc
    if str(instance_type).lower() != "fluid":
        raise TypeError(f"public label {label!r} does not refer to a fluid")

    try:
        fluid = layout.get_scene_object(env_idx=index, inst_name=instance_name)
    except KeyError as exc:
        raise RuntimeError(f"fluid for public label {label!r} is unavailable") from exc
    if fluid is None:
        raise RuntimeError(f"fluid for public label {label!r} is unavailable")

    sampler = getattr(fluid, "get_particle_positions", None)
    if not callable(sampler):
        raise RuntimeError("fluid object does not expose live particle positions")
    raw_sample = sampler()
    if not isinstance(raw_sample, (tuple, list)) or not raw_sample or raw_sample[0] is None:
        raise RuntimeError("fluid particle sampler returned an invalid payload")

    if not hasattr(fluid, "init_pos") or not hasattr(fluid, "init_ori"):
        raise RuntimeError("fluid object does not expose its PointInstancer transform")
    if not hasattr(fluid, "env_origin"):
        raise RuntimeError("fluid object does not expose its environment origin")

    scene_origin = _finite_vector(
        scene_manager.env_origins[index], name="scene environment origin", size=3
    )
    fluid_origin = _finite_vector(fluid.env_origin, name="fluid environment origin", size=3)
    if not np.allclose(scene_origin, fluid_origin, rtol=0.0, atol=_ORIGIN_ATOL):
        raise ValueError("fluid and scene environment origins disagree")

    return fluid_particle_observation(
        raw_sample[0],
        fluid.init_pos,
        fluid.init_ori,
        scene_origin,
        cap=cap_int,
    )
