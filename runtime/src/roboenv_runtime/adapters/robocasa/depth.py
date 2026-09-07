"""Metric depth transport matching the existing Complex RGB-D driver contract."""

from pathlib import Path

import numpy as np

# This is our output encoding, not the renderer's nonlinear [0, 1] depth buffer.
DEPTH_SCALE = 0.001


def write_depth(file, metres):
    """Write unsigned 16-bit big-endian PGM, millimetres; 0 denotes invalid."""
    depth = np.asarray(metres, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional metric image")
    valid = np.isfinite(depth) & (depth >= DEPTH_SCALE) & (depth <= 65535 * DEPTH_SCALE)
    raw = np.zeros(depth.shape, dtype=">u2")
    raw[valid] = np.rint(depth[valid] / DEPTH_SCALE).astype(">u2")
    height, width = depth.shape
    Path(file).write_bytes(f"P5\n{width} {height}\n65535\n".encode("ascii") + raw.tobytes())
