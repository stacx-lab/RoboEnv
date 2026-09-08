"""RoboCasa uses the shared metric depth encoding."""

from roboenv_runtime.bindings.backends.shared.depth import DEPTH_SCALE, write_depth

__all__ = ["DEPTH_SCALE", "write_depth"]
