"""Owner-side recording configuration; importing the SDK needs no renderer."""

import math
import os
from collections.abc import Mapping
from pathlib import Path


def recording_config(value):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"fps", "views", "directory"}:
        raise ValueError("recording must be a mapping with optional fps, views and directory")
    fps = value.get("fps", 10)
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError("recording fps must be a positive finite number")
    views = value.get("views")
    if views is not None and (not isinstance(views, list) or not views
                              or any(not isinstance(view, dict) for view in views)):
        raise ValueError("recording views must be a nonempty list of view objects")
    result = {"fps": fps, "views": views}
    if "directory" in value:
        directory = value["directory"]
        if not isinstance(directory, (str, os.PathLike)) or not os.fspath(directory):
            raise ValueError("recording directory must be a nonempty path")
        path = Path(directory).expanduser().resolve()
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError("recording directory must be new or empty")
        result["directory"] = str(path)
    return result
