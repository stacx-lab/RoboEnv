"""Immutable image/data deployments: no original executable source-tree mount."""

import re
from pathlib import Path, PurePosixPath

from roboenv.validation import json_loads


def load_deployment(path, backend):
    """Validate an owner manifest for a standalone simulator image."""
    path = Path(path).resolve()
    spec = json_loads(path.read_text())
    if (
        not isinstance(spec, dict)
        or spec.get("schema") != "roboenv/deployment/1"
        or spec.get("backend") != backend
    ):
        raise ValueError("deployment schema/backend does not match the task")
    if spec.get("legacy_tree_required") is not False:
        raise ValueError("native deployment must not depend on an original working tree")
    image = spec.get("image", "")
    if not isinstance(image, str) or not re.fullmatch(
        r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", image
    ):
        raise ValueError("build the image first; native deployments require an immutable digest")
    mounts = spec.get("mounts", [])
    if not isinstance(mounts, list):
        raise ValueError("deployment mounts must be an array")
    normalized, targets = [], []
    for index, mount in enumerate(mounts):
        if (
            not isinstance(mount, dict)
            or set(mount) != {"source", "target", "read_only"}
            or not isinstance(mount["source"], str)
            or not mount["source"]
        ):
            raise ValueError("deployment mounts need source, target and read_only")
        source = Path(mount["source"])
        source = (source if source.is_absolute() else path.parent / source).resolve()
        if (
            not source.exists()
            or source == Path("/")
            or (source / "pyproject.toml").is_file()
            or (source / "roboenv").is_dir()
        ):
            raise ValueError(f"mount {index} is not an independent existing data depot")
        target = mount["target"]
        if (
            not isinstance(target, str)
            or not target.startswith("/opt/roboenv/runtime/")
            or ".." in PurePosixPath(target).parts
            or PurePosixPath(target).as_posix() != target
            or any(character in target for character in (":", "\\", "$", "\x00", "\n"))
        ):
            raise ValueError("data mounts must be below the native runtime data prefix")
        location = PurePosixPath(target)
        if any(
            location == other or location in other.parents or other in location.parents
            for other in targets
        ):
            raise ValueError("data mount targets must not overlap")
        targets.append(location)
        if mount.get("read_only") is not True:
            raise ValueError("native input data mounts must be read-only")
        normalized.append({**mount, "source": str(source)})
    protected = {
        "ROBOENV_TASK",
        "ROBOENV_SEED",
        "ROBOENV_BUDGET_MODE",
        "ROBOENV_ROOT",
        "ROBOENV_TASKS",
        "ROBOENV_SERVER_ROBOT",
        "ROBOENV_SERVER_MANIFEST",
        "ROBOENV_SDK_PATH",
        "ROBOENV_SESSION",
        "ROBOENV_GENERATION",
        "ROBOENV_SOCKET",
        "ROBOENV_RUN_DIR",
        "ROBOENV_SOCKET_ROOT",
        "ROBOENV_ENDPOINT",
        "ROBOENV_HTTP_HOST",
        "ROBOENV_HTTP_PORT",
        "ROBOENV_PUBLISH_DEVICES",
        "ROBOENV_FRESH_SESSION",
    }
    extra = spec.get("environment", {})
    if (
        not isinstance(extra, dict)
        or protected & set(extra)
        or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in extra)
        or any(not isinstance(v, str) or "\x00" in v for v in extra.values())
    ):
        raise ValueError("deployment environment cannot override task identity or runtime roots")
    options = spec.get("runtime_options", {})
    if not isinstance(options, dict) or set(options) - {"shm_size", "start_period"}:
        raise ValueError("unsupported deployment runtime options")
    patterns = {"shm_size": r"[1-9][0-9]*(?:b|kb|mb|gb)",
                "start_period": r"[1-9][0-9]*(?:ms|s|m)"}
    if any(not isinstance(value, str) or not re.fullmatch(patterns[key], value)
           for key, value in options.items()):
        raise ValueError("invalid deployment runtime option")
    return {**spec, "mounts": normalized, "environment": extra, "runtime_options": options}


def deployment_contract(spec):
    """Portable deployment identity; local depot locations are deliberately excluded."""
    return {
        "backend": spec["backend"],
        "image": spec["image"],
        "mounts": [
            {"target": mount["target"], "read_only": mount["read_only"]} for mount in spec["mounts"]
        ],
        "environment": spec["environment"],
        "runtime_options": spec.get("runtime_options", {}),
    }
