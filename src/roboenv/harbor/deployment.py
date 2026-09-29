"""Immutable image/data deployments: no original executable source-tree mount."""

import re
from pathlib import Path, PurePosixPath

from roboenv.validation import json_loads


def load_deployment(path, backend):
    """Validate the same owner manifest for task export and local task execution."""
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


def native_service(path, task, *, portable=False):
    spec = load_deployment(path, task.backend)
    volumes = ["roboenv-run:/run/roboenv", "./sim:/opt/serve:ro"]
    for index, mount in enumerate(spec["mounts"]):
        variable = f"ROBOENV_DEPOT_{task.backend.upper()}_{index}"
        origin = (
            "${" + variable + ":?set the asset depot location}" if portable else mount["source"]
        )
        # Compose short syntax uses colons as delimiters. Local paths are never
        # expanded here in portable exports, but direct callers must be unambiguous.
        if not portable and any(character in origin for character in (":", "\n")):
            raise ValueError("data depot location cannot be represented safely in Compose")
        volumes.append(f"{origin}:{mount['target']}:ro")
    service = {
        "image": spec["image"],
        "pull_policy": "never",
        "command": ["/opt/serve/serve-native.sh"],
        "volumes": volumes,
        "environment": {
            "ROBOENV_SOCKET": "/run/roboenv/sim.sock",
            "ROBOENV_SESSION": "harbor",
            "ROBOENV_GENERATION": "0",
            "ROBOENV_RUN_DIR": "/run/roboenv/run",
            "ROBOENV_TASK": task.id,
            "ROBOENV_SEED": str(task.seed),
            "ROBOENV_BUDGET_MODE": task.budget_mode,
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            **spec.get("environment", {}),
        },
        "healthcheck": {
            "test": ["CMD", "/usr/bin/python3", "-I", "/opt/serve/healthcheck.py"],
            "interval": "5s",
            "timeout": "10s",
            "retries": 120,
            "start_period": "900s"
            if task.backend in {"behavior", "robodojo", "isaaclab"}
            else "120s",
        },
        "shm_size": "4gb",
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "capabilities": ["gpu"],
                            "device_ids": ["${ROBOENV_GPU:-0}"],
                        }
                    ]
                }
            }
        },
    }

    if task.backend == "fakeshell":
        service.pop("deploy", None)
    options = spec["runtime_options"]
    service["shm_size"] = options.get("shm_size", service["shm_size"])
    service["healthcheck"]["start_period"] = options.get(
        "start_period", service["healthcheck"]["start_period"],
    )
    return service
