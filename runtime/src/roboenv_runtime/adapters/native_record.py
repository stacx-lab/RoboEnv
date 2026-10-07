"""Private native command records and strict, backend-independent comparisons."""

import copy
import hashlib
import json
import math
from importlib import metadata, resources


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def asset_identity(roots):
    """Hash actual native asset bytes; paths are relative to named asset roots."""
    from roboenv.provisioning.assets import inventory

    manifests = {}
    for name, directory in sorted(roots.items()):
        entries = inventory(directory)
        files = [entry for entry in entries if entry["kind"] == "file"]
        if not files:
            raise ValueError("Native asset root is empty")
        manifests[name] = {
            "files": len(files),
            "bytes": sum(v["bytes"] for v in files),
            "sha256": digest(entries),
        }
    return manifests


# These are independently owned implementations, not shared helpers. Everything
# outside this explicit table remains shared, including registries, transport,
# scoring, asset infrastructure, bindings and imported RoboCasa camera geometry.
_BACKEND_PACKAGES = ("robotwin", "robodojo", "behavior", "simple", "abc")
_ASSET_OWNERS = {
    "abc_assets.py": "abc",
    "abc_blender.py": "abc",
    "simple_assets.py": "simple",
    "robotwin_assets.py": "robotwin",
    "robodojo_assets.py": "robodojo",
    "behavior_assets.py": "behavior",
}


def python_files(root):
    """Every *.py below a package root as (relative POSIX path, bytes), for a package that is a
    directory or an importable archive (a wheel on sys.path: its Traversable has no rglob)."""

    def walk(node, prefix):
        for child in sorted(node.iterdir(), key=lambda entry: entry.name):
            relative = prefix + child.name
            if child.is_dir():
                yield from walk(child, relative + "/")
            elif child.name.endswith(".py"):
                yield relative, child.read_bytes()

    yield from walk(root, "")


def implementation_inventory(backend, *, roots=None):
    """Include shared Python and this backend's independently owned modules.

    The selection is static: imports exercised by one particular task cannot
    shrink its identity. New modules are shared unless explicitly owned here.
    Cross-backend reuse belongs in a shared module, outside the owned packages.
    """
    if backend not in _BACKEND_PACKAGES:
        raise ValueError("Unsupported native implementation backend")
    roots = roots if roots is not None else {
        package: resources.files(package) for package in ("roboenv", "roboenv_runtime")
    }
    content = {}
    for package, root in sorted(roots.items()):
        for relative, data in python_files(root):
            parts = relative.split("/")
            owner = None
            if package == "roboenv_runtime" and len(parts) >= 3 and parts[0] == "adapters":
                owner = parts[1] if parts[1] in _BACKEND_PACKAGES else None
            elif package == "roboenv" and len(parts) == 2 and parts[0] == "provisioning":
                owner = _ASSET_OWNERS.get(parts[1])
            if owner is not None and owner != backend:
                continue
            content[package + "/" + relative] = hashlib.sha256(data).hexdigest()
    return content


def implementation_identity(backend):
    """Hash selected installed Python, source pins, patches and dependencies."""
    content = implementation_inventory(backend)
    installed = sorted(
        (item.metadata["Name"].lower(), item.version)
        for item in metadata.distributions()
        if item.metadata["Name"]
    )
    from roboenv.provisioning.sources import load_lock, selected_sources, status

    project, lock = load_lock()
    if not status(project, [backend])["ready"]:
        raise ValueError("Native source tree differs from its verified lock and managed patches")
    sources = {name: lock["sources"][name] for name in selected_sources(lock, [backend])}
    return {
        "selection": "shared-and-selected-backend/1",
        "backend": backend,
        "implementation_sha256": digest(content),
        "dependencies_sha256": digest(installed),
        "sources_sha256": digest(sources),
    }


def initialize_record(world, config, backend):
    world.replay_config = copy.deepcopy(config)
    world.initial_state = copy.deepcopy(world.owner_state())
    world.implementation_identity = implementation_identity(backend)
    if hasattr(world, "asset_roots"):
        world.implementation_identity["assets"] = asset_identity(world.asset_roots)


def observe(adapter):
    world = adapter._env
    return {
        "schema": "roboenv/native-device-episode/1",
        "backend": adapter.name,
        "execution_mode": adapter.execution_mode,
        "embodiment": adapter.embodiment,
        "environment": copy.deepcopy(world.replay_config),
        "implementation": world.implementation_identity,
        "initialization": copy.deepcopy(world.initialization),
        "initial_state": copy.deepcopy(world.initial_state),
        "commands": copy.deepcopy(world.control.commands),
        "state": world.owner_state(),
        "control_state": {
            "targets": copy.deepcopy(world.control.targets),
            "gripper_holds": copy.deepcopy(getattr(world, "gripper_holds", {})),
        },
        "budget": adapter.budget().as_dict(),
        "native_success": world.task_completed(),
    }


def compare(expected, actual, *, atol, path="state"):
    """Compare the complete declared state; never silently drop unknown fields."""
    if isinstance(expected, dict):
        if set(expected) == {"state", "sha256"}:
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError(f"Replay state wrapper differs at {path}")
            if (
                digest(expected["state"]) != expected["sha256"]
                or digest(actual["state"]) != actual["sha256"]
            ):
                raise ValueError(f"Replay state checksum differs at {path}")
            return compare(expected["state"], actual["state"], atol=atol, path=path)
        keys = set(expected)
        if not isinstance(actual, dict) or set(actual) != keys:
            raise ValueError(f"Replay state keys differ at {path}")
        return max(
            (
                compare(expected[key], actual[key], atol=atol, path=f"{path}.{key}")
                for key in sorted(keys)
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Replay state dimensions differ at {path}")
        return max(
            (
                compare(left, right, atol=atol, path=f"{path}[{index}]")
                for index, (left, right) in enumerate(zip(expected, actual, strict=True))
            ),
            default=0.0,
        )
    if type(expected) in (int, float):
        if (
            type(actual) not in (int, float)
            or not math.isfinite(expected)
            or not math.isfinite(actual)
        ):
            raise ValueError(f"Replay state is not finite at {path}")
        difference = abs(expected - actual)
        if difference > atol:
            raise ValueError(f"Replay state differs at {path}: {difference} > {atol}")
        return difference
    if type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Replay state differs at {path}")
    return 0.0
