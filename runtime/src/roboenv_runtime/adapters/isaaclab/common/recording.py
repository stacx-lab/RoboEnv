"""Source identities for the selected Isaac task and its shared implementation."""

import hashlib
import inspect
import json
from pathlib import Path


def source_identity(task_name, robot_directory, *, extra_directories=(), backends=("isaaclab",)):
    from roboenv.provisioning.sources import selected_sources

    from roboenv_runtime.paths import repo_root

    from ..registry import get_task

    backend = Path(__file__).resolve().parents[1]
    task_directory = Path(inspect.getfile(get_task(task_name))).resolve().parent
    paths = set(backend.glob("*.py"))
    for directory in (backend / "common", Path(robot_directory).resolve(), task_directory,
                      *extra_directories):
        paths.update(directory.rglob("*.py"))
        # Package initializers may define behavior shared by sibling implementations.
        for parent in directory.parents:
            if parent == backend:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                paths.add(initializer)
    sources = {
        path.relative_to(backend).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    root = repo_root()
    lock = json.loads((root / "upstreams.lock.json").read_text())
    if lock.get("schema") != "roboenv/upstreams/1":
        raise ValueError("Unsupported upstream lock")
    selected = {
        "schema": lock["schema"],
        "sources": {
            name: lock["sources"][name] for name in selected_sources(lock, list(backends))
        },
    }
    sources["upstreams.lock.json#" + "+".join(backends)] = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    name = "runtime/locks/isaaclab-resolved.txt"
    sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return sources
