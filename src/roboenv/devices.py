"""Access the environment's device files without a particular agent framework."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


def robot_root():
    return Path(os.environ.get("ROBOENV_ROBOT", "/env/robot"))


def profile(name, *, root=None):
    base = Path(root or robot_root()).resolve()
    device = (base / "devices" / name).resolve()
    if device.parent != base / "devices":
        raise ValueError("device must name one directory in the environment")
    with (device / "profile.yaml").open() as stream:
        return yaml.safe_load(stream)


def list_devices(*, root=None):
    base = Path(root or robot_root())
    return {
        path.parent.name: profile(path.parent.name, root=base)
        for path in sorted((base / "devices").glob("*/profile.yaml"))
    }


def invoke(name, action="read", args=None, *, root=None):
    base = Path(root or robot_root()).resolve()
    spec = profile(name, root=base)
    driver = spec.get("access", {}).get("driver")
    if not driver or action not in driver.get("actions", []):
        raise ValueError(f"device {name} does not declare action {action}")
    directory = base / "devices" / name
    entry = (directory / driver.get("entry", "driver.py")).resolve()
    if directory.resolve() not in entry.parents:
        raise ValueError("driver entry escapes its device directory")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(base / "lib") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [driver.get("interpreter", "python3"), str(entry), action, json.dumps(args or {})],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=driver.get("timeout_s", 600),
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not lines:
        raise RuntimeError(f"driver exited {result.returncode}: {result.stderr[-1000:]}")
    value = json.loads(lines[-1])
    if value.get("ok") is not True:
        raise RuntimeError(f"device {name}: {value.get('error', 'missing ok verdict')}")
    return value
