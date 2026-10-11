"""Install a static robot catalog in the current Kinex workspace, without a simulator."""

import re
import shutil
import tempfile
from pathlib import Path

from roboenv.devices.export import driver_resources, shared_driver_resources
from roboenv.project import project_root
from roboenv.validation import yaml_load


def _no_symlinks(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Workspace resources must not contain symlinks: {path}")


def _catalog_root(project, catalog):
    root = Path(catalog).expanduser().absolute() if catalog else project_root(project) / "catalog"
    _no_symlinks(root / "robots")
    if not (root / "robots").is_dir():
        raise ValueError("Selected catalog must contain a robots directory")
    return root.resolve()


def list_robots(*, project=None, catalog=None):
    """List canonical simulator/robot IDs without loading tasks or simulator bindings."""
    root = _catalog_root(project, catalog) / "robots"
    result = []
    for manifest in sorted(root.glob("*/*/workspace.yaml")):
        _no_symlinks(manifest)
        if manifest.is_file():
            result.append(manifest.parent.relative_to(root).as_posix())
    return result


def _mapping(path):
    _no_symlinks(path)
    if not path.is_file():
        raise ValueError(f"Missing robot resource: {path}")
    value = yaml_load(path.read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Robot resource must be a nonempty YAML mapping: {path}")
    return value


def _resources(robot):
    """Validate and enumerate only public profiles and portable support files."""
    _mapping(robot / "workspace.yaml")
    collection = robot / "devices"
    _no_symlinks(collection)
    if not collection.is_dir():
        raise ValueError("Robot catalog is missing its devices directory")
    resources = [Path("workspace.yaml")]
    resources.extend(Path("devices") / name for name in shared_driver_resources(collection))
    devices = []
    for device in sorted(collection.iterdir()):
        _no_symlinks(device)
        if not device.is_dir() or device.name == "__pycache__":
            continue
        profile = _mapping(device / "profile.yaml")
        if profile.get("name") != device.name or not isinstance(profile.get("type"), str):
            raise ValueError(f"Invalid device name or type: {device.name}")
        relative = Path("devices") / device.name
        resources.append(relative / "profile.yaml")
        support = driver_resources(device)
        if profile.get("access"):
            access = profile["access"]
            driver = access.get("driver") if isinstance(access, dict) else None
            entry = driver.get("entry") if isinstance(driver, dict) else None
            if not isinstance(entry, str) or Path(entry) not in support:
                raise ValueError(f"Missing portable driver for device: {device.name}")
            if Path("devices/_roboenv.py") not in resources:
                raise ValueError("Robot catalog is missing shared transport devices/_roboenv.py")
        resources.extend(relative / name for name in support)
        devices.append(device.name)
    if not devices:
        raise ValueError("Robot catalog contains no device profiles")
    return resources, devices


def _destination(config, *, force):
    _no_symlinks(config)
    if config.exists() and not config.is_dir():
        raise ValueError(".kinex must be a directory")
    for name in ("workspace.yaml", "devices"):
        target = config / name
        _no_symlinks(target)
        if target.exists():
            expected = target.is_dir() if name == "devices" else target.is_file()
            if not expected:
                raise ValueError(f"Unexpected file type for .kinex/{name}")
            if not force:
                raise ValueError(
                    ".kinex already contains robot configuration; "
                    "use --force to replace workspace.yaml and the entire devices directory"
                )


def _publish(staging, config, *, force):
    # Recheck after copying, before touching an existing workspace.
    _destination(config, force=force)
    prepared = staging / "prepared"
    if not config.exists():
        prepared.rename(config)
        return
    backup = staging / "backup"
    backup.mkdir()
    saved, installed = [], []
    try:
        for name in ("workspace.yaml", "devices"):
            target = config / name
            if target.exists():
                target.rename(backup / name)
                saved.append(name)
            (prepared / name).rename(target)
            installed.append(name)
    except BaseException:
        try:
            for name in reversed(installed):
                (config / name).rename(prepared / name)
            for name in reversed(saved):
                (backup / name).rename(config / name)
        except OSError as exc:
            raise OSError(
                f"Could not restore robot configuration; recover files from {backup}"
            ) from exc
        raise


def setup_robot(reference, *, project=None, catalog=None, force=False):
    """Export a robot into cwd/.kinex, preserving unrelated workspace content."""
    if not isinstance(reference, str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*", reference
    ):
        raise ValueError("Use a simulator/robot ID from roboenv setup --list")
    root = _catalog_root(project, catalog)
    robot = root / "robots" / reference
    _no_symlinks(robot / "workspace.yaml")
    if not (robot / "workspace.yaml").is_file():
        raise ValueError(f"Unknown robot {reference!r}; use roboenv setup --list")
    config = Path.cwd() / ".kinex"
    if config.is_relative_to(root) or root.is_relative_to(config):
        raise ValueError("Workspace destination must not overlap the source catalog")
    _destination(config, force=force)
    resources, devices = _resources(robot)
    staging = Path(tempfile.mkdtemp(dir=Path.cwd(), prefix=".roboenv-setup-"))
    completed = False
    try:
        for relative in resources:
            target = staging / "prepared" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(robot / relative, target)
        _publish(staging, config, force=force)
        completed = True
    finally:
        # If restoration itself failed, retain the backup for manual recovery.
        backup = staging / "backup"
        if completed or not backup.exists() or not any(backup.iterdir()):
            shutil.rmtree(staging)
    return {"robot": reference, "destination": str(config), "devices": devices}
