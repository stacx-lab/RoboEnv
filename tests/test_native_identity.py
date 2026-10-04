"""Identity changes follow shared and simulator ownership boundaries."""

import ast
from pathlib import Path

import pytest
from roboenv_runtime.adapters.native_record import (
    _BACKEND_PACKAGES,
    implementation_inventory,
)


def test_backend_changes_do_not_invalidate_an_independent_native_world(tmp_path):
    roots = {package: tmp_path / package for package in ("roboenv", "roboenv_runtime")}
    files = {
        "roboenv_runtime/adapters/device_control.py": "shared_control = 1\n",
        "roboenv_runtime/adapters/robotwin/devices.py": "robotwin_control = 1\n",
        "roboenv_runtime/adapters/robodojo/devices.py": "robodojo_control = 1\n",
        "roboenv/provisioning/behavior_assets.py": "behavior_assets = 1\n",
        "roboenv/assets/joint-client.py": "shared_driver = 1\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    before = {name: implementation_inventory(name, roots=roots) for name in _BACKEND_PACKAGES}
    (tmp_path / "roboenv_runtime/adapters/robodojo/devices.py").write_text("robodojo_control = 2\n")
    assert implementation_inventory("robotwin", roots=roots) == before["robotwin"]
    assert implementation_inventory("behavior", roots=roots) == before["behavior"]
    assert implementation_inventory("robodojo", roots=roots) != before["robodojo"]
    (tmp_path / "roboenv/provisioning/behavior_assets.py").write_text("behavior_assets = 2\n")
    assert implementation_inventory("robotwin", roots=roots) == before["robotwin"]
    assert implementation_inventory("behavior", roots=roots) != before["behavior"]
    before = {name: implementation_inventory(name, roots=roots) for name in _BACKEND_PACKAGES}
    (tmp_path / "roboenv/assets/joint-client.py").write_text("shared_driver = 2\n")
    for name in _BACKEND_PACKAGES:
        assert implementation_inventory(name, roots=roots) != before[name]
    before = {name: implementation_inventory(name, roots=roots) for name in _BACKEND_PACKAGES}
    (tmp_path / "roboenv_runtime/adapters/device_control.py").write_text("shared_control = 2\n")
    for name in _BACKEND_PACKAGES:
        assert implementation_inventory(name, roots=roots) != before[name]
    with pytest.raises(ValueError, match="Unsupported"):
        implementation_inventory("unknown", roots=roots)


def test_independent_backend_packages_do_not_import_each_other():
    # Shared geometry lives outside these packages. Keep future direct imports
    # from silently crossing the independently hashed implementation boundary.
    import roboenv_runtime.adapters

    root = Path(roboenv_runtime.adapters.__file__).parent
    for backend in _BACKEND_PACKAGES:
        other = [
            f"roboenv_runtime.adapters.{name}" for name in _BACKEND_PACKAGES if name != backend
        ]
        for path in (root / backend).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parent = ["roboenv_runtime", "adapters", *path.relative_to(root).parts[:-1]]
                        parent = parent[: len(parent) - node.level + 1]
                        base = ".".join([*parent, *([node.module] if node.module else [])])
                    else:
                        base = node.module or ""
                    modules = [base, *(base + "." + alias.name for alias in node.names)]
                else:
                    continue
                for module in modules:
                    assert not any(
                        module == prefix or module.startswith(prefix + ".") for prefix in other
                    ), f"{path}: move cross-backend dependencies into a shared module"


def test_inventory_reads_a_package_imported_from_an_archive(tmp_path, monkeypatch):
    """The SDK may be on sys.path as a wheel (the standalone image's launch): the inventory
    walks its Traversable, which has no rglob, and matches the same tree on disk."""
    import sys
    import zipfile
    from importlib import resources

    files = {
        "zipsdk/__init__.py": "",
        "zipsdk/provisioning/__init__.py": "",
        "zipsdk/provisioning/robotwin_assets.py": "robotwin = 1\n",
        "ziprt/__init__.py": "",
        "ziprt/adapters/__init__.py": "",
        "ziprt/adapters/device_control.py": "shared = 1\n",
        "ziprt/adapters/robotwin/__init__.py": "",
        "ziprt/adapters/robotwin/devices.py": "control = 1\n",
    }
    archive = tmp_path / "sdk-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(name, content)
    for name, content in files.items():
        (tmp_path / "tree" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "tree" / name).write_text(content)
    monkeypatch.syspath_prepend(str(archive))
    for module in [name for name in sys.modules if name.startswith(("zipsdk", "ziprt"))]:
        del sys.modules[module]
    archived = {"roboenv": resources.files("zipsdk"), "roboenv_runtime": resources.files("ziprt")}
    on_disk = {"roboenv": tmp_path / "tree/zipsdk", "roboenv_runtime": tmp_path / "tree/ziprt"}
    assert implementation_inventory("robotwin", roots=archived) == implementation_inventory(
        "robotwin", roots=on_disk
    )
    assert "roboenv_runtime/adapters/robotwin/devices.py" in implementation_inventory(
        "robotwin", roots=archived
    )
