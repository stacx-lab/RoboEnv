"""Release versions have one editable source and work outside the source tree."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from helpers import load_tool

ROOT = Path(__file__).resolve().parents[1]
VERSION = load_tool("version")
FILES = ["VERSION", "pyproject.toml", "runtime/pyproject.toml",
         "src/roboenv/__init__.py", "runtime/src/roboenv_runtime/__init__.py"]


@pytest.fixture
def project(tmp_path):
    for name in [*FILES, "tools/version.py"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return tmp_path


def snapshot(root):
    return {name: (root / name).read_bytes() for name in FILES}


def test_sync_updates_metadata_preserves_other_fields_and_is_idempotent(project):
    originals = {name: tomllib.loads((project / name).read_text())
                 for name in ("pyproject.toml", "runtime/pyproject.toml")}
    (project / "VERSION").write_text("0.11.0\n")
    before = snapshot(project)
    with pytest.raises(ValueError, match="project.version must be 0.11.0"):
        VERSION.check_version(project)
    assert snapshot(project) == before
    assert len(VERSION.sync_version(project)) == 4
    VERSION.check_version(project)
    for name, original in originals.items():
        expected = original
        expected["project"]["version"] = "0.11.0"
        if name.startswith("runtime/"):
            expected["project"]["dependencies"] = [
                "roboenv[server]==0.11.0" if item.startswith("roboenv[server]==") else item
                for item in expected["project"]["dependencies"]
            ]
        assert tomllib.loads((project / name).read_text()) == expected
    for name in ("src/roboenv/__init__.py", "runtime/src/roboenv_runtime/__init__.py"):
        tree = ast.parse((project / name).read_text())
        assert any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                   and node.value.value == "0.11.0" for node in tree.body)
    synced = snapshot(project)
    assert VERSION.sync_version(project) == []
    assert snapshot(project) == synced


@pytest.mark.parametrize("name", FILES[1:])
def test_check_identifies_each_stale_copy_without_writing(project, name):
    path = project / name
    path.write_text(path.read_text().replace(VERSION.read_version(project), "99.0.0"))
    before = snapshot(project)
    with pytest.raises(ValueError, match=name):
        VERSION.check_version(project)
    assert snapshot(project) == before
    VERSION.sync_version(project)
    VERSION.check_version(project)


@pytest.mark.parametrize("value", ["", "v0.10.0", "0.10", "01.10.0", "0.10.0-rc.1"])
def test_invalid_version_never_partially_updates_files(project, value):
    (project / "VERSION").write_text(value)
    before = snapshot(project)
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        VERSION.sync_version(project)
    assert snapshot(project) == before


def test_missing_version_or_malformed_metadata_fails_before_writes(project):
    (project / "VERSION").unlink()
    with pytest.raises(FileNotFoundError):
        VERSION.check_version(project)
    (project / "VERSION").write_text("0.11.0\n")
    (project / "runtime/pyproject.toml").write_text("invalid toml")
    before = snapshot(project)
    with pytest.raises(tomllib.TOMLDecodeError):
        VERSION.sync_version(project)
    assert snapshot(project) == before


def test_command_resolves_repository_from_script_not_working_directory(project, tmp_path):
    before = snapshot(project)
    subprocess.run([sys.executable, str(project / "tools/version.py"), "--check"],
                   cwd=tmp_path.parent, check=True, capture_output=True)
    assert snapshot(project) == before
