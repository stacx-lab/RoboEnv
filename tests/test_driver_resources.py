"""Portable bundles include local geometry and reject private or escaping paths."""

import json

import pytest
from roboenv.devices.export import driver_resources


def manifest(root, names):
    (root / "driver-resources.json").write_text(json.dumps({
        "schema": "roboenv/driver-resources/1", "files": names,
    }))


def test_nested_geometry_is_explicit_and_episode_files_are_excluded(tmp_path):
    (tmp_path / "driver.py").write_text("pass\n")
    (tmp_path / "geometry").mkdir()
    (tmp_path / "geometry/arm.urdf").write_text('<robot name="arm"/>')
    (tmp_path / "connection.json").write_text('{"secret": "owner"}')
    manifest(tmp_path, ["driver.py", "geometry/arm.urdf"])
    assert [str(path) for path in driver_resources(tmp_path)] == [
        "driver.py", "geometry/arm.urdf", "driver-resources.json",
    ]


@pytest.mark.parametrize("name", [
    "../private.json", "/tmp/private.json", "connection.json", "owner-record.json",
    "verdict.json", ".private/data.json", "geometry/../arm.urdf", "profile.yaml",
])
def test_private_or_escaping_resource_is_rejected(tmp_path, name):
    (tmp_path / "driver.py").write_text("pass\n")
    manifest(tmp_path, ["driver.py", name])
    with pytest.raises(ValueError, match="Invalid portable"):
        driver_resources(tmp_path)


def test_symlinked_resource_directory_is_rejected(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "driver.py").write_text("pass\n")
    (tmp_path / "arm.urdf").write_text('<robot name="arm"/>')
    (root / "geometry").symlink_to(tmp_path, target_is_directory=True)
    manifest(root, ["driver.py", "geometry/arm.urdf"])
    with pytest.raises(ValueError, match="symlinks"):
        driver_resources(root)


def test_legacy_python_bundle_and_missing_manifest_resource(tmp_path):
    (tmp_path / "driver.py").write_text("pass\n")
    (tmp_path / "_roboenv.py").write_text("pass\n")
    assert {str(path) for path in driver_resources(tmp_path)} == {"driver.py", "_roboenv.py"}
    manifest(tmp_path, ["driver.py", "missing.urdf"])
    with pytest.raises(ValueError, match="Missing portable"):
        driver_resources(tmp_path)
