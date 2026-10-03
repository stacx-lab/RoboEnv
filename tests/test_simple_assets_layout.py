"""SIMPLE depot exposure for the Isaac engine's package-relative data lookup."""

import os
from pathlib import Path

from roboenv.provisioning.simple_assets import expose_native_layout


def test_expose_native_layout_links_depot_into_package_data(tmp_path):
    root = tmp_path / "project"
    data = root / ".cache/simple/data"
    for name in ("robots", "scenes", "vMaterials_2"):
        (data / name).mkdir(parents=True)
    (data / "robots/g1.usd").write_text("usd")
    (data / "receipt.json").write_text("{}")  # files are never linked
    package_data = root / "third_party/SIMPLE/data"
    package_data.mkdir(parents=True)
    (package_data / "README.md").write_text("upstream placeholder")

    expose_native_layout(root, data)

    for name in ("robots", "scenes", "vMaterials_2"):
        link = package_data / name
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
        assert link.resolve() == (data / name).resolve()
    assert (package_data / "robots/g1.usd").read_text() == "usd"
    assert not (package_data / "receipt.json").exists()
    assert (package_data / "README.md").read_text() == "upstream placeholder"


def test_expose_native_layout_keeps_existing_entries_and_tolerates_missing_checkout(tmp_path):
    root = tmp_path / "project"
    data = root / ".cache/simple/data"
    (data / "robots").mkdir(parents=True)
    expose_native_layout(root, data)  # no third_party/SIMPLE: nothing to do
    assert not (root / "third_party").exists()

    package_data = root / "third_party/SIMPLE/data"
    (package_data / "robots").mkdir(parents=True)
    (package_data / "robots/local.txt").write_text("keep")
    expose_native_layout(root, data)
    assert not (package_data / "robots").is_symlink()
    assert (package_data / "robots/local.txt").read_text() == "keep"
    assert Path(package_data / "robots").is_dir()
