"""Source asset integrity and preservation failures without network or GPU imports."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from roboenv.provisioning import robodojo_assets, robotwin_assets
from roboenv_runtime.adapters.native_record import asset_identity, compare, digest


def test_robotwin_archive_verifies_existing_user_assets_without_overwriting(tmp_path, monkeypatch):
    assets = tmp_path / "third_party/RoboTwin/assets"
    assets.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "objects.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("objects/item/model.urdf", "<robot/>")
    monkeypatch.setattr(
        robotwin_assets,
        "ARCHIVES",
        {
            "objects": (archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest()),
        },
    )
    report = robotwin_assets.prepare_assets(tmp_path, cache=cache)
    assert report["environment_assets_verified"] is True
    model = assets / "objects/item/model.urdf"
    assert model.read_text() == "<robot/>"
    model.write_text("user changes")
    with pytest.raises(ValueError, match="differ from the pin"):
        robotwin_assets.prepare_assets(tmp_path, cache=cache)
    assert model.read_text() == "user changes"


def test_robodojo_download_verifies_git_and_lfs_blobs_and_preserves_existing(tmp_path, monkeypatch):
    data = b"native asset bytes"
    entry = {
        "rfilename": "Assets/Robots/x5/robot_config.yml",
        "size": len(data),
        "blobId": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    monkeypatch.setattr(robodojo_assets, "urlopen", lambda *_a, **_kw: io.BytesIO(data))
    robodojo_assets.download_file(tmp_path, entry)
    path = tmp_path / Path(entry["rfilename"]).relative_to("Assets")
    assert robodojo_assets.verified(path, entry)
    assert robodojo_assets.verified(
        path, {**entry, "lfs": {"sha256": hashlib.sha256(data).hexdigest()}}
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing asset differs"):
        robodojo_assets.download_file(tmp_path, entry)
    assert path.read_bytes() == b"changed"


def test_robodojo_external_materials_verify_original_bytes_and_preserve_changes(
    tmp_path, monkeypatch,
):
    data = {"Metals/native.mdl": b"native shader", "Metals/native/texture.png": b"native texture"}
    pin = {"base_url": "https://materials.example/", "files": {
        name: {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in data.items()}}
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(data[url.removeprefix(pin["base_url"])])

    monkeypatch.setattr(robodojo_assets, "material_manifest", lambda: pin)
    monkeypatch.setattr(robodojo_assets, "urlopen", download)
    report = robodojo_assets.prepare_materials(tmp_path)
    assert report == {"files": 2, "bytes": sum(len(value) for value in data.values())}
    assert robodojo_assets.verify_materials(tmp_path) == pin
    robodojo_assets.prepare_materials(tmp_path)
    assert requests == [pin["base_url"] + name for name in data]
    texture = tmp_path / ".roboenv-materials/Metals/native/texture.png"
    texture.write_bytes(b"preserve user edits")
    with pytest.raises(ValueError, match="missing or changed"):
        robodojo_assets.verify_materials(tmp_path)
    with pytest.raises(ValueError, match="Existing asset differs"):
        robodojo_assets.prepare_materials(tmp_path)
    assert texture.read_bytes() == b"preserve user edits"


@pytest.mark.parametrize("linked", [False, True])
def test_robodojo_default_preparation_reuses_existing_assets(tmp_path, monkeypatch, linked):
    monkeypatch.delenv("ROBOENV_ASSET_CACHE", raising=False)
    link = tmp_path / "third_party/RoboDojo/Assets"
    link.parent.mkdir(parents=True)
    assets = tmp_path / "existing-user-depot" if linked else link
    (assets / "Robots").mkdir(parents=True)
    if linked:
        link.symlink_to(assets, target_is_directory=True)
    data = b"pinned robot geometry"
    asset = assets / "Robots/geometry.bin"
    asset.write_bytes(data)
    entry = {
        "rfilename": "Assets/Robots/geometry.bin",
        "size": len(data),
        "blobId": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    cache = tmp_path / ".cache/roboenv/assets/robodojo"
    cache.mkdir(parents=True)
    (cache / "source.json").write_text(json.dumps({
        "sha": robodojo_assets.REVISION, "siblings": [entry], "gated": False, "private": False,
    }))
    monkeypatch.setattr(robodojo_assets, "urlopen", lambda *a, **k: pytest.fail("no download"))
    monkeypatch.setattr(robodojo_assets, "material_manifest", lambda: {"files": {}})
    report = robodojo_assets.prepare_assets(tmp_path)
    assert Path(report["asset_depot"]) == assets.resolve()
    assert report["public_assets_verified"] and report["environment_assets_verified"]
    assert "yam_native_conversion_required" not in report
    assert link.resolve() == assets.resolve()
    assert not (cache / robodojo_assets.REVISION).exists()
    asset.write_bytes(b"user modifications")
    with pytest.raises(ValueError, match="Existing asset differs"):
        robodojo_assets.prepare_assets(tmp_path)
    assert asset.read_bytes() == b"user modifications"


@pytest.mark.parametrize("name", ["", "/Assets/a", "Assets/../outside", "Assets/a\\b", "another/a"])
def test_asset_paths_cannot_escape_depot(name):
    with pytest.raises(ValueError, match="Unsafe"):
        robodojo_assets.safe_relative(name)


def test_replay_asset_identity_detects_content_changes_and_ignores_root_location(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "asset").write_bytes(b"same")
    baseline = asset_identity({"objects": first})
    assert baseline == asset_identity({"objects": second})
    (second / "asset").write_bytes(b"changed")
    assert baseline != asset_identity({"objects": second})
    expected = {"state": {"objects": [1.0, 2.0]}, "sha256": digest({"objects": [1.0, 2.0]})}
    assert compare(expected, expected, atol=1e-5) == 0
    with pytest.raises(ValueError, match="checksum"):
        compare(expected, {**expected, "sha256": "invalid"}, atol=1e-5)


def test_behavior_acquisition_requires_both_licenses_before_network_or_engine(
    tmp_path, monkeypatch
):
    from roboenv.provisioning.behavior_assets import acquire, prepare_assets

    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.delenv("ROBOENV_BEHAVIOR_ACCEPT_LICENSE", raising=False)
    assert len(prepare_assets(tmp_path)["license_acceptance_required"]) == 2
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    assert prepare_assets(tmp_path)["license_acceptance_required"] == [
        "BEHAVIOR Data Bundle agreement"
    ]
    with pytest.raises(ValueError, match="Explicit acceptance"):
        acquire(tmp_path, tmp_path / "not-created")
    assert not list(tmp_path.iterdir())


def test_behavior_bundle_preserves_existing_modified_data(tmp_path):
    from roboenv.provisioning.behavior_assets import install_archive

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("VERSION", "test")
        bundle.writestr("models/robot.urdf", "fixture geometry")
    destination = tmp_path / "data"
    expected = install_archive(archive, destination)
    assert install_archive(archive, destination) == expected
    target = destination / "models/robot.urdf"
    target.write_text("preserve changed geometry")
    with pytest.raises(ValueError, match="Existing BEHAVIOR data differs"):
        install_archive(archive, destination)
    assert target.read_text() == "preserve changed geometry"


@pytest.mark.parametrize("name", ["../outside", "/outside", "models\\outside"])
def test_behavior_archive_cannot_escape_depot(tmp_path, name):
    from roboenv.provisioning.behavior_assets import install_archive

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        install_archive(archive, tmp_path / "data")
    assert not (tmp_path / "data").exists()
