"""Source asset integrity and preservation failures without network or GPU imports."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from roboenv.provisioning import robodojo_assets, robotwin_assets
from roboenv.provisioning.yam_usd import convert
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


@pytest.mark.parametrize("linked", [False, True])
def test_robodojo_default_preparation_reuses_existing_assets(tmp_path, monkeypatch, linked):
    from roboenv.provisioning import yam_assets, yam_usd

    monkeypatch.delenv("ROBOENV_ASSET_CACHE", raising=False)
    link = tmp_path / "third_party/RoboDojo/Assets"
    link.parent.mkdir(parents=True)
    assets = tmp_path / "existing-user-depot" if linked else link
    (assets / "Robots/yam").mkdir(parents=True)
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
    monkeypatch.setattr(yam_assets, "verify_source", lambda *a, **k: True)
    monkeypatch.setattr(yam_usd, "verify", lambda *a, **k: False)
    report = robodojo_assets.prepare_assets(tmp_path)
    assert Path(report["asset_depot"]) == assets.resolve()
    assert report["public_assets_verified"] and report["yam_native_conversion_required"]
    assert not report["environment_assets_verified"]
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


def test_yam_conversion_requires_explicit_license_before_engine_import(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    with pytest.raises(ValueError, match="explicitly accepted EULA"):
        convert(tmp_path)
    assert not list(Path(tmp_path).iterdir())


@pytest.mark.parametrize("receipt", [None, {"passed": False, "error": "conversion failed"}])
def test_yam_cli_rejects_zero_exit_without_success_receipt(tmp_path, monkeypatch, receipt):
    from types import SimpleNamespace

    from roboenv.provisioning import yam_usd

    def exited_worker(command, **kwargs):
        if receipt is not None:
            Path(command[-1]).write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(yam_usd.subprocess, "run", exited_worker)
    monkeypatch.setattr(yam_usd.sys, "argv", ["yam_usd", str(tmp_path / "yam")])
    with pytest.raises(RuntimeError, match="no completion receipt|conversion failed"):
        yam_usd.main()


@pytest.mark.parametrize("modified", [False, True])
def test_yam_entrypoint_upgrade_preserves_geometry_and_rejects_drift(
    tmp_path, monkeypatch, modified
):
    from roboenv.provisioning import yam_usd

    output = tmp_path / "native-usd"
    output.mkdir()
    (tmp_path / "YAM.urdf").write_text("source fixture")
    geometry = output / "YAM.usd"
    geometry.write_bytes(b"native geometry fixture")
    link = tmp_path / "YAM.usd"
    link.symlink_to("native-usd/YAM.usd")
    previous = {
        "schema": "roboenv/yam-native-conversion/1",
        "source_files_sha256": yam_usd.checksum(yam_usd.source_inventory(tmp_path)),
        "output_files_sha256": yam_usd.checksum(yam_usd.inventory(output)),
    }
    (output / "conversion.json").write_text(json.dumps(previous))
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    if modified:
        geometry.write_bytes(b"user modification")
        with pytest.raises(ValueError, match="geometry or source inputs changed"):
            yam_usd.upgrade_entrypoint(tmp_path)
        assert link.is_symlink() and geometry.read_bytes() == b"user modification"
    else:
        assert yam_usd.verify(tmp_path) is False
        receipt = yam_usd.upgrade_entrypoint(tmp_path)
        assert receipt["previous_conversion"] == previous
        assert geometry.read_bytes() == b"native geometry fixture"
        assert not link.is_symlink() and link.read_bytes() == yam_usd.ENTRYPOINT
        assert yam_usd.verify(tmp_path, require_current=False) is True
        assert yam_usd.verify(tmp_path) is False


@pytest.mark.parametrize("drift", [None, "current", "staged"])
def test_yam_collision_upgrade_archives_old_bytes_and_rejects_drift(tmp_path, drift):
    from roboenv.provisioning import yam_usd

    directory = tmp_path / "yam"
    directory.mkdir()
    (directory / "YAM.urdf").write_text("source fixture")
    (directory / "YAM.usd").write_bytes(yam_usd.ENTRYPOINT)
    output = directory / "native-usd"
    output.mkdir()
    (output / "YAM.usd").write_bytes(b"old geometry")
    previous = {
        "schema": "roboenv/yam-native-conversion/2",
        "entrypoint_sha256": yam_usd.hashlib.sha256(yam_usd.ENTRYPOINT).hexdigest(),
        "source_files_sha256": yam_usd.checksum(yam_usd.source_inventory(directory)),
        "output_files_sha256": yam_usd.checksum(yam_usd.output_inventory(output)),
    }
    old_receipt = json.dumps(previous)
    (output / "conversion.json").write_text(old_receipt)
    assert yam_usd.verify(directory) is False
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "YAM.usd").write_bytes(b"new geometry")
    receipt = {
        **previous, "schema": yam_usd.SCHEMA, "previous_conversion": previous,
        "output_files_sha256": yam_usd.checksum(yam_usd.output_inventory(staged)),
        "finger_colliders": [
            {"finger": name, "approximation": "convexDecomposition"}
            for name in ("tip_left", "tip_right")
        ],
    }
    (staged / "conversion.json").write_text(json.dumps(receipt))
    if drift:
        changed = output if drift == "current" else staged
        (changed / "YAM.usd").write_bytes(b"concurrent modification")
        with pytest.raises(ValueError, match="changed during upgrade|did not verify"):
            yam_usd.publish_upgrade(directory, staged, previous)
        assert (output / "conversion.json").read_text() == old_receipt
        assert not (directory / yam_usd.HISTORY).exists()
        return
    yam_usd.publish_upgrade(directory, staged, previous)
    archive = directory / yam_usd.HISTORY / previous["output_files_sha256"]
    assert (archive / "YAM.usd").read_bytes() == b"old geometry"
    assert (archive / "conversion.json").read_text() == old_receipt
    assert (output / "YAM.usd").read_bytes() == b"new geometry"
    assert yam_usd.verify(directory) is True
    (archive / "YAM.usd").write_bytes(b"modified old evidence")
    with pytest.raises(ValueError, match="Archived YAM conversion changed"):
        yam_usd.verify(directory)


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
