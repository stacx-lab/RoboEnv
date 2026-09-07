"""Asset preparation uses small local archives, without network or simulator imports."""

import io
import json
from zipfile import ZipFile

import pytest
from roboenv.provisioning import robocasa_assets as assets
from roboenv.provisioning import setup, sources


def bundle(name, content=b"asset"):
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(name, content)
    data.seek(0)
    return data


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "third_party/robocasa/robocasa/models/assets"
    (root / "box_links").mkdir(parents=True)
    (root / "box_links/box_links_assets.json").write_text(
        json.dumps(
            {
                "fixtures_lightwheel": "https://utexas.box.com/s/example",
            }
        )
    )
    monkeypatch.setattr(assets, "ARCHIVES", {"fixtures_lightwheel": "fixtures"})
    return tmp_path


def test_download_reuse_and_repair(project, monkeypatch):
    calls = []

    def download(*args, **kwargs):
        calls.append(args)
        return bundle("fixtures/cabinet/model.xml")

    monkeypatch.setattr(assets, "urlopen", download)
    assert assets.prepare_assets(project)["downloaded_assets_verified"]
    assets.prepare_assets(project)
    assert len(calls) == 1
    model = project / "third_party/robocasa/robocasa/models/assets/fixtures/cabinet/model.xml"
    model.write_bytes(b"wrong")
    assets.prepare_assets(project)
    assert len(calls) == 2
    assert model.read_bytes() == b"asset"
    model.unlink()
    assets.prepare_assets(project)
    assert len(calls) == 3


@pytest.mark.parametrize("name", ["../escape", "/absolute", "fixtures/../../escape", "other/model"])
def test_rejects_unsafe_archive_before_extraction(project, monkeypatch, name):
    monkeypatch.setattr(assets, "urlopen", lambda *a, **kw: bundle(name))
    with pytest.raises(RuntimeError, match="rerun"):
        assets.prepare_assets(project)
    assert not list((project / ".cache/roboenv/assets/robocasa").glob("*.json"))
    assert not (project / "third_party/robocasa/robocasa/models/assets/fixtures").exists()


def test_network_failure_can_be_retried(project, monkeypatch):
    def fail(*a, **kw):
        raise TimeoutError("private network detail")

    monkeypatch.setattr(assets, "urlopen", fail)
    with pytest.raises(RuntimeError, match="TimeoutError") as error:
        assets.prepare_assets(project)
    assert "private network detail" not in str(error.value)
    monkeypatch.setattr(assets, "urlopen", lambda *a, **kw: bundle("fixtures/model.xml"))
    assert assets.prepare_assets(project)["downloaded_assets_verified"]


@pytest.mark.parametrize("reuse", [False, True])
def test_prepare_acquires_assets_for_new_and_registered_interpreters(project, monkeypatch, reuse):
    monkeypatch.setattr(setup, "project_root", lambda _: project)
    monkeypatch.setattr(setup, "audit", lambda *a: ({}, {}))
    calls = []
    monkeypatch.setattr(sources, "sync", lambda *a: calls.append("sync"))
    monkeypatch.setattr(setup, "install", lambda *a, **kw: calls.append("install") or {})
    monkeypatch.setattr(assets, "urlopen", lambda *a, **kw: bundle("fixtures/model.xml"))
    if reuse:
        interpreter = project / "python"
        interpreter.touch()
        registration = project / ".cache/roboenv/installations/robocasa.json"
        registration.parent.mkdir(parents=True)
        registration.write_text(json.dumps({"python": str(interpreter), "report": {}}))
    result = setup.prepare("robocasa")
    assert result["downloaded_assets_verified"]
    assert not result["gpu_replay_verified"]
    assert calls == ([] if reuse else ["sync", "install"])


def test_extraction_rejects_existing_symlink_escape(project, tmp_path, monkeypatch):
    destination = project / "third_party/robocasa/robocasa/models/assets/fixtures"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "cabinet").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(assets, "urlopen", lambda *a, **kw: bundle("fixtures/cabinet/model.xml"))
    with pytest.raises(RuntimeError, match="ValueError"):
        assets.prepare_assets(project)
    assert list(outside.iterdir()) == []


def test_prepare_propagates_asset_failure_without_losing_registration(project, monkeypatch):
    monkeypatch.setattr(setup, "project_root", lambda _: project)
    monkeypatch.setattr(setup, "audit", lambda *a: ({}, {}))
    interpreter = project / "python"
    interpreter.touch()
    registration = project / ".cache/roboenv/installations/robocasa.json"
    registration.parent.mkdir(parents=True)
    content = json.dumps({"python": str(interpreter), "report": {}})
    registration.write_text(content)

    def fail(*a, **kw):
        raise OSError("download failed")

    monkeypatch.setattr(assets, "urlopen", fail)
    with pytest.raises(RuntimeError, match="rerun"):
        setup.prepare("robocasa")
    assert registration.read_text() == content
    assert interpreter.is_file()


def test_upstream_fixtures_readme_is_not_installed(project, monkeypatch):
    def download(*a, **kw):
        data = io.BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("fixtures/cabinet/model.xml", b"asset")
            archive.writestr("README.md", b"upstream documentation")
        data.seek(0)
        return data

    monkeypatch.setattr(assets, "urlopen", download)
    root = project / "third_party/robocasa/robocasa/models/assets"
    (root / "README.md").write_text("existing documentation")
    assert assets.prepare_assets(project)["downloaded_assets_verified"]
    assert (root / "fixtures/cabinet/model.xml").read_bytes() == b"asset"
    assert (root / "README.md").read_text() == "existing documentation"
    receipt = json.loads(
        (project / ".cache/roboenv/assets/robocasa/fixtures_lightwheel.json").read_text()
    )
    assert [item["path"] for item in receipt["files"]] == ["fixtures/cabinet/model.xml"]


def test_readme_only_is_not_a_valid_asset_bundle(project, monkeypatch):
    monkeypatch.setattr(assets, "urlopen", lambda *a, **kw: bundle("README.md"))
    with pytest.raises(RuntimeError, match="inspect .*private.log"):
        assets.prepare_assets(project)
    logfile = project / ".cache/roboenv/assets/robocasa/fixtures_lightwheel.private.log"
    assert "empty asset archive" in logfile.read_text()
    assert logfile.stat().st_mode & 0o777 == 0o600
