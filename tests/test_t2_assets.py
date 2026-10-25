"""Pinned download integrity and non-destructive preparation without network access."""

import hashlib
import io
import zipfile

import pytest
from roboenv.provisioning import booster_t2_assets as assets


def test_downloader_uses_immutable_sources_and_extracts_only_pinned_members(tmp_path, monkeypatch):
    data = b"robot"
    scene = b"house"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scenes/hssd/42/42.usd", scene)
        archive.writestr("../../outside", b"ignored")
    archive = buffer.getvalue()
    def sha(x):
        return hashlib.sha256(x).hexdigest()
    pin = {
        "files": {"assets/robot": sha(data)},
        "scene_files": {"scenes/hssd/42/42.usd": sha(scene)},
        "source_revision": "a" * 40,
        "scene_source": {
            "repository": "owner/data",
            "revision": "b" * 40,
            "downloads": {"scenes_hssd_42.zip": sha(archive)},
        },
    }
    monkeypatch.setattr(assets, "asset_pin", lambda: pin)
    requested = []

    def download(url, destination, expected):
        requested.append(url)
        value = archive if url.endswith(".zip") else data
        assert sha(value) == expected
        destination.write_bytes(value)

    monkeypatch.setattr(assets, "_download", download)
    assert assets.download_assets(tmp_path)["files"] == 2
    assert any("ref=" + "a" * 40 in url for url in requested)
    assert any("/" + "b" * 40 + "/" in url for url in requested)
    assert not (tmp_path.parent / "outside").exists()
    requested.clear()
    assets.download_assets(tmp_path)
    assert not requested
    (tmp_path / "assets/robot").write_bytes(b"existing work")
    with pytest.raises(ValueError, match="Existing T2 asset differs"):
        assets.download_assets(tmp_path)
    assert (tmp_path / "assets/robot").read_bytes() == b"existing work"


def test_checksum_mismatch_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad bytes"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets._download("https://example.invalid/pinned", tmp_path / "part", "0" * 64)
