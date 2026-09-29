"""Asset depots and prepared interpreters resolve the same way for every backend."""

from roboenv.provisioning import behavior_assets, libero_assets
from roboenv.provisioning.assets import asset_cache


def test_asset_cache_honours_the_override(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBOENV_ASSET_CACHE", raising=False)
    assert asset_cache(tmp_path, "robotwin") == tmp_path / ".cache/roboenv/assets/robotwin"
    monkeypatch.setenv("ROBOENV_ASSET_CACHE", str(tmp_path / "depot"))
    assert asset_cache(tmp_path, "behavior") == tmp_path / "depot/behavior"


def test_asset_preparation_requires_a_prepared_interpreter(tmp_path, monkeypatch):
    # Both used to read installations/<backend>.json directly (KeyError or a missing file);
    # they now give the same guidance as launch.
    monkeypatch.setattr(behavior_assets, "acceptance_missing", lambda: [])
    for prepare in (libero_assets.prepare_assets, behavior_assets.prepare_assets):
        try:
            prepare(tmp_path)
        except ValueError as error:
            assert "roboenv prepare" in str(error)
        else:
            raise AssertionError("an unprepared simulator was accepted")
