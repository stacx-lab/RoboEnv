"""Launch selects the prepared depot without overriding an owner's dataset."""

import os
from pathlib import Path

import pytest
from roboenv_runtime.adapters.behavior import backend
from roboenv_runtime.adapters.behavior.backend import fix_cache_env


@pytest.mark.parametrize("prepared,override", [(True, None), (True, "custom"), (False, None)])
def test_prepared_dataset_default(tmp_path, monkeypatch, prepared, override):
    monkeypatch.setattr("roboenv_runtime.paths.repo_root", lambda: tmp_path)
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME",
                 "OMNIGIBSON_DATA_PATH"):
        monkeypatch.delenv(name, raising=False)
    depot = tmp_path / "third_party/BEHAVIOR-1K/OmniGibson/datasets"
    if prepared:
        depot.mkdir(parents=True)
    if override:
        monkeypatch.setenv("OMNIGIBSON_DATA_PATH", override)
    fix_cache_env()
    assert os.environ.get("OMNIGIBSON_DATA_PATH") == (
        override or (str(depot) if prepared else None)
    )
    assert depot.exists() == prepared


def isaac_tree(root):
    """The parts of an Isaac Sim installation that matter here: ``apps`` and its siblings."""
    apps = root / "isaacsim/apps"
    apps.mkdir(parents=True)
    (apps / "isaacsim.exp.base.kit").write_text("[package]\n")
    (apps / "omnigibson_5_1_0.kit").write_text("stale\n")
    (root / "isaacsim/exts").mkdir()
    (root / "isaacsim/VERSION").write_text("5.1.0\n")
    return apps


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_isaac_apps_get_a_writable_stand_in(tmp_path, monkeypatch):
    monkeypatch.setattr("roboenv_runtime.paths.repo_root", lambda: tmp_path)
    monkeypatch.delenv("EXP_PATH", raising=False)
    real = isaac_tree(tmp_path / "depot")
    monkeypatch.setattr(backend, "isaac_apps", lambda: real)
    real.chmod(0o555)
    try:
        fix_cache_env()
        apps = Path(os.environ["EXP_PATH"])
        assert apps == tmp_path / ".cache/behavior/isaacsim/apps"
        # Kit resolves the extension folders relative to the experience file.
        assert (apps.parent / "exts").resolve() == (real.parent / "exts").resolve()
        assert (apps.parent / "VERSION").read_text() == "5.1.0\n"
        assert (apps / "isaacsim.exp.base.kit").is_symlink()
        # OmniGibson writes its own files there at every launch; a link would write through.
        assert not (apps / "omnigibson_5_1_0.kit").exists()
        (apps / "omnigibson_5_1_0.kit").write_text("fresh\n")
        monkeypatch.delenv("EXP_PATH")
        fix_cache_env()
        assert not (apps / "omnigibson_5_1_0.kit").exists()
        assert (real / "omnigibson_5_1_0.kit").read_text() == "stale\n"
    finally:
        real.chmod(0o755)


def test_writable_isaac_apps_are_left_to_isaac_sim(tmp_path, monkeypatch):
    monkeypatch.setattr("roboenv_runtime.paths.repo_root", lambda: tmp_path)
    monkeypatch.delenv("EXP_PATH", raising=False)
    real = isaac_tree(tmp_path / "depot")
    monkeypatch.setattr(backend, "isaac_apps", lambda: real)
    fix_cache_env()
    assert "EXP_PATH" not in os.environ
    assert not (tmp_path / ".cache/behavior").exists()
