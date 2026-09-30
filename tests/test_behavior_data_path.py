"""Launch selects the prepared depot without overriding an owner's dataset."""

import os

import pytest
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
