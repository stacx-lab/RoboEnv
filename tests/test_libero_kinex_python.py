"""Exercise the native acceptance script's Python preparation without a simulator."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import ROOT, load_script


@pytest.mark.parametrize("valid", [True, False])
def test_python_preparation_completes_or_fails_before_native_launch(tmp_path, monkeypatch, valid):
    kinex = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")).resolve()
    node = shutil.which(os.environ.get("NODE", "node"))
    if not node or not shutil.which("uv") or not (kinex / "packages/core/dist/index.js").is_file():
        pytest.skip("Requires the built Kinex core, Node and uv")
    smoke = load_script("tests/libero_kinex.py")
    out = tmp_path / "acceptance"

    def prepare(task, workspace, **kwargs):
        (workspace / ".kinex/python").mkdir(parents=True)
        (workspace / ".kinex/python/pyproject.toml").write_text(
            '[project]\nname = "acceptance-probe"\nversion = "0.0.0"\n'
            'requires-python = ">=3.11"\ndependencies = []\n'
            if valid else "invalid toml ["
        )

    class Prepared(Exception):
        pass

    def launch(*args, **kwargs):
        assert valid, "Native launch must not follow failed Python preparation"
        assert (out / "workspace/.kinex/python/.venv/bin/python3").is_file()
        assert (out / "workspace/.kinex/python/uv.lock").is_file()
        raise Prepared

    monkeypatch.setattr(smoke, "tool", lambda name: SimpleNamespace(prepare=prepare))
    monkeypatch.setattr(smoke.roboenv, "launch", launch)
    monkeypatch.setenv("UV_PYTHON", sys.executable)
    monkeypatch.setenv("UV_OFFLINE", "true")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_LINK_MODE", "copy")
    monkeypatch.setenv("UV_LOCKED", "false")
    with pytest.raises(Prepared if valid else subprocess.CalledProcessError):
        smoke.check(out, kinex, node)
    log = (out / "python.log").read_text()
    assert "Unknown argument: python" not in log
    if not valid:
        assert "uv sync failed" in log
