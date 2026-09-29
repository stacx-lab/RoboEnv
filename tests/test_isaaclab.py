"""Isaac Lab remains installable without a bundled G1 task or runtime."""

import json
from pathlib import Path

import pytest
from roboenv.tasks.catalog import _isaaclab
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.core.adapter import BUILTIN_ADAPTERS

ROOT = Path(__file__).resolve().parents[1]


def test_isaaclab_keeps_installation_recipe_without_g1():
    recipes = json.loads((ROOT / "runtime/environments.json").read_text())
    assert "isaaclab" in recipes["backends"]
    assert recipes["backends"]["isaaclab"]["import_probes"] == ["isaaclab"]
    assert list(_isaaclab(ROOT, [])) == []
    assert "isaaclab" not in BUILTIN_ADAPTERS
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_task("isaaclab/g1-pick-place", project=ROOT)
