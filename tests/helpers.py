"""Small shared builders; each script load gets independent mutable module state."""

import importlib.util
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_kinex_world_budget(budget):
    """Check a world budget against Kinex's strict contract: `{elapsed_s, max_s}`.

    Kinex 0.9.1 parses it as a strict object with a finite nonnegative `elapsed_s`
    and a finite positive or null `max_s`; other keys, and the former
    `{used, max, unit}` shape, are rejected. Booleans are not numbers.
    """
    assert set(budget) == {"elapsed_s", "max_s"}, budget
    elapsed, cap = budget["elapsed_s"], budget["max_s"]
    assert type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, budget
    assert cap is None or (type(cap) in (int, float) and math.isfinite(cap) and cap > 0), budget


def kinex_sources():
    """The sibling Kinex checkout and the import prefix for its device schema.

    Kinex moved its robotics core under packages/core, so accept either layout.
    Returns (root, prefix); prefix is None when no schema source is present.
    """
    root = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex"))
    for prefix in ("packages/core/src", "src"):
        if (root / prefix / "device/profile/schema.ts").is_file():
            return root, "./" + prefix
    return root, None


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tool(name, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return load_script(f"tools/{name}.py")
