"""Small shared builders; each script load gets independent mutable module state."""

import importlib.util
import json
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


def native_interpreter():
    """The prepared RoboCasa interpreter, or skip: simulator deps never enter the SDK."""
    registration = ROOT / ".cache/roboenv/installations/robocasa.json"
    interpreter = os.environ.get("ROBOENV_ROBOCASA_TEST_PYTHON")
    if interpreter is None and registration.is_file():
        interpreter = json.loads(registration.read_text())["python"]
    if not interpreter or not Path(interpreter).is_file():
        import pytest  # helpers also load inside simulator interpreters

        pytest.skip("Requires prepared RoboCasa interpreter; never install simulator deps in SDK")
    return interpreter


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tool(name, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return load_script(f"tools/{name}.py")


def kinex_python():
    """Python assets from the explicitly selected or sibling Kinex checkout."""
    root = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")).expanduser().resolve()
    library = root / "packages/core/assets/python"
    if "KINEX_ROOT" in os.environ and not (library / "kinex").is_dir():
        raise FileNotFoundError(f"Selected Kinex Python library is missing: {library}")
    return library
