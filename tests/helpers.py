"""Small shared builders; each script load gets independent mutable module state."""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
