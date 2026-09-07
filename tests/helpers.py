"""Small shared builders; each script load gets independent mutable module state."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tool(name, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return load_script(f"tools/{name}.py")
