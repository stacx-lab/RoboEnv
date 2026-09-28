"""Domain packages stay independent of CLI and simulator runtime."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "src/roboenv"


def imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.module or ""
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_domain_packages_do_not_depend_on_cli_or_simulator_runtime():
    for package in ("devices", "tasks", "provisioning", "harbor", "control"):
        for path in (SDK / package).rglob("*.py"):
            for module in imports(path):
                assert module != "roboenv.cli", path
                assert not module.startswith(("roboenv.cli.", "roboenv_runtime")), path
                if package == "tasks":
                    assert not module.startswith("roboenv.harbor"), path
