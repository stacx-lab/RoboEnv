"""Package ownership and deduplication stay explicit as the codebase grows."""

import ast
import importlib
from pathlib import Path

from roboenv_runtime.bindings.simulators import binding_index

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


def test_robot_variants_share_handlers_but_keep_separate_binding_records():
    index = binding_index()
    for device in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
        for backend, first, second in (
            ("robotwin", "piper", "aloha_agilex"),
            ("robodojo", "arx_x5", "yam"),
        ):
            a, b = index[f"{backend}:{first}:{device}"], index[f"{backend}:{second}:{device}"]
            assert a is not b
            assert a["module"] == b["module"]
            assert a["source_sha256"] == b["source_sha256"]
    cameras = [record["module"] for key, record in index.items() if key.endswith(":camera")]
    assert len(cameras) == 2  # RoboCasa and LIBERO bind cameras directly to native rendering.
    assert set(cameras) == {"shared.camera"}


def test_binding_implementations_have_no_duplicate_bodies():
    bodies = set()
    for name in sorted({record["module"] for record in binding_index().values()}):
        module = importlib.import_module("roboenv_runtime.bindings.backends." + name)
        tree = ast.parse(Path(module.__file__).read_text())
        if ast.get_docstring(tree):
            tree.body.pop(0)
        body = ast.dump(tree)
        assert body not in bodies, name
        bodies.add(body)
