"""Launch and image building use the same prepared-interpreter selection."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import load_tool
from roboenv.provisioning.interpreter import resolve_interpreter

ROOT = Path(__file__).resolve().parents[1]


def register(root, record):
    path = root / ".cache/roboenv/installations/libero.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_explicit_interpreter_overrides_invalid_registration(tmp_path):
    register(tmp_path, {})
    assert resolve_interpreter(tmp_path, "libero", sys.executable) == Path(sys.executable)


@pytest.mark.parametrize("record", [None, {}, [], {"python": None}, {"python": ""},
                                    {"python": sys.executable},
                                    {"python": "/missing/python", "backend": "libero"},
                                    {"python": sys.executable, "backend": "robocasa"}])
def test_missing_and_invalid_installations_give_preparation_guidance(tmp_path, record):
    if record is not None:
        register(tmp_path, record)
    with pytest.raises(ValueError, match="prepare libero.*--interpreter"):
        resolve_interpreter(tmp_path, "libero")


def test_nonexecutable_interpreter_is_rejected(tmp_path):
    python = tmp_path / "python"
    python.touch(mode=0o600)
    register(tmp_path, {"python": str(python), "backend": "libero"})
    with pytest.raises(ValueError, match="existing executable"):
        resolve_interpreter(tmp_path, "libero")


@pytest.mark.parametrize("explicit", [False, True])
def test_image_staging_uses_prepared_interpreter_without_catalog_presets(
    tmp_path, monkeypatch, explicit,
):
    builder = load_tool("build_simulator_image")
    project = tmp_path / "project"
    dependencies = tmp_path / "assets"
    dependencies.mkdir()
    environment = tmp_path / "prepared-env"
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport json\nprint(json.dumps([[3, 12], {str(environment)!r}]))\n"
    )
    python.chmod(0o700)
    register(project, {} if explicit else {"backend": "libero", "python": str(python)})
    shutil.copytree(ROOT / "runtime/packaging", project / "runtime/packaging")
    (project / "third_party/LIBERO").mkdir(parents=True)
    (project / "upstreams.lock.json").write_text("{}")
    wheel = tmp_path / "runtime.whl"
    wheel.write_bytes(b"test-only wheel")
    out = tmp_path / "staged"
    monkeypatch.setattr(builder, "WORKSPACE", project)
    monkeypatch.setattr(builder, "status", lambda *args: {"ready": True, "sources": []})
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    monkeypatch.setattr(builder.runpy, "run_path", lambda path: {
        "prepare_contract_dependencies": lambda *args: {},
    })
    arguments = [
        "builder", "libero", "--dependencies", str(dependencies), "--runtime-wheel", str(wheel),
        "--out", str(out), "--tag", "test:stage", "--stage-only",
    ]
    if explicit:
        arguments += ["--interpreter", str(python)]
    monkeypatch.setattr(sys, "argv", arguments)
    builder.main()
    assert (out / "context/sim-env/bin/python").read_bytes() == python.read_bytes()
    manifest = json.loads((out / "deployment.json").read_text())
    assert manifest["runtime_options"] == {}
    assert manifest["interpreter"] == "/opt/sim-env/bin/python"
    assert not (project / "catalog").exists()
