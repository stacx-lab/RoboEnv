import runpy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from roboenv_runtime.adapters.robocasa.assets import writable_object_xml

ROOT = Path(__file__).resolve().parents[1]


def test_transient_xml_preserves_asset_base_and_cleans_up_after_failure(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    before = list(assets.iterdir())
    xml = '<mujoco><asset><mesh file="mesh.obj"/><texture file="/opt/data/x.png"/></asset></mujoco>'
    with pytest.raises(RuntimeError):
        with writable_object_xml(xml, assets) as name:
            path = Path(name)
            assert not path.is_relative_to(assets)
            root = ElementTree.parse(path).getroot()
            assert root.find("./asset/mesh").get("file") == str(assets / "mesh.obj")
            assert root.find("./asset/texture").get("file") == "/opt/data/x.png"
            raise RuntimeError("simulated object initialization failure")
    assert not path.exists()
    assert list(assets.iterdir()) == before


def test_kit_generated_documents_and_env_files_are_not_staged():
    ignore = runpy.run_path(str(ROOT / "tools/build_simulator_image.py"))[
        "ignore_interpreter_state"
    ]
    assert ignore(
        "prefix/site-packages/isaacsim/kit/data", ["documents", "extensions", ".env", ".env.local"]
    ) == {"documents", ".env", ".env.local"}
    assert ignore("prefix/site-packages/package/data", ["documents", "extensions"]) == set()


def test_builder_checks_docker_storage_separately(monkeypatch):
    scope = runpy.run_path(str(ROOT / "tools/build_simulator_image.py"))
    monkeypatch.setattr(
        scope["subprocess"], "check_output", lambda *args, **kwargs: "/var/lib/docker"
    )
    monkeypatch.setattr(scope["shutil"], "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="Docker storage"):
        scope["check_docker_storage"](1024)


def test_kit_extension_is_copied_without_mutating_original_link(tmp_path):
    import shutil

    scope = runpy.run_path(str(ROOT / "tools/build_simulator_image.py"))
    root = tmp_path / "prepared"
    env = root / "env"
    extension = root / "omni-home/.local/share/ov/data/exts/v2/test-extension"
    (extension / "config").mkdir(parents=True)
    (extension / "config/extension.toml").write_text('[package]\nversion="1.0.0"\n')
    (extension / "module.py").write_text("value = 1\n")
    relative = (
        "lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/test-extension"
    )
    original = env / relative
    original.parent.mkdir(parents=True)
    original.symlink_to(extension, target_is_directory=True)
    duplicate = original.parent / "same-extension-alias"
    duplicate.symlink_to(extension, target_is_directory=True)
    stage = tmp_path / "stage"
    shutil.copytree(env, stage, symlinks=True)
    receipt = scope["bundle_kit_extensions"](stage, env, root)
    assert len(receipt) == 1 and original.is_symlink()
    assert (stage / duplicate.relative_to(env)).resolve() == (stage / relative).resolve()
    assert not (stage / relative).readlink().is_absolute()
    assert (stage / relative / "module.py").read_text() == "value = 1\n"
    (extension / "module.py").write_text("original changed later\n")
    assert (stage / relative / "module.py").read_text() == "value = 1\n"
