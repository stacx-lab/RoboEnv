"""No-GPU integrity and isolation regressions for executable dependency depots."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.deployment import native_service

ROOT = Path(__file__).resolve().parents[1]
CONTAINER_PREFIX = "/opt/roboenv/runtime/dependencies"


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_verifier(tmp_path, monkeypatch):
    """Explicit physical roots let the real verifier run without writing container paths."""
    module = load_script("runtime/packaging/verify_dependencies.py")
    depot_root = tmp_path / "dependencies"
    depot_root.mkdir()

    real_verify = module.verify

    def verify(manifest):
        return real_verify(manifest, roots={manifest["target"]: depot_root / "sim-env"})

    monkeypatch.setattr(module, "verify", verify)
    return module, depot_root


def test_complete_snapshot_rejects_content_mode_extra_and_missing_files(tmp_path, monkeypatch):
    module, root = local_verifier(tmp_path, monkeypatch)
    depot = root / "sim-env"
    depot.mkdir()
    file = depot / "python"
    file.write_text("original-interpreter")
    file.chmod(0o755)
    manifest = module.snapshot(depot, CONTAINER_PREFIX + "/sim-env")
    module.verify(manifest)
    file.write_text("substituted-interpreter")
    with pytest.raises(ValueError, match="mismatch"):
        module.verify(manifest)
    file.write_text("original-interpreter")
    file.chmod(0o644)
    with pytest.raises(ValueError, match="mismatch"):
        module.verify(manifest)
    file.chmod(0o755)
    injected = depot / "sitecustomize.py"
    injected.write_text("raise RuntimeError('injected')")
    with pytest.raises(ValueError, match="mismatch"):
        module.verify(manifest)
    injected.unlink()
    file.unlink()
    with pytest.raises(ValueError, match="mismatch"):
        module.verify(manifest)


def test_snapshot_covers_link_targets_and_internal_target_contents(tmp_path, monkeypatch):
    module, root = local_verifier(tmp_path, monkeypatch)
    depot = root / "sim-env"
    depot.mkdir()
    target = depot / "python3"
    target.write_text("interpreter")
    link = depot / "python"
    link.symlink_to("python3")
    manifest = module.snapshot(depot, CONTAINER_PREFIX + "/sim-env")
    module.verify(manifest)
    assert manifest["files"]["python"] == {"symlink": "python3"}
    link.unlink()
    link.symlink_to("elsewhere")
    with pytest.raises(ValueError):
        module.verify(manifest)


@pytest.mark.parametrize("target", ["/tmp/unverified-interpreter", "../../unverified-interpreter"])
def test_snapshot_rejects_unverified_external_links(tmp_path, target):
    module = load_script("runtime/packaging/verify_dependencies.py")
    depot = tmp_path / "sim-env"
    depot.mkdir()
    (depot / "python").symlink_to(target)
    with pytest.raises(ValueError, match="(?i)link|escape|target"):
        module.snapshot(depot, CONTAINER_PREFIX + "/sim-env")


def test_snapshot_rejects_special_files(tmp_path):
    import os

    module = load_script("runtime/packaging/verify_dependencies.py")
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="special file"):
        module.snapshot(tmp_path, CONTAINER_PREFIX + "/sim-env")


@pytest.mark.parametrize("target", ["/tmp/interpreter", CONTAINER_PREFIX + "/../outside"])
def test_verifier_rejects_unapproved_mount_target(target):
    module = load_script("runtime/packaging/verify_dependencies.py")
    with pytest.raises(ValueError, match="target"):
        module.verify(
            {"schema": "roboenv/executable-depot/1", "target": target, "files": {"placeholder": {}}}
        )


def test_relocation_requires_staging_marker_and_rewrites_only_staging(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = {"replacements": {"SOURCE_PREFIX": "/opt/sim-env"}, "python_base": False}
    path = staging / "paths.pth"
    path.write_text("SOURCE_PREFIX/packages\n")
    with pytest.raises(ValueError, match="unmarked"):
        module.relocate(staging, manifest)
    assert path.read_text() == "SOURCE_PREFIX/packages\n"
    (staging / ".roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}'
    )
    binary = staging / "native.so"
    binary.write_bytes(b"SOURCE_PREFIX\x00binary")
    outside = tmp_path / "original.pth"
    outside.write_text("SOURCE_PREFIX/original")
    (staging / "outside.pth").symlink_to("../original.pth")
    module.relocate(staging, manifest)
    assert path.read_text() == "/opt/sim-env/packages\n"
    assert outside.read_text() == "SOURCE_PREFIX/original"
    assert binary.read_bytes() == b"SOURCE_PREFIX\x00binary"


def test_relocation_connects_base_interpreter_using_container_paths(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    (tmp_path / ".roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}'
    )
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").write_text("staged-original")
    module.relocate(tmp_path, {"replacements": {}, "python_base": True, "python_version": "3.11"})
    for name in ("python", "python3", "python3.11"):
        assert (tmp_path / "bin" / name).readlink() == Path("/opt/python/bin/python3.11")


def test_interpreter_staging_excludes_common_credential_locations():
    module = load_script("tools/build_simulator_image.py")
    secrets = {".env", ".env.local", ".ssh", ".aws", ".netrc", ".pypirc"}
    assert secrets <= module.ignore_interpreter_state("prefix", list(secrets | {"lib", "bin"}))


def test_python_base_uses_same_secret_exclusion_as_simulator_environment():
    tree = ast.parse((ROOT / "tools/build_simulator_image.py").read_text())
    copies = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copytree"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "base"
    ]
    assert len(copies) == 1
    ignore = next(keyword.value for keyword in copies[0].keywords if keyword.arg == "ignore")
    assert isinstance(ignore, ast.Name) and ignore.id == "ignore_interpreter_state"


def test_venv_python_alias_is_replaced_without_reading_its_original_target(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    (tmp_path / ".roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}'
    )
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").symlink_to("/unavailable/original/python3.12")
    module.relocate(
        tmp_path, {"replacements": {}, "python_base": True, "python_version": "3.12"}
    )
    assert (tmp_path / "bin/python").readlink() == Path("/opt/python/bin/python3.12")


def test_base_relocation_does_not_follow_a_bin_directory_link(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / ".roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}'
    )
    outside = tmp_path / "original"
    outside.mkdir()
    (outside / "python").write_text("untouched")
    (stage / "bin").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="real staging"):
        module.relocate(
            stage, {"replacements": {}, "python_base": True, "python_version": "3.12"}
        )
    assert (outside / "python").read_text() == "untouched"


def test_healthcheck_never_executes_depot_python_before_verification(tmp_path):
    manifest = tmp_path / "deployment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "robodojo",
                "legacy_tree_required": False,
                "image": "sha256:" + "a" * 64,
                "mounts": [],
                "environment": {},
            }
        )
    )
    task = SimpleNamespace(
        backend="robodojo", backend_task="fixture", seed=0, budget_mode="official"
    )
    service = native_service(manifest, task)
    health = service["healthcheck"]["test"]
    assert health[:2] == ["CMD", "/usr/bin/python3"]
    assert "/opt/sim-env/bin/python" not in health
    script = (ROOT / "src/roboenv/assets/serve-native.sh").read_text()
    assert script.index("/usr/bin/python3 -I") < script.index("exec /opt/sim-env/bin/python")


def test_cross_depot_links_require_every_terminal_to_be_verified(tmp_path):
    module = load_script("runtime/packaging/verify_dependencies.py")
    sim, base = tmp_path / "sim", tmp_path / "base"
    (sim / "bin").mkdir(parents=True)
    (base / "bin").mkdir(parents=True)
    terminal = base / "bin/python3.11"
    terminal.write_text("verified-interpreter")
    (sim / "bin/python").symlink_to("/opt/python/bin/python3.11")
    sim_target, base_target = CONTAINER_PREFIX + "/sim-env", CONTAINER_PREFIX + "/python-base"
    with pytest.raises(ValueError, match="verified"):
        module.snapshot(sim, sim_target)
    peers = {sim_target: sim, base_target: base}
    manifests = [module.snapshot(root, target, peers) for target, root in peers.items()]
    module.verify_all(manifests, roots=peers)
    with pytest.raises(ValueError, match="exactly one manifest"):
        module.verify_all(manifests[:1], roots=peers)
    terminal.write_text("replaced-interpreter")
    with pytest.raises(ValueError, match="mismatch"):
        module.verify_all(manifests, roots=peers)


def test_symlink_resolution_handles_components_before_parent_navigation(tmp_path):
    module = load_script("runtime/packaging/verify_dependencies.py")
    (tmp_path / "real/nested").mkdir(parents=True)
    (tmp_path / "real/payload").write_text("covered")
    (tmp_path / "jump").symlink_to("real/nested")
    (tmp_path / "result").symlink_to("jump/../payload")
    target = CONTAINER_PREFIX + "/sim-env"
    module.snapshot(tmp_path, target)
    graph = module._entries({target: tmp_path})
    assert module._resolve_link(target + "/result", graph) == target + "/real/payload"


@pytest.mark.parametrize("link_target", ["missing", "self", "/opt/python/missing"])
def test_dangling_cyclic_and_uncovered_alias_links_are_rejected(tmp_path, link_target):
    module = load_script("runtime/packaging/verify_dependencies.py")
    (tmp_path / "self").symlink_to(link_target)
    with pytest.raises(ValueError, match="link"):
        module.snapshot(tmp_path, CONTAINER_PREFIX + "/sim-env")


def test_relocation_rewrites_known_absolute_links_and_rejects_unknown_ones(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    (tmp_path / ".roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}'
    )
    (tmp_path / "python").symlink_to("/source/environment/bin/python3.11")
    module.relocate(tmp_path, {"replacements": {"/source/environment": "/opt/python"}})
    assert (tmp_path / "python").readlink() == Path("/opt/python/bin/python3.11")
    (tmp_path / "unsafe").symlink_to("/tmp/unverified")
    with pytest.raises(ValueError, match="safe relocation"):
        module.relocate(tmp_path, {"replacements": {}})


def test_relocation_rejects_symlinked_or_invalid_stage_marker(tmp_path):
    module = load_script("runtime/packaging/relocate.py")
    marker = tmp_path / ".roboenv-staged-environment.json"
    original = tmp_path / "original"
    original.write_text('{"schema":"roboenv/staged-environment/1"}')
    marker.symlink_to("original")
    with pytest.raises(ValueError, match="unmarked"):
        module.relocate(tmp_path, {"replacements": {}})
    marker.unlink()
    marker.write_text("{}")
    with pytest.raises(ValueError, match="staging marker"):
        module.relocate(tmp_path, {"replacements": {}})
