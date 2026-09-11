"""Fail-closed installation and portable asset inventory contracts."""

import json
import shutil
from pathlib import Path

import pytest
from helpers import load_tool
from roboenv.provisioning import setup
from roboenv.provisioning.assets import create, inventory, verify

ROOT = Path(__file__).resolve().parents[1]


def test_robocasa_uses_resolved_lock():
    recipe = json.loads((ROOT / "runtime/environments.json").read_text())["backends"]["robocasa"]
    assert recipe["lock"] == "runtime/locks/robocasa-resolved.txt"
    assert recipe["index_url"] == "https://pypi.org/simple"
    resolved = setup._requirements(ROOT / recipe["lock"])
    assert str(resolved["gym-notices"].specifier) == "==0.1.0"
    assert {"jsonschema", "jsonschema-specifications", "packaging", "referencing", "rpds-py"} <= (
        resolved.keys()
    )


def test_prepare_default_install_and_reuse(tmp_path, monkeypatch):
    from roboenv.provisioning import setup
    from roboenv.provisioning import sources as upstreams

    monkeypatch.setattr(setup, "project_root", lambda project: tmp_path)
    monkeypatch.setattr(setup, "audit", lambda *args: ({}, {}))
    calls = []
    monkeypatch.setattr(upstreams, "sync", lambda *args: calls.append("sync"))
    monkeypatch.setattr(
        setup, "install", lambda backend, destination, **kwargs: calls.append(destination)
    )
    progress = []
    monkeypatch.setattr(setup, "_prepare_assets", lambda backend, root, report, progress: report)
    setup.prepare("robotwin", progress=progress.append)
    assert any("Synchronizing" in message for message in progress)
    assert calls == ["sync", tmp_path / ".venv-sim/robotwin"]
    calls.clear()
    interpreter = tmp_path / ".venv-sim/robotwin/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    registration = tmp_path / ".cache/roboenv/installations/robotwin.json"
    registration.parent.mkdir(parents=True)
    registration.write_text(json.dumps({"python": str(interpreter), "report": {"reused": True}}))
    assert setup.prepare("robotwin", progress=progress.append) == {"reused": True}
    assert "Reusing registered interpreter" in progress[-1]
    assert not calls
    interpreter.unlink()
    with pytest.raises(ValueError, match="existing environments are preserved"):
        setup.prepare("robotwin")
    assert not calls


def test_prepare_rejects_unknown_backend_without_effects(tmp_path):
    from roboenv.provisioning import setup

    with pytest.raises(ValueError, match="unknown simulator"):
        setup.prepare("../unknown", project=ROOT, install_to=tmp_path / "unused")
    assert list(tmp_path.iterdir()) == []


def test_dependency_lock_rejects_duplicates_ranges_and_unpinned_git(tmp_path):
    lock = tmp_path / "bad.txt"
    lock.write_text(
        "package-one==1\npackage_one==2\nloose>=1\n"
        "unfixed @ git+https://github.com/example/source.git@main\n"
    )
    result = setup.audit_lock(lock)
    assert not result["passed"]
    assert {item["reason"] for item in result["issues"]} == {
        "duplicate-package",
        "not-exactly-pinned",
        "unresolved-source",
    }


def test_all_published_freezes_have_unambiguous_pins():
    auditor = setup.audit_lock
    for lock in (ROOT / "runtime/locks").glob("*.txt"):
        assert auditor(lock)["passed"], lock.name


@pytest.fixture
def recipe_project(tmp_path):
    """Real recipe metadata without any developer checkout or simulator assets."""
    project = tmp_path / "project"
    (project / "src/roboenv").mkdir(parents=True)
    recipes = json.loads((ROOT / "runtime/environments.json").read_text())
    files = {
        "pyproject.toml",
        "runtime/pyproject.toml",
        "runtime/environments.json",
        "upstreams.lock.json",
    }
    for recipe in recipes["backends"].values():
        files.update(path for path in (recipe.get("lock"), recipe.get("source_pins")) if path)
        if recipe.get("recorded_validation"):
            files.add(recipe["recorded_validation"]["evidence"])
    for relative in files:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copytree(ROOT / "third_party/patches", project / "third_party/patches")
    return project, recipes["backends"]


def test_recipe_preflight_is_not_an_installation_or_replay_certificate(
    recipe_project,
    monkeypatch,
):
    project, recipes = recipe_project
    # This test isolates certificate semantics, not Git/source installation.
    # Both readiness collaborators are explicit: verified Git status and the
    # declared checkout directories. Neither depends on local submodule state.
    monkeypatch.setattr(setup, "status", lambda *args: {"ready": True})
    for backend in ("libero", "robotwin"):
        recipe = recipes[backend]
        for relative in recipe["editables"] + recipe.get("source_only", []):
            (project / relative).mkdir(parents=True, exist_ok=True)
        report, _ = setup.audit(backend, project=project)
        assert report["install_preflight_passed"]
        assert not report["clean_install_verified"]
        assert not report["gpu_replay_verified"]
        assert not report["recorded_recipe_validation"]
        assert bool(report["source_only"]) is (backend == "robotwin")
        assert report["source_pins_audit"]["passed"]
    behavior, _ = setup.audit("behavior", project=project)
    assert not behavior["install_preflight_passed"]
    assert not behavior["clean_install_verified"]
    assert not behavior["gpu_replay_verified"]


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_missing_checkouts_fail_despite_valid_locks(recipe_project, backend):
    project, _ = recipe_project
    # No status stub: exercise the actual source auditor on a metadata-only tree.
    report, _ = setup.audit(backend, project=project)
    assert report["lock_audit"]["passed"]
    assert report["source_pins_audit"]["passed"]
    assert not report["recorded_recipe_validation"]
    assert not report["sources_ready"]
    assert not report["install_preflight_passed"]
    assert "missing-local-package-checkout" in report["blockers"]
    assert any(item.startswith("source-check-failed:") for item in report["blockers"])
    assert not report["clean_install_verified"]
    assert not report["gpu_replay_verified"]


@pytest.mark.parametrize(
    "override",
    [
        "numpy>=1.26",
        "numpy==1.26.4",
        "jaxls @ git+https://github.com/other/source.git@" + "a" * 40,
        "jaxls @ git+https://github.com/example/source.git@" + "b" * 40,
        "jaxls @ git+https://github.com/example/source.git@main",
        "jaxls[dev] @ git+https://github.com/example/source.git@" + "a" * 40,
    ],
)
def test_source_override_cannot_rewrite_version_ranges_or_locked_identity(tmp_path, override):
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "numpy==1.26.4\njaxls @ git+https://github.com/example/source.git@" + "a" * 40 + "\n"
    )
    pins = tmp_path / "pins.txt"
    pins.write_text(override + "\n")
    result = setup.audit_source_pins(lock, pins)
    assert result["passed"] is False


def test_source_override_accepts_only_the_exact_locked_git_pin(tmp_path):
    line = "jaxls @ git+https://github.com/example/source.git@" + "a" * 40 + "\n"
    lock, pins = tmp_path / "lock.txt", tmp_path / "pins.txt"
    lock.write_text(line)
    pins.write_text(line)
    result = setup.audit_source_pins(lock, pins)
    assert result["passed"]
    assert result["pins"]["jaxls"]["revision"] == "a" * 40


def installer_fixture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "runtime").mkdir(parents=True)
    (project / "src/roboenv").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname="roboenv"\nversion="0.2.0"\n')
    (project / "runtime/pyproject.toml").write_text(
        '[project]\nname="roboenv-runtime"\nversion="0.2.0"\n'
    )
    (project / "lock.txt").write_text(
        "jaxls @ git+https://github.com/example/source.git@" + "a" * 40 + "\n"
    )
    (project / "pins.txt").write_bytes((project / "lock.txt").read_bytes())
    (project / "third_party/RoboTwin").mkdir(parents=True)
    recipe = {
        "python": "3.10",
        "lock": "lock.txt",
        "source_pins": "pins.txt",
        "source_only": ["third_party/RoboTwin"],
        "editables": [],
        "extra_indexes": [],
    }
    monkeypatch.setattr(
        setup,
        "audit",
        lambda *args: ({"install_preflight_passed": True, "gpu_replay_verified": False}, recipe),
    )
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    calls = []
    monkeypatch.setattr(
        setup.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    return setup, project, calls


@pytest.mark.parametrize("cache", [None, "shared-package-cache"])
def test_installer_resolves_sources_once_without_installing_source_only_checkout(
    tmp_path, monkeypatch, cache
):
    setup, project, calls = installer_fixture(tmp_path, monkeypatch)
    if cache is None:
        monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("UV_CACHE_DIR", cache)
    monkeypatch.setenv("PYTHONPATH", "unselected-checkout")
    monkeypatch.setenv("UV_NO_DEPS", "1")
    monkeypatch.setenv("PIP_NO_DEPS", "1")
    progress = []
    result = setup.install(
        "robotwin", tmp_path / "new-env", project=project, progress=progress.append
    )
    assert any("Detailed installation log: .cache/setup/" in line for line in progress)
    assert any("Installing dependencies" in line for line in progress)
    assert any("Verifying imports" in line for line in progress)
    assert "Interpreter registered" in progress[-1]
    assert str(tmp_path) not in "\n".join(progress)
    installation = next(command for command, _ in calls if command[1:3] == ["pip", "install"])
    assert installation.count("--override") == 1
    assert str(project / "pins.txt") in installation
    assert str(project / "third_party/RoboTwin") not in installation
    assert str(project) + "[server]" in installation
    assert str(project / "runtime") in installation
    assert "--no-deps" not in installation and "--no-build-isolation" not in installation
    for _, kwargs in calls:
        assert not {"PYTHONPATH", "UV_NO_DEPS", "PIP_NO_DEPS"} & kwargs["env"].keys()
        assert kwargs["env"]["ROBOENV_ROOT"] == str(project)
        assert kwargs["env"].get("UV_CACHE_DIR") == cache
    assert calls[-1][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["clean_install_verified"]
    assert not result["gpu_replay_verified"]
    assert str(tmp_path) not in json.dumps(result)


def test_installer_accepts_explicit_wheels_and_rejects_ambiguous_mixed_inputs(
    tmp_path,
    monkeypatch,
):
    setup, project, calls = installer_fixture(tmp_path, monkeypatch)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    sdk, runtime = (
        wheels / "roboenv-0.2.0-py3-none-any.whl",
        wheels / "roboenv_runtime-0.2.0-py3-none-any.whl",
    )
    sdk.write_bytes(b"test-only wheel")
    runtime.write_bytes(b"test-only wheel")
    report = setup.install("robotwin", tmp_path / "env", project=project, wheels=wheels)
    command = next(command for command, _ in calls if command[1:3] == ["pip", "install"])
    assert str(sdk) in command and str(runtime) in command
    assert "-e" not in command
    assert all(item["kind"] == "wheel" for item in report["package_inputs"])
    with pytest.raises(ValueError, match="cannot be combined"):
        setup.install("robotwin", tmp_path / "other", project=project, sdk=sdk, wheels=wheels)
    with pytest.raises(ValueError, match="not the roboenv distribution"):
        setup.install("robotwin", tmp_path / "wrong", project=project, sdk=runtime)


def test_recipe_paths_cannot_escape_the_selected_checkout(tmp_path):
    for path in ("../outside.txt", "/etc/config", "nested\\outside"):
        with pytest.raises(ValueError, match="relative"):
            setup._project_path(tmp_path, path)


def test_installer_never_touches_existing_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "audit", lambda *args: ({"install_preflight_passed": True}, {}))
    monkeypatch.setattr(setup.shutil, "which", lambda name: "uv")
    with pytest.raises(ValueError, match="existing environment"):
        setup.install("robocasa", tmp_path)


def test_contract_dependency_destination_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        load_tool("contract_dependencies").prepare_contract_dependencies("python", tmp_path)


def test_asset_receipt_is_portable_and_detects_changes(tmp_path):
    depot = tmp_path / "assets"
    depot.mkdir()
    (depot / "robot.xml").write_text("<robot/>\n")
    (depot / "same.xml").symlink_to("robot.xml")
    receipt = create(
        depot,
        asset_id="example-assets",
        source_url="https://github.com/example/assets",
        source_revision="v1",
        license_url="https://github.com/example/assets/blob/v1/LICENSE",
    )
    assert str(tmp_path) not in json.dumps(receipt)
    assert receipt["redistribution_reviewed"] is False
    manifest = tmp_path / "receipt.json"
    manifest.write_text(json.dumps(receipt))
    assert verify(depot, manifest)["passed"]
    (depot / "robot.xml").write_text("changed")
    assert not verify(depot, manifest)["passed"]


def test_asset_inventory_rejects_link_escape_and_source_checkout(tmp_path):
    depot = tmp_path / "assets"
    depot.mkdir()
    (tmp_path / "outside").write_text("private")
    (depot / "escape").symlink_to("../outside")
    with pytest.raises(ValueError, match="escapes"):
        inventory(depot)
    (depot / "escape").unlink()
    (depot / "pyproject.toml").write_text("[project]\n")
    with pytest.raises(ValueError, match="data-only"):
        inventory(depot)


def test_asset_provenance_never_accepts_tokenized_urls(tmp_path):
    with pytest.raises(ValueError, match="credentials or tokens"):
        create(
            tmp_path,
            asset_id="assets",
            source_url="https://example.org/data?token=secret",
            source_revision="1",
            license_url="https://example.org/license",
        )
