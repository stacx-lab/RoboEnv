"""Static robot setup is portable, task-independent, and preserves existing work."""

import json
import shutil
from pathlib import Path

import pytest
from roboenv.cli.main import main
from roboenv.devices.workspace import list_robots, setup_robot

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
ROBOT = "robocasa/robocasa-panda-omron"


def snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    destination = tmp_path / "workspace"
    destination.mkdir()
    monkeypatch.chdir(destination)
    return destination


def test_every_catalog_robot_exports_unchanged(workspace, monkeypatch):
    monkeypatch.delenv("ROBOENV_PROJECT", raising=False)
    monkeypatch.delenv("ROBOENV_ROOT", raising=False)
    robots = list_robots()
    assert robots == sorted(
        path.parent.relative_to(CATALOG / "robots").as_posix()
        for path in (CATALOG / "robots").rglob("workspace.yaml")
    )
    for robot in robots:
        result = setup_robot(robot, force=True)
        source = CATALOG / "robots" / robot
        assert result["robot"] == robot
        assert result["destination"] == str(workspace / ".kinex")
        assert result["devices"] == sorted(p.name for p in (source / "devices").iterdir()
                                           if p.is_dir() and p.name != "__pycache__")
        assert snapshot(workspace / ".kinex") == snapshot(source)
        assert not list((workspace / ".kinex").rglob("__pycache__"))
    assert {p.name for p in workspace.iterdir()} == {".kinex"}


def test_existing_workspace_and_forced_replacement(workspace):
    config = workspace / ".kinex"
    for name in ("python/local.txt", "recordings/take.json", "tools/script.py"):
        path = config / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep")
    (workspace / "AGENTS.md").write_text("User instructions")
    unrelated = snapshot(workspace)
    setup_robot(ROBOT, project=ROOT)
    before = snapshot(workspace)
    with pytest.raises(ValueError, match="--force"):
        setup_robot("libero/libero-panda-tabletop", project=ROOT)
    assert snapshot(workspace) == before
    setup_robot("libero/libero-panda-tabletop", project=ROOT, force=True)
    assert not (config / "devices/torso").exists()
    assert snapshot(config / "devices") == snapshot(
        CATALOG / "robots/libero/libero-panda-tabletop/devices"
    )
    for name, content in unrelated.items():
        assert (workspace / name).read_bytes() == content


@pytest.mark.parametrize("existing", ["workspace.yaml", "devices"])
def test_partial_configuration_is_protected(workspace, existing):
    config = workspace / ".kinex"
    config.mkdir()
    if existing == "devices":
        (config / existing).mkdir()
    else:
        (config / existing).write_text("keep")
    with pytest.raises(ValueError, match="--force"):
        setup_robot(ROBOT, catalog=CATALOG)
    assert {p.name for p in config.iterdir()} == {existing}
    setup_robot(ROBOT, catalog=CATALOG, force=True)
    assert snapshot(config) == snapshot(CATALOG / "robots" / ROBOT)


@pytest.mark.parametrize("reference", [
    "../robot", "/robot", "robocasa/../robot", "robocasa//robot", "robocasa/./robot",
    "robocasa\\robot", "robocasa/unknown", "robocasa", "",
])
def test_invalid_robot_has_no_effects(workspace, reference):
    with pytest.raises(ValueError, match="roboenv setup --list"):
        setup_robot(reference, catalog=CATALOG)
    assert not list(workspace.iterdir())


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "catalog"
    shutil.copytree(CATALOG / "robots" / ROBOT, root / "robots" / ROBOT)
    return root


def test_explicit_catalog_and_declared_resources(workspace, catalog):
    robot = catalog / "robots" / ROBOT
    device = robot / "devices/arm"
    (device / "geometry").mkdir()
    (device / "geometry/model.urdf").write_text("<robot/>")
    (device / "driver-resources.json").write_text(json.dumps({
        "schema": "roboenv/driver-resources/1", "files": ["driver.py", "geometry/model.urdf"],
    }))
    (device / "connection.json").write_text('{"session": "private"}')
    (robot / "bindings.yaml").write_text("not public")
    (robot / "devices/__pycache__").mkdir()
    (robot / "devices/__pycache__/transport.pyc").write_bytes(b"not portable")
    (catalog / "tasks").mkdir()
    (catalog / "tasks/task.yaml").write_text("invalid task; setup must not read this")
    assert list_robots(catalog=catalog) == [ROBOT]
    setup_robot(ROBOT, catalog=catalog)
    copied = workspace / ".kinex/devices/arm"
    assert (copied / "geometry/model.urdf").read_text() == "<robot/>"
    assert (copied / "driver-resources.json").read_bytes() == (
        device / "driver-resources.json"
    ).read_bytes()
    assert not (copied / "connection.json").exists()
    assert not (workspace / ".kinex/bindings.yaml").exists()


@pytest.mark.parametrize("resource,content", [
    ("workspace.yaml", "[]"),
    ("devices/arm/profile.yaml", "name: wrong\ntype: joint\n"),
    ("devices/arm/profile.yaml", "[invalid"),
    ("devices/arm/driver.py", None),
    ("devices/_roboenv.py", None),
    ("devices/arm/driver-resources.json", '{"schema": "unknown", "files": []}'),
])
def test_bad_resources_preserve_existing_configuration(workspace, catalog, resource, content):
    setup_robot(ROBOT, catalog=CATALOG)
    before = snapshot(workspace)
    target = catalog / "robots" / ROBOT / resource
    if content is None:
        target.unlink()
    else:
        target.write_text(content)
    with pytest.raises(ValueError):
        setup_robot(ROBOT, catalog=catalog, force=True)
    assert snapshot(workspace) == before
    assert {p.name for p in workspace.iterdir()} == {".kinex"}


@pytest.mark.parametrize("resource", [".kinex", ".kinex/workspace.yaml", ".kinex/devices"])
def test_destination_symlinks_are_rejected(workspace, tmp_path, resource):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = workspace / resource
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        setup_robot(ROBOT, catalog=CATALOG, force=True)
    assert link.is_symlink()
    assert not list(target.iterdir())


@pytest.mark.parametrize("resource", ["workspace.yaml", "devices/arm/profile.yaml",
                                      "devices/arm/driver.py", "devices/_roboenv.py"])
def test_source_symlinks_are_rejected(workspace, catalog, resource):
    target = catalog / "robots" / ROBOT / resource
    target.unlink()
    target.symlink_to(CATALOG / "robots" / ROBOT / resource)
    with pytest.raises(ValueError, match="symlinks"):
        setup_robot(ROBOT, catalog=catalog)
    assert not list(workspace.iterdir())


def test_destination_inside_source_is_rejected(catalog, monkeypatch):
    monkeypatch.chdir(catalog / "robots" / ROBOT)
    before = snapshot(catalog)
    with pytest.raises(ValueError, match="overlap"):
        setup_robot(ROBOT, catalog=catalog)
    assert snapshot(catalog) == before


@pytest.mark.parametrize("resource", [".kinex", ".kinex/devices", ".kinex/workspace.yaml"])
def test_wrong_destination_types_are_rejected(workspace, resource):
    target = workspace / resource
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name == "workspace.yaml":
        target.mkdir()
    else:
        target.write_text("keep")
    before = snapshot(workspace)
    with pytest.raises(ValueError, match="directory|file type"):
        setup_robot(ROBOT, catalog=CATALOG, force=True)
    assert snapshot(workspace) == before


def test_failed_staging_preserves_workspace(workspace, monkeypatch):
    setup_robot(ROBOT, catalog=CATALOG)
    before = snapshot(workspace)
    copy = shutil.copy2

    def fail(source, destination):
        if source.name == "driver.py":
            raise OSError("injected copy failure")
        return copy(source, destination)

    monkeypatch.setattr(shutil, "copy2", fail)
    with pytest.raises(OSError, match="injected"):
        setup_robot(ROBOT, catalog=CATALOG, force=True)
    assert snapshot(workspace) == before
    assert {p.name for p in workspace.iterdir()} == {".kinex"}


@pytest.mark.parametrize("existing", [False, True])
def test_failed_publication_restores_workspace(workspace, monkeypatch, existing):
    if existing:
        setup_robot(ROBOT, catalog=CATALOG)
    before = snapshot(workspace)
    rename = Path.rename

    def fail(source, destination):
        if source.name == ("devices" if existing else "prepared") and (
            source.parent.name == "prepared" or source.name == "prepared"
        ):
            raise OSError("injected publication failure")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="injected"):
        setup_robot("libero/libero-panda-tabletop", catalog=CATALOG, force=True)
    assert snapshot(workspace) == before
    assert {p.name for p in workspace.iterdir()} == ({".kinex"} if existing else set())


def test_failed_restoration_retains_recoverable_backup(workspace, monkeypatch):
    setup_robot(ROBOT, catalog=CATALOG)
    before = snapshot(workspace / ".kinex")
    rename = Path.rename

    def fail(source, destination):
        if source.parent.name == "backup" or (
            source.parent.name == "prepared" and source.name == "devices"
        ):
            raise OSError("injected filesystem failure")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="recover files from") as error:
        setup_robot("libero/libero-panda-tabletop", catalog=CATALOG, force=True)
    staging, = workspace.glob(".roboenv-setup-*")
    assert str(staging / "backup") in str(error.value)
    assert snapshot(staging / "backup") == before


def test_cli_listing_export_and_errors(workspace, catalog, capsys):
    assert main(["setup", "--list", "--catalog", str(catalog)]) == 0
    assert capsys.readouterr().out == ROBOT + "\n"
    assert main(["setup", "--list", "--catalog", str(catalog), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"robots": [ROBOT]}
    assert not list(workspace.iterdir())
    assert main(["setup", ROBOT, "--project", str(ROOT), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["destination"] == str(workspace / ".kinex")
    assert "arm" in result["devices"]
    assert main(["setup", ROBOT, "--catalog", str(catalog)]) == 1
    assert "--force" in json.loads(capsys.readouterr().out)["error"]
    assert main(["setup", ROBOT, "--catalog", str(catalog), "--force"]) == 0
    assert f"Robot: {ROBOT}" in capsys.readouterr().out
    assert main(["setup", "--list", "--force", "--catalog", str(catalog)]) == 1
    assert "--force requires a robot ID" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [[], [ROBOT, "--list"]])
def test_cli_requires_robot_or_list(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["setup", *arguments])
    assert exc.value.code == 2
    assert "--list" in capsys.readouterr().err


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["setup", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert all(flag in help_text for flag in ("--list", "--force", "--catalog", "--json"))
