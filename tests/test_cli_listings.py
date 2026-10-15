"""Catalog listings are readable without booting a simulator or exporting devices."""

import json
import shutil
from pathlib import Path

import pytest
from roboenv.cli.main import main

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "robotwin/handover-block-aloha-agilex": "robotwin/robotwin-aloha-agilex",
    "robocasa/load-condiments-in-fridge": "robocasa/robocasa-panda-omron",
}


@pytest.fixture
def catalog(tmp_path):
    catalog = tmp_path / "catalog"
    for task, robot in PAIRS.items():
        task_dir = catalog / "tasks" / task
        task_dir.mkdir(parents=True)
        shutil.copyfile(ROOT / "catalog/tasks" / task / "task.yaml", task_dir / "task.yaml")
        robot_dir = catalog / "robots" / robot
        robot_dir.mkdir(parents=True)
        shutil.copyfile(
            ROOT / "catalog/robots" / robot / "workspace.yaml", robot_dir / "workspace.yaml"
        )
    return catalog


@pytest.mark.parametrize("kind", ["tasks", "robots"])
@pytest.mark.parametrize("simulator", [None, "robotwin", "robot", "unknown"])
def test_listing_lines_and_json(kind, simulator, catalog, capsys):
    source = ["--source", str(catalog / "tasks")] if kind == "tasks" else [
        "--catalog", str(catalog)
    ]
    args = [kind, "list", *source]
    expected = sorted(PAIRS if kind == "tasks" else PAIRS.values())
    if simulator:
        args += ["--simulator", simulator]
        expected = [value for value in expected if value.startswith(simulator + "/")]
    assert main(args) == 0
    assert capsys.readouterr().out == "".join(value + "\n" for value in expected)
    assert main([*args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ids" if kind == "tasks" else "robots"] == expected
    if kind == "tasks":
        assert result["tasks"] == len(expected)
        assert sum(result["backends"].values()) == len(expected)
        assert result["execution_modes"] == dict.fromkeys(expected, "realtime")


def test_robot_listing_matches_setup_and_does_not_export(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["setup", "--list", "--project", str(ROOT)]) == 0
    expected = capsys.readouterr().out
    assert main(["robots", "list", "--project", str(ROOT)]) == 0
    assert capsys.readouterr().out == expected
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind,option", [("tasks", "--source"), ("robots", "--catalog")])
def test_missing_catalog_reports_error(kind, option, tmp_path, capsys):
    assert main([kind, "list", option, str(tmp_path / "missing")]) == 1
    assert "ValueError" in json.loads(capsys.readouterr().out)["error"]


def test_robotwin_lists_only_aloha_and_cannot_export_piper(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["robots", "list", "--simulator", "robotwin", "--project", str(ROOT)]) == 0
    assert capsys.readouterr().out == "robotwin/robotwin-aloha-agilex\n"
    assert main(["setup", "robotwin/robotwin-dual-piper", "--project", str(ROOT)]) == 1
    assert "Unknown robot" in json.loads(capsys.readouterr().out)["error"]
    assert not list(tmp_path.iterdir())
