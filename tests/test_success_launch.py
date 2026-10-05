"""Launch-time success policy overrides persist to workers without changing catalogs."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from roboenv.cli.workflows import add_commands, execute
from roboenv.environment import launch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-toy"


@pytest.mark.parametrize("flags,expected", [
    ([], None), (["--stop-on-success"], True), (["--no-stop-on-success"], False),
])
def test_cli_forwards_success_override(monkeypatch, flags, expected):
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["launch", "simple/xmove-pick", "--detach", *flags])
    received = {}

    def start(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(record={}, stop_logs=lambda: None)

    monkeypatch.setattr("roboenv.environment.launch", start)
    assert execute(args) == 0
    assert received["stop_on_success"] is expected
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "simple/xmove-pick",
                           "--stop-on-success", "--no-stop-on-success"])


@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_success_override_is_retained_in_worker_configuration(
    tmp_path, monkeypatch, setting, override
):
    task = yaml.safe_load((FIXTURE / "task.yaml").read_text())
    if setting is not None:
        task["stop_on_success"] = setting
    else:
        task.pop("stop_on_success")
    source = tmp_path / "task.yaml"
    source.write_text(yaml.safe_dump(task))
    before = source.read_bytes()
    seen = []

    def capture(command, **kwargs):
        seen.append(kwargs["env"])
        raise OSError("stop before starting a simulator")

    monkeypatch.setattr("roboenv.environment.subprocess.Popen", capture)
    with pytest.raises(OSError, match="stop before starting"):
        launch(source, project=ROOT, catalog=FIXTURE / "catalog",
               run_dir=tmp_path / "runs", stop_on_success=override)
    env = seen[0]
    try:
        expected = {"stop_on_success": False, **task}
        if override is not None:
            expected["stop_on_success"] = override
        manifest = json.loads(Path(env["ROBOENV_SERVER_MANIFEST"]).read_text())
        worker = yaml.safe_load(
            (Path(env["ROBOENV_TASKS"]) / "task.yaml").read_text()
        )
        assert manifest["task"] == worker == expected
        assert source.read_bytes() == before
    finally:
        shutil.rmtree(env["ROBOENV_SOCKET_ROOT"])


@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_invalid_policy_fails_before_discovery_or_launch(monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid policy reached task discovery")

    monkeypatch.setattr("roboenv.environment.resolve_task", forbidden)
    with pytest.raises(ValueError, match="stop_on_success must be a boolean"):
        launch("simple/xmove-pick", stop_on_success=value)
