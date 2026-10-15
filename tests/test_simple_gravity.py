"""Startup/replay contracts; physical compensation is checked in the native probe."""

import argparse
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from roboenv.cli.workflows import add_commands, execute
from roboenv.environment import launch
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.simple.backend import SimpleEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_gravity_resolution_precedence_and_persistence(tmp_path, setting, override):
    original = resolve_task("simple/xmove-pick", project=ROOT)
    source = original.path.read_bytes()
    definition = yaml.safe_load(source)
    definition["backend_kwargs"].pop("arm_gravity_compensation")
    if setting is not None:
        definition["backend_kwargs"]["arm_gravity_compensation"] = setting
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(definition))
    resolved = resolve_task(path, project=ROOT, arm_gravity_compensation=override)
    expected = override if override is not None else setting if setting is not None else True
    assert resolved.task.backend_kwargs["arm_gravity_compensation"] is expected
    assert resolved.task.as_dict()["backend_kwargs"]["arm_gravity_compensation"] is expected
    assert original.path.read_bytes() == source


@pytest.mark.parametrize("value", [1, 0, "false", [], {}])
def test_invalid_gravity_rejected_before_native_start(tmp_path, value):
    with pytest.raises(ValueError, match="boolean"):
        launch("simple/xmove-pick", project=ROOT, arm_gravity_compensation=value)
    with pytest.raises(ValueError, match="boolean"):
        SimpleEnv(arm_gravity_compensation=value)
    definition = yaml.safe_load(resolve_task("simple/xmove-pick").path.read_text())
    definition["backend_kwargs"]["arm_gravity_compensation"] = value
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="boolean"):
        resolve_task(path, project=ROOT)


def test_other_backends_reject_gravity_option(tmp_path):
    with pytest.raises(ValueError, match="only for SIMPLE"):
        launch("libero/libero-10-10", project=ROOT, arm_gravity_compensation=False)
    definition = yaml.safe_load(resolve_task("libero/libero-10-10").path.read_text())
    definition["backend_kwargs"]["arm_gravity_compensation"] = True
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="only for SIMPLE"):
        resolve_task(path, project=ROOT)


@pytest.mark.parametrize(
    "flag,expected",
    [
        ([], None),
        (["--arm-gravity-compensation"], True),
        (["--no-arm-gravity-compensation"], False),
    ],
)
def test_cli_forwards_gravity_selection(monkeypatch, flag, expected):
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    options = parser.parse_args(["launch", "simple/xmove-pick", "--detach", *flag])
    received = {}

    def start(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(record={}, stop_logs=lambda: None)

    monkeypatch.setattr("roboenv.environment.launch", start)
    assert execute(options) == 0
    assert received["arm_gravity_compensation"] is expected
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "simple/xmove-pick",
                "--arm-gravity-compensation",
                "--no-arm-gravity-compensation",
            ]
        )


@pytest.mark.parametrize(
    "schema,protocol,setting",
    [
        ("roboenv/simple-record/2", "simple-g1-metric-wbc/3", False),
        ("roboenv/simple-record/3", "simple-g1-metric-wbc/3", True),
        ("roboenv/simple-record/3", "simple-g1-metric-wbc/4", None),
        ("roboenv/simple-record/3", "simple-g1-metric-wbc/4", "false"),
    ],
)
def test_replay_rejects_ambiguous_gravity_before_boot(schema, protocol, setting):
    replay = runpy.run_path(str(ROOT / "tools/replay_simple_devices.py"))["replay"]
    metadata = {"control_protocol": protocol}
    if setting is not None:
        metadata["arm_gravity_compensation"] = setting
    with pytest.raises(ValueError):
        replay({"schema": schema, "initialization": metadata})
