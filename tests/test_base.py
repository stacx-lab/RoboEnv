"""Named mobile-base motion modes, including device-side posture filtering."""

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from roboenv.devices.base import command, command_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["robocasa/robocasa-panda-omron", "isaaclab/isaaclab-g1"])
def control(request):
    path = ROOT / "catalog/robots" / request.param / "devices/base/profile.yaml"
    return yaml.safe_load(path.read_text())["intrinsics"]["control"]


def test_motion_schema_filters_untracked_axes(control):
    args = {
        "mode": next(iter(control["modes"])),
        "velocity": {"vx": 0.1, "vy": 0, "wz": 0},
        "posture": {"height": 0.7, "roll": 0.1, "pitch": -0.2, "yaw": 0.3},
    }
    schema = command_schema(control)
    assert len(schema["oneOf"]) == len(control["modes"])
    Draft202012Validator(schema).validate(args)
    expected = {"mode": args["mode"], "velocity": args["velocity"]}
    if control["modes"][args["mode"]]["posture"]:
        expected["posture"] = {"height": 0.7}
    assert command(control, args) == expected
    assert args["posture"]["pitch"] == -0.2  # validation does not mutate the caller


def test_posture_is_required_only_for_tracked_axes(control):
    args = {"mode": next(iter(control["modes"])), "velocity": {"vx": 0, "vy": 0, "wz": 0}}
    validator = Draft202012Validator(command_schema(control))
    if control["modes"][args["mode"]]["posture"]:
        assert not validator.is_valid(args)
        with pytest.raises(ValueError):
            command(control, args)
        return
    validator.validate(args)
    assert command(control, args) == args


@pytest.mark.parametrize(
    "patch",
    [
        {"mode": "unsupported"},
        {"target": [0, 0, 0]},
        {"velocity": {"vx": 0, "vy": 0}},
        {"velocity": {"vx": 2, "vy": 0, "wz": 0}},
        {"velocity": {"vx": True, "vy": 0, "wz": 0}},
        {"posture": {"height": 0.7, "pitch": True}},
        {"posture": {"height": 0.7, "typo": 0}},
    ],
)
def test_schema_and_validator_reject_invalid_commands(control, patch):
    args = {
        "mode": next(iter(control["modes"])),
        "velocity": {"vx": 0, "vy": 0, "wz": 0},
        "posture": {"height": 0.7},
        **patch,
    }
    assert not Draft202012Validator(command_schema(control)).is_valid(args)
    with pytest.raises(ValueError):
        command(control, args)


def test_ignored_posture_still_requires_finite_numbers(control):
    with pytest.raises(ValueError):
        command(
            control,
            {
                "mode": next(iter(control["modes"])),
                "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": 0.7, "pitch": float("nan")},
            },
        )


def test_standing_mode_has_distinct_capabilities(control):
    posture = {
        "reference": "ground",
        "components": [
            {"axis": "height", "units": "m", "bounds": {"min": 0.65, "max": 0.8}, "default": 0.75},
            {"axis": "roll", "units": "rad", "bounds": {"min": -0.2, "max": 0.2}, "default": 0},
        ],
    }
    control["modes"]["stand"] = {"velocity": None, "posture": posture}
    validator = Draft202012Validator(command_schema(control))
    args = {"mode": "stand", "posture": {"height": 0.7, "roll": 0.1, "pitch": 0.3}}
    validator.validate(args)
    assert command(control, args) == {"mode": "stand", "posture": {"height": 0.7, "roll": 0.1}}
    for invalid in [
        {"posture": args["posture"]},
        {**args, "velocity": {"vx": 0, "vy": 0, "wz": 0}},
        {**args, "posture": {"height": 0.7}},
        {**args, "posture": {"height": 0.7, "roll": 0.3}},
    ]:
        assert not validator.is_valid(invalid)
        with pytest.raises(ValueError):
            command(control, invalid)
