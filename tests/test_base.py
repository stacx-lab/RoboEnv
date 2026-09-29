"""Named velocity/posture modes and strict per-mode capabilities."""

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from roboenv.devices.velocity import command, command_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["robocasa/robocasa-panda-omron", "simple/simple-g1"])
def control(request):
    path = ROOT / "catalog/robots" / request.param / "devices/base/profile.yaml"
    return yaml.safe_load(path.read_text())["intrinsics"]["control"]


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
        {"velocity": {"vx": True, "vy": 0, "wz": 0}},
        {"posture": {"height": 0.7, "pitch": True}},
        {"posture": {"height": 0.7, "typo": 0}},
    ],
)
def test_schema_and_validator_reject_invalid_commands(control, patch):
    args = {
        "mode": next(iter(control["modes"])),
        "velocity": {"vx": 0, "vy": 0, "wz": 0},
        **(
            {
                "posture": {
                    c["axis"]: c["default"]
                    for c in control["modes"][next(iter(control["modes"]))]["posture"]["components"]
                }
            }
            if control["modes"][next(iter(control["modes"]))]["posture"]
            else {}
        ),
    }
    validator = Draft202012Validator(command_schema(control))
    validator.validate(args)
    assert command(control, args) == args
    args.update(patch)
    assert not validator.is_valid(args)
    with pytest.raises(ValueError):
        command(control, args)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_declared_axes_require_finite_numbers(control, value):
    with pytest.raises(ValueError):
        command(
            control,
            {
                "mode": next(iter(control["modes"])),
                "velocity": {"vx": value, "vy": 0, "wz": 0},
                **(
                    {
                        "posture": {
                            c["axis"]: c["default"]
                            for c in control["modes"][next(iter(control["modes"]))]["posture"][
                                "components"
                            ]
                        }
                    }
                    if control["modes"][next(iter(control["modes"]))]["posture"]
                    else {}
                ),
            },
        )


def test_standing_mode_has_distinct_capabilities():
    posture = {
        "reference": "ground",
        "components": [
            {"axis": "height", "units": "m", "bounds": {"min": 0.65, "max": 0.8}, "default": 0.75},
            {"axis": "roll", "units": "rad", "bounds": {"min": -0.2, "max": 0.2}, "default": 0},
        ],
    }
    control = {"mode": "motion", "modes": {"stand": {"velocity": None, "posture": posture}}}
    validator = Draft202012Validator(command_schema(control))
    args = {"mode": "stand", "posture": {"height": 0.7, "roll": 0.1}}
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
