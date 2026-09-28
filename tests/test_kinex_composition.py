"""Test-only composition fixture contract, without an adjacent Kinex checkout."""

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

FIXTURE = Path(__file__).parent / "fixtures/device-composition/tool.py"


@pytest.mark.parametrize("waist", [False, True])
@pytest.mark.parametrize("failure", [None, "read", "write", "fault"])
def test_composition(monkeypatch, waist, failure):
    names = [
        "base",
        *(["waist"] if waist else []),
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    calls = []

    def profile(name):
        if name == "base":
            return {
                "intrinsics": {
                    "control": {
                        "modes": {
                            "locomotion": {
                                "posture": {
                                    "components": [{"axis": "height", "default": 0.74}],
                                }
                            }
                        },
                    }
                }
            }
        return {"extrinsics": {"mount": {"parent": "waist" if waist else "base"}}}

    def dispatch(commands):
        calls.append(commands)
        assert [c["device"] for c in commands] == names
        action = commands[0]["action"]
        assert {c["action"] for c in commands} == {action}
        if action == "write":
            assert commands[0]["args"] == {
                "mode": "locomotion",
                "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": 0.74},
                "duration_s": 0.1,
            }
            assert all(
                c["args"] == {"actions": [[0, 0, 0]], "duration_s": 0.1} for c in commands[1:]
            )
        return [
            {
                "device": name,
                "ok": not (action == failure and len(calls) > 1),
                "data": {
                    "positions": [0, 0, 0],
                    "sample_time_s": 0.0,
                    "elapsed_s": 0.0,
                    "execution_id": "fixture",
                    "faults": ["fault"] if failure == "fault" else [],
                },
            }
            for name in names
        ]

    kinex = Mock()
    kinex.args.return_value = {"steps": 3}
    kinex.profile.side_effect = profile
    kinex.devices.side_effect = dispatch
    kinex.device.return_value = {"data": {"elapsed_s": 0.1}}
    joints = Mock()
    joints.JointDevice.return_value.project.side_effect = list
    for name, module in {
        "kinex": kinex,
        "kinex.joints": joints,
        "kinex.geometry": Mock(),
        "kinex.motion": Mock(),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    main = runpy.run_path(str(FIXTURE))["main"]
    if failure:
        message = {"read": "read failed", "write": "write failed", "fault": "Device fault"}[failure]
        with pytest.raises(ValueError, match=message):
            main()
        kinex.finish.assert_not_called()
        assert calls[-1][0]["action"] == "stop"
        return
    main()
    # One read and one write per round, whoever advances the clock, then one stop.
    assert [c[0]["action"] for c in calls] == [
        "read",
        *["read", "write"] * 3,
        "stop",
    ]
    assert kinex.device.call_count == 3
    kinex.device.assert_called_with("world", "wait", duration_s=0.1)
    kinex.finish.assert_called_once_with(
        {"steps": 3, "elapsed_s": pytest.approx(0.3), "errors": {}, "grasp_confirmed": False}
    )
