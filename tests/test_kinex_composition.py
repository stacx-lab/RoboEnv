"""Test-only composition fixture contract, without an adjacent Kinex checkout."""

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

FIXTURE = Path(__file__).parent / "fixtures/device-composition/tool.py"


@pytest.mark.parametrize("waist", [False, True])
@pytest.mark.parametrize("realtime", [False, True])
@pytest.mark.parametrize("failure", [None, "read", "write", "fault", "clock"])
def test_composition(monkeypatch, waist, realtime, failure):
    names = ["base", *(["waist"] if waist else []),
             "left_arm", "right_arm", "left_hand", "right_hand"]
    calls = []
    stamps = iter([0, 0, 0, -0.02 if failure == "clock" else 0.02])

    def profile(name):
        if name == "base":
            return {"intrinsics": {"control": {
                "execution": {"mode": "realtime" if realtime else "synchronous"},
                "modes": {"locomotion": {"posture": {
                    "components": [{"axis": "height", "default": 0.74}],
                }}},
            }}}
        return {"extrinsics": {"mount": {"parent": "waist" if waist else "base"}}}

    def dispatch(commands):
        calls.append(commands)
        assert [c["device"] for c in commands] == names
        action = commands[0]["action"]
        assert {c["action"] for c in commands} == {action}
        stamp = next(stamps) if action == "read" else 0
        if action == "write":
            assert commands[0]["args"] == {
                "mode": "locomotion", "velocity": {"vx": 0, "vy": 0, "wz": 0},
                "posture": {"height": 0.74},
            }
            assert all(c["args"] == {"target": [0, 0, 0]} for c in commands[1:])
        return [{"device": name, "ok": not (action == failure and len(calls) > 1),
                 "data": {"positions": [0, 0, 0], "sample_time_s": stamp,
                          "faults": ["fault"] if failure == "fault" else []}}
                for name in names]

    def loop(tick, *, max_steps, timeout_s, on_stop, rate_hz):
        assert rate_hz == (50 if realtime else None)
        try:
            for index in range(max_steps):
                assert tick(index) is False
            return {"steps": max_steps}
        finally:
            on_stop()

    kinex = Mock()
    kinex.args.return_value = {"steps": 3, "rate_hz": 50}
    kinex.profile.side_effect = profile
    kinex.devices.side_effect = dispatch
    kinex.loop.side_effect = loop
    joints = Mock()
    joints.JointDevice.return_value.project.side_effect = list
    for name, module in {"kinex": kinex, "kinex.joints": joints,
                         "kinex.geometry": Mock(), "kinex.motion": Mock()}.items():
        monkeypatch.setitem(sys.modules, name, module)
    main = runpy.run_path(str(FIXTURE))["main"]
    if failure:
        message = {"read": "read failed", "write": "write failed",
                   "fault": "Device fault", "clock": "clock moved backwards"}[failure]
        with pytest.raises(ValueError, match=message):
            main()
        kinex.finish.assert_not_called()
        assert calls[-1][0]["action"] == "stop"
        return
    main()
    expected = ["read", "read", "write", "read"]
    if not realtime:
        expected.append("write")
    assert [c[0]["action"] for c in calls] == [*expected, "read", "write", "stop"]
    kinex.finish.assert_called_once_with({"steps": 3, "errors": {}, "grasp_confirmed": False})
