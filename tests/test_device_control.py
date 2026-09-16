"""Cross-backend actuator contract; these tests do not certify native physics."""

import copy

import pytest
from roboenv_runtime.adapters.device_control import DeviceControl


def control(*, realtime=False, horizon=3):
    states = []
    targets = {"left": [0.0], "right": [0.0]}
    holder = {}
    owner = DeviceControl(
        {name: [(-1, 1)] for name in targets},
        targets,
        lambda goals: holder.update(goals),
        lambda: states.append(copy.deepcopy(holder)),
        realtime=realtime,
        exhausted=lambda: len(states) >= horizon,
    )
    return owner, states, holder


def test_each_write_advances_one_interval_and_preserves_peer_targets():
    owner, states, _ = control()
    assert owner.write("left", [0.5]) == {"accepted": True}
    assert states == [{"left": [0.5], "right": [0]}]
    owner.write("right", [-0.5])
    owner.write("left", [0.2])
    assert states == [
        {"left": [0.5], "right": [0]},
        {"left": [0.5], "right": [-0.5]},
        {"left": [0.2], "right": [-0.5]},
    ]
    assert len(owner.commands) == 3


@pytest.mark.parametrize("target", [[float("nan")], [float("inf")], [True], [2], [], [0, 0]])
def test_invalid_write_has_zero_effects(target):
    owner, states, holder = control()
    owner.write("left", [0.5])
    before = copy.deepcopy((owner.targets, owner.commands, states, holder))
    with pytest.raises(ValueError):
        owner.write("right", target)
    assert (owner.targets, owner.commands, states, holder) == before


def test_stop_does_not_step_and_keeps_measured_overshoot():
    owner, states, holder = control()
    owner.stop("left", [1.0001])
    assert states == [] and holder == {"left": [1.0001], "right": [0]}
    owner.write("right", [0.4])
    assert states == [{"left": [1.0001], "right": [0.4]}]


def test_realtime_only_shared_ticks_advance_and_budget_freezes_commands():
    owner, states, _ = control(realtime=True, horizon=1)
    owner.write("left", [0.3])
    assert not states
    owner.tick()
    assert states == [{"left": [0.3], "right": [0]}]
    before = copy.deepcopy(owner.commands)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        owner.write("right", [0.5])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        owner.tick()
    assert owner.commands == before and len(states) == 1
