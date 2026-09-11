"""Cross-backend actuator contract; these tests do not certify native physics."""

import copy

import pytest
from roboenv_runtime.adapters.device_control import DeviceControl


def control(*, realtime=False, horizon=3):
    states = []
    targets = {"left": [0.0], "right": [0.0]}
    holder = {}
    owner = DeviceControl(
        {name: [(-1, 1)] for name in targets}, targets,
        lambda goals: holder.update(goals), lambda: states.append(copy.deepcopy(holder)),
        realtime=realtime, exhausted=lambda: len(states) >= horizon,
    )
    return owner, states, holder


def test_coordinated_update_is_one_interval_and_preserves_peer_targets():
    owner, states, _ = control()
    step = {"id": 0, "devices": ["left", "right"]}
    assert owner.write("left", [0.5], step)["pending"]
    assert states == []
    assert owner.targets["left"] == [0]
    assert not owner.write("right", [-0.5], step)["pending"]
    assert states == [{"left": [0.5], "right": [-0.5]}]
    owner.write("left", [0.2])
    assert states[-1] == {"left": [0.2], "right": [-0.5]}
    assert len(owner.commands) == 2


@pytest.mark.parametrize("target", [[float("nan")], [float("inf")], [True], [2], [], [0, 0]])
def test_invalid_partner_has_zero_effects(target):
    owner, states, holder = control()
    step = {"id": 0, "devices": ["left", "right"]}
    owner.write("left", [0.5], step)
    before = copy.deepcopy(owner.pending)
    with pytest.raises(ValueError):
        owner.write("right", target, step)
    assert owner.pending == before
    assert states == [] and holder == {}
    assert owner.commands == []


def test_duplicates_mixed_groups_and_reused_ids_are_rejected():
    owner, states, _ = control()
    step = {"id": 2, "devices": ["left", "right"]}
    owner.write("left", [0.5], step)
    for device, candidate in [("left", step), ("right", None),
                              ("right", {"id": 3, "devices": step["devices"]})]:
        with pytest.raises(ValueError):
            owner.write(device, [0], candidate)
    owner.write("right", [0], step)
    with pytest.raises(ValueError, match="increase"):
        owner.write("left", [0], step)
    assert len(states) == 1


def test_stop_cancels_pending_without_steps_and_keeps_measured_overshoot():
    owner, states, holder = control()
    owner.write("left", [0.5], {"id": 1, "devices": ["left", "right"]})
    owner.stop("left", [1.0001])
    assert states == [] and holder == {"left": [1.0001], "right": [0]}
    assert owner.pending is None
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
