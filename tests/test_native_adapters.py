"""Native adapters expose the existing control clock and frame budget."""

from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter
from roboenv_runtime.adapters.robotwin.adapter import RoboTwinAdapter


@pytest.mark.parametrize("adapter_type", [BehaviorAdapter, RoboDojoAdapter, RoboTwinAdapter])
def test_native_adapters_share_clock_without_creating_another_owner(adapter_type):
    advanced = []
    control = DeviceControl(
        {"arm": [(-1, 1)]},
        {"arm": [0]},
        lambda _: None,
        lambda: advanced.append(True),
        exhausted=lambda: False,
        interval_s=1 / 30,
    )
    adapter = adapter_type.__new__(adapter_type)
    adapter._env = SimpleNamespace(control=control, control_interval=1 / 30, frame_limit=20)
    assert adapter.control_interval_s == 1 / 30
    control.write("arm", [0.2])
    assert len(advanced) == 0
    adapter.advance_control()
    assert adapter.budget().used == 1
    assert adapter.budget().max == 20
    if adapter_type in (BehaviorAdapter, RoboDojoAdapter):
        assert adapter.owner_thread_required
