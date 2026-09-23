"""Native clock selection, portable rig paths, and BEHAVIOR execution metadata."""

from types import SimpleNamespace

import pytest
from helpers import ROOT, load_tool
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.behavior.adapter import BehaviorAdapter
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices
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


def test_behavior_clock_reaches_native_and_portable_base_profile(tmp_path):
    import yaml

    original = resolve_task("behavior/turning-on-radio", project=ROOT)
    resolved = resolve_task(original.path, project=ROOT)
    world = BehaviorDevices("turning_on_radio", "house_double_floor_lower")
    assert world.env is None
    base = next(spec for spec in resolved.bindings.devices if spec.entity == "base")
    # Timing is the device's own business now: a write reports the time it spent,
    # so a profile declares no execution block at all.
    control = base.resolved_profile(resolved.robot)["intrinsics"]["control"]
    assert "execution" not in control
    # A real task YAML follows the same path used by exported jobs and workspaces.
    from dataclasses import asdict

    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(asdict(resolved.task)))
    output = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare(task, output, project=ROOT)
    copied = yaml.safe_load((output / ".kinex/devices/base/profile.yaml").read_text())
    assert "execution" not in copied["intrinsics"]["control"]
    assert original.path.read_bytes() == resolved.path.read_bytes()


@pytest.mark.parametrize(
    "backend,rigs",
    [
        ("robotwin", ("robotwin-dual-piper", "robotwin-aloha-agilex")),
        ("robodojo", ("robodojo-arx-x5", "robodojo-yam")),
    ],
)
def test_native_catalogs_and_external_bindings_share_the_namespace(backend, rigs):
    for rig in rigs:
        root = ROOT / "catalog/robots" / backend / rig
        assert {p.name for p in root.iterdir()} == {"workspace.yaml", "devices"}
        plan = BindingPlan.load(root, backend=backend)
        assert all(spec.binding == backend + ".device/1" for spec in plan.devices)
        assert not (ROOT / "catalog/robots" / rig).exists()
