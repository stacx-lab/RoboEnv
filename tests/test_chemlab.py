"""Instrument practice contracts and opt-in checks against the real pinned release."""

import copy
import json
import os
from pathlib import Path

import pytest
import yaml
from helpers import load_tool
from roboenv.devices.model import DeviceError
from roboenv.provisioning.chemlab_assets import resource
from roboenv.provisioning.chemlab_catalog import generate, public_capabilities
from roboenv.tasks.catalog import native_identity
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.tasks.chemlab.environment import ChemLabEnvironment, matches
from roboenv_runtime.bindings.backends.isaaclab.chemlab import action_name

ROOT = Path(__file__).resolve().parents[1]


def test_native_capability_names_use_valid_portable_driver_verbs():
    assert action_name("thermal.set_temperature") == "thermal_set_temperature"
    assert action_name("operation.poll") == "operation_poll"
    with pytest.raises(ValueError, match="identifier"):
        action_name("not/a/capability")


SOLUTIONS = {
    "hood-light-sash": [
        ("light.set", {"on": True}),
        ("sash.set", {"height_m": 0.457}),
        ("fan.set", {"on": True}),
    ],
    "water-activity-temperature": [
        ("power.set", {"enabled": True}),
        ("thermal.set_temperature", {"temperature_k": 303.15}),
    ],
    "hotplate-stirring": [
        ("power.set", {"on": True}),
        ("stirring.set_speed", {"speed_rad_s": 62.83185307179586}),
        ("stirring.set", {"on": True}),
    ],
    "balance-power": [("power.set", {"enabled": True})],
}


@pytest.fixture
def assets():
    value = os.environ.get("ROBOENV_CHEMLAB_TEST_ASSETS")
    if not value:
        pytest.skip("Set ROBOENV_CHEMLAB_TEST_ASSETS to the verified pinned Git/LFS depot")
    return Path(value)


def configuration(name):
    task = yaml.safe_load((ROOT / f"catalog/tasks/isaaclab/chemlab-{name}/task.yaml").read_text())
    return {key: value for key, value in task["backend_kwargs"].items() if key != "task_name"}


def test_distinct_practice_goals_have_distinct_discovery_identities():
    tasks = [resolve_task(f"isaaclab/chemlab-{name}", project=ROOT).task for name in SOLUTIONS]
    assert len({native_identity(task) for task in tasks}) == 4
    assert all(task.budget_mode == "practice" for task in tasks)


def test_goal_rejects_missing_unavailable_nonfinite_and_boolean_measurements():
    goal = {"state": "temperature", "min": 300, "max": 301}
    for value in (None, True, "300", float("nan"), float("inf")):
        assert not matches({"temperature": {"value": value}}, goal)
    assert not matches({}, goal)
    assert not matches({"temperature": {"value": 300.5, "quality": "unavailable"}}, goal)
    assert matches({"temperature": {"value": 300.5, "quality": "valid"}}, goal)
    assert not matches({"switch": {"value": 1}}, {"state": "switch", "equals": True})


def test_asset_paths_do_not_escape_depot(tmp_path):
    root = tmp_path / "depot"
    root.mkdir()
    secret = tmp_path / "private"
    secret.write_text("outside")
    (root / "link").symlink_to(secret)
    for name in ("../private", str(secret), "link", "missing"):
        with pytest.raises(ValueError, match="Missing or unsafe"):
            resource(root, name)


def test_agent_scene_kind_is_not_an_owner_permission():
    public = public_capabilities(
        {
            "capabilities": {
                "scene.set_door": {"visibility": "agent", "kind": "simulation"},
                "plant.configure": {"visibility": "agent", "kind": "configuration"},
                "panel.frame": {"visibility": "evaluator", "kind": "query"},
                "fault.acknowledge": {"visibility": "agent", "kind": "device_command"},
                "fault.inject": {"visibility": "evaluator", "kind": "simulation"},
            }
        }
    )
    assert set(public) == {"scene.set_door", "fault.acknowledge"}


def test_release_catalog_resolves_every_instrument_without_scored_goals(assets, tmp_path):
    result = generate(tmp_path, assets, project=ROOT)
    assert result["instruments"] == 203 and result["scored_tasks"] == 0
    paths = sorted((tmp_path / "tasks").rglob("task.yaml"))
    assert len(paths) == 203
    for path in paths:
        resolved = resolve_task(path, catalog=tmp_path)
        assert resolved.task.execution_mode == "realtime"
        assert resolved.task.backend_kwargs["task_spec"]["goals"] == []
        profile = resolved.bindings.devices[0].resolved_profile(resolved.robot)
        assert profile["intrinsics"]["instrument"]["asset_id"].startswith("instrument.")
        assert "plant_configure" not in profile["access"]["driver"]["actions"]
    # Regeneration uses the existing content-conflict policy.
    generate(tmp_path, assets, project=ROOT)
    paths[0].write_text("edited")
    with pytest.raises(ValueError, match="edited/unmanaged"):
        generate(tmp_path, assets, project=ROOT)


def test_native_scene_commands_change_geometry_and_invalid_commands_do_not(assets):
    import numpy as np

    config = configuration("balance-power")
    config["task_spec"]["goals"] = []
    environment = ChemLabEnvironment(**config, asset_directory=assets, render=False)
    try:
        environment.reset(0)
        projection = environment.projections["instrument"]
        initial = projection.model.get_transform("door_left").copy()
        response = environment.invoke(
            "instrument",
            "scene.set_door",
            {
                "side": "left",
                "opening_m": 0.05,
            },
        )
        assert response["status"] == "succeeded"
        moved = projection.model.get_transform("door_left").copy()
        assert np.isclose(np.linalg.norm(moved[:3, 3] - initial[:3, 3]), 0.05)
        response = environment.invoke(
            "instrument",
            "scene.set_door",
            {
                "side": "left",
                "opening_m": 100,
            },
        )
        assert response["status"] == "rejected"
        assert np.array_equal(projection.model.get_transform("door_left"), moved)
        environment.reset(0)
        assert np.allclose(
            environment.projections["instrument"].model.get_transform("door_left"), initial
        )
    finally:
        environment.close()


def test_velocity_projection_uses_world_time_and_instances_are_isolated(assets):
    import numpy as np
    from roboenv.provisioning.chemlab_assets import catalog
    from roboenv_runtime.adapters.isaaclab.tasks.chemlab.projection import InstrumentProjection

    row = catalog(assets)["instrument.wtw_oxitopi"]
    first = InstrumentProjection(assets, row, "first")
    second = InstrumentProjection(assets, row, "second")
    values = {"platform.stir_velocity_rad_s": 2.0}
    first.update(values, dt=0.25)
    assert first.config["magnet_rotor_1_joint"] == pytest.approx(0.5)
    transform = first.model.get_transform("magnet_rotor_1").copy()
    for _ in range(3):
        first.update(values)
    assert np.array_equal(transform, first.model.get_transform("magnet_rotor_1"))
    second.update(values)
    assert second.config["magnet_rotor_1_joint"] == 0
    first.update(values, dt=0.25)
    assert first.config["magnet_rotor_1_joint"] == pytest.approx(1.0)


def test_released_accessories_and_individual_centrifuge_samples_are_visible(assets):
    from roboenv.provisioning.chemlab_assets import catalog
    from roboenv_runtime.adapters.isaaclab.tasks.chemlab.projection import InstrumentProjection

    rows = catalog(assets)
    projection = InstrumentProjection(assets, rows["instrument.jasco_ftir4700"], "instrument")
    projection.update({"accessory.mode": "atr", "sample.loaded": False})
    nodes = list(projection.model.scene.graph.nodes_geometry)
    assert any("accessory_atr" in n and projection.node_visible(n) for n in nodes)
    assert not any("accessory_transmission" in n and projection.node_visible(n) for n in nodes)
    projection = InstrumentProjection(assets, rows["instrument.eppendorf_5910ri"], "instrument")
    projection.update({"rotor.configuration": "fixed_angle", "sample.occupied_slots": ["slot_2"]})
    shown = {source for node, source in projection.groups.items() if projection.node_visible(node)}
    assert "rotor_fixed_plate" in shown and "rotor_large_hub" not in shown
    assert "sample_carrier_2_body" in shown and "sample_carrier_1_body" not in shown
    projection.update({"lid.locked": False, "alarm.imbalance": False})
    shown = {source for node, source in projection.groups.items() if projection.node_visible(node)}
    assert "eppendorf_5910ri__touchscreen" in shown
    assert "lock_indicator" not in shown and "imbalance_indicator" not in shown


def test_readout_projection_does_not_claim_native_scene_coverage():
    from types import SimpleNamespace

    from roboenv_runtime.adapters.isaaclab.tasks.chemlab.geometry import binding_report

    report = binding_report(
        SimpleNamespace(
            contract={
                "model_bindings": {
                    "readout": {"state_id": "temperature", "channel": "material"},
                    "input": {"state_id": None, "direction": "scene_to_process"},
                    "unbound": {"state_id": None},
                    "missing": {"state_id": "extension", "channel": "pose"},
                }
            }
        ),
        readout_keys={"readout", "unbound"},
    )
    assert report == {
        "synchronized": [],
        "readout_only": ["readout"],
        "input_only": ["input"],
        "unbound": ["unbound"],
        "unsupported": ["missing"],
    }


def test_released_rack_formats_and_sampler_frames_follow_runtime_state(assets):
    import numpy as np
    from roboenv.provisioning.chemlab_assets import catalog
    from roboenv_runtime.adapters.isaaclab.tasks.chemlab.projection import InstrumentProjection

    rows = catalog(assets)
    projection = InstrumentProjection(assets, rows["instrument.ziath_dp5_mirage_2023"], "rack")
    for size in (24, 48, 96, 240):
        projection.update({"scene.rack_format": size, "scene.rack_lift_m": 0.04})
        shown = [
            n for n in projection.model.scene.graph.nodes_geometry if projection.node_visible(n)
        ]
        assert any(f"supplement_rack_{size}_" in n for n in shown)
        assert not any(
            f"supplement_rack_{other}_" in n
            for other in (24, 48, 96, 240)
            if other != size
            for n in shown
        )
        assert np.isclose(projection.model.get_transform("rack_carriage")[2, 3], 0.04)
    projection = InstrumentProjection(assets, rows["instrument.agilent_gc8890b"], "gc")
    for loaded in (False, True):
        projection.update({"sample.loaded": loaded})
        assert "sample_visibility" in projection.binding_keys()
        nodes = [
            n for n in projection.model.scene.graph.nodes_geometry if "supplement_sample_vial" in n
        ]
        assert nodes and all(projection.node_visible(n) == loaded for n in nodes)
    projection = InstrumentProjection(assets, rows["instrument.agilent_hs8697"], "sampler")
    node = next(
        n
        for n in projection.model.scene.graph.nodes_geometry
        if projection.model.scene.graph.transforms.parents[n] == "manipulator_arm"
    )
    projection.update({"lid.angle_rad": 0.6, "manipulator.position_slot": "home"})
    initial = projection.node_transform(node)
    assert projection.config["lid_hinge_joint"] == 0.6
    projection.update({"manipulator.position_slot": "tray_01"})
    assert np.allclose(
        projection.node_transform(node)[:3, 3] - initial[:3, 3], [-0.314, -0.395, -0.002]
    )
    assert "manipulator_pose" in projection.binding_keys()


@pytest.mark.parametrize("name", SOLUTIONS)
def test_real_task_requires_commands_dwell_and_replays(assets, name):
    environment = ChemLabEnvironment(**configuration(name), asset_directory=assets, render=False)
    environment.stop_on_success = True
    try:
        environment.reset(7)
        initial = environment.read_device("instrument")
        assert environment.read_device("instrument") == initial
        assert environment.frame == 0
        # Cover the entire old practice horizon: a powered balance used to solve
        # itself after 5.9 s, beyond the former two-second negative check.
        for _ in range(6000):
            environment.tick()
        assert not environment.completed
        for capability, arguments in SOLUTIONS[name]:
            response = environment.invoke("instrument", capability, arguments)
            assert response["status"] == "succeeded", response
        assert not environment.completed
        for _ in range(3000):
            environment.tick()
            if environment.completed:
                break
        assert environment.completed
        record = environment.record()
        environment.tick()
        assert environment.record() == record
        with pytest.raises(DeviceError, match="ended"):
            environment.invoke("instrument", *SOLUTIONS[name][0])
    finally:
        environment.close()
    assert load_tool("replay_chemlab").replay(record, asset_directory=assets)["passed"]
    changed = copy.deepcopy(record)
    changed["sources"]["chemlab-runtimes.json"] = "changed"
    with pytest.raises(ValueError, match="source identity"):
        load_tool("replay_chemlab").replay(changed, asset_directory=assets)


def test_real_instances_share_clock_without_sharing_state_or_owner_commands(assets):
    config = configuration("hood-light-sash")
    spec = config["task_spec"]
    spec["instruments"]["second"] = copy.deepcopy(spec["instruments"]["instrument"])
    config["max_steps"] = 3
    environment = ChemLabEnvironment(**config, asset_directory=assets, render=False)
    try:
        environment.reset(0)
        initial = environment.owner_state()
        with pytest.raises(DeviceError, match="not exposed"):
            environment.invoke("instrument", "plant.configure", {})
        assert environment.owner_state() == initial
        environment.invoke("instrument", "light.set", {"on": True})
        assert not environment.read_device("second")["observations"]["light.on"]["value"]
        for _ in range(5):
            environment.tick()
        assert environment.frame == 5
        assert environment.instruments["instrument"].time == environment.instruments["second"].time
        environment.reset(0)
        assert environment.owner_state() == initial
        with pytest.raises(ValueError, match="realtime"):
            environment.configure_execution_mode("stepped")
        assert environment.budget().max is None
        # Actual panel bytes cross the transport, never the host's temporary path.
        instrument = environment.instruments["instrument"]
        capture = instrument.contract["capabilities"]["panel.capture"]
        display = capture["input_schema"]["properties"]["display_id"]["enum"][0]
        response = environment.invoke("instrument", "panel.capture", {"display_id": display})
        image = response["result"]["image"]
        assert image["mime_type"] == "image/png" and image["base64"].startswith("iVBOR")
        assert "path" not in image
        assert "roboenv_owner" not in json.dumps(response)
    finally:
        environment.close()
