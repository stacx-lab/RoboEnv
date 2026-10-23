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
