"""Agent-neutral documentation preserves the public device contract without drivers."""

import json
from pathlib import Path

import pytest
import yaml
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.documentation import export_documentation, render_documentation
from roboenv.devices.velocity import command_schema

ROOT = Path(__file__).resolve().parents[1]


def test_robocasa_dimensions_connections_and_schemas(tmp_path):
    root = ROOT / "catalog/robots/robocasa/robocasa-panda-omron"
    robot = yaml.safe_load((root / "workspace.yaml").read_text())
    profiles = [spec.resolved_profile(root)
                for spec in BindingPlan.load(root, backend="robocasa").devices]
    base = next(p for p in profiles if p["name"] == "base")
    base["actions"] = {
        "write": {"supported": True,
                  "input_schema": command_schema(base["intrinsics"]["control"])},
        "ik": {"supported": False, "reason": "unsupported"},
    }
    base["binding"] = {"entity": "private-selector"}
    base["implementation"] = "private-adapter"
    connection = {"endpoint": "http://sim:8123", "session": "session", "generation": 0,
                  "token": "owner-secret"}
    export_documentation(robot, profiles, tmp_path, connection)
    text = (tmp_path / "harness.md").read_text()
    for fact in ("Franka Panda", "roboenv.connect", "roboenv device", "ROBOENV_ENDPOINT",
                 "robot0_joint7", "radians", "metres", "native simulator", "required",
                 "`actions: [value, ...]`", "`-1` opens", "`1` closes", "velocity.vx",
                 "velocity.wz", "normalized", "not m/s or rad/s", "[v, u]", "0.001", "pgm16_be",
                 "+X right", "quaternion [x, y, z, w]", "## camera_left (rgbd)"):
        assert fact in text, fact
    assert "### ik" not in text
    assert "devices/" not in text and "YAML" not in text
    assert "private-selector" not in text and "private-adapter" not in text
    assert "owner-secret" not in text
    assert not list(tmp_path.rglob("*.py"))
    for name in ("AGENTS.md", "CLAUDE.md", "ENVIRONMENT.md", "TASK.md"):
        assert not (tmp_path / name).exists()
    public = yaml.safe_load((tmp_path / "devices/base.yaml").read_text())
    assert public["actions"] == base["actions"]
    assert not {"access", "binding", "implementation"} & public.keys()
    assert "owner-secret" not in (tmp_path / "connection.json").read_text()
    assert "private-selector" not in (tmp_path / "devices.json").read_text()
    # Export must not mutate the supplied facts.
    assert base["binding"] == {"entity": "private-selector"}


def test_markdown_retains_geometry_calibration_and_execution_without_sidecar_files():
    device = {"name": "arm", "type": "joint", "intrinsics": {
        "model": {"chain": [{"axis": [0.25, -0.5, 0.75], "link_name": "unique_link"}]},
        "kinematics": {"ik": {"damping": 0.012345}},
        "calibrated_at": "2026-09-17", "optional": None,
    }, "extrinsics": {"mount": {"parent": "unique_mount", "translation_m": [1, 2, 3]}},
        "execution": {"mode": "synchronous", "interruptible": False},
        "actions": {"write": {"supported": False, "reason": "read-only fixture"}}}
    text = render_documentation({"identity": {"model": "Example arm"}}, [device],
                                {"endpoint": "http://sim:8123"})
    for value in ("0.25, -0.5, 0.75", "unique_link", "0.012345", "2026-09-17",
                  "optional: not specified", "unique_mount", "translation m: 1, 2, 3",
                  "mode: synchronous", "interruptible: no", "read-only fixture"):
        assert value in text
    assert "devices/" not in text and "```yaml" not in text


@pytest.mark.parametrize("names", [["../escape"], ["same", "same"]])
def test_rejects_invalid_device_names_before_writing(tmp_path, names):
    target = tmp_path / "export"
    with pytest.raises(ValueError):
        export_documentation({}, [{"name": n, "type": "other"} for n in names],
                             target, {"endpoint": "http://sim:8123"})
    assert not target.exists()


def test_unbounded_joint_and_profile_only_device():
    devices = [{"name": "axis", "type": "joint", "intrinsics": {
        "model": {"joints": [{"name": "spin", "type": "continuous", "limits": None}]}}},
        {"name": "world", "type": "world", "actions": {}}]
    text = render_documentation({}, devices, {"endpoint": "http://sim:8123"})
    assert "radians" in text and "not specified" in text
    assert "## world (world)" in text
    assert "### write" not in text


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_documentation_exports_actual_live_action_contract(device_service, tmp_path):
    from roboenv import connect

    _, supervisor, _ = device_service
    with connect(supervisor.http_endpoint) as env:
        profiles = env.devices()
        export_documentation({}, profiles, tmp_path, {"endpoint": supervisor.http_endpoint})
    exported = json.loads((tmp_path / "devices.json").read_text())
    assert exported[0]["actions"] == profiles[0]["actions"]
    text = (tmp_path / "harness.md").read_text()
    for device in profiles:
        for name, action in device["actions"].items():
            if action["supported"]:
                assert f"### {name}\n" in text
    assert "### read" in text


@pytest.mark.parametrize("fingers", [False, True])
def test_hand_description_and_optional_fk_export(tmp_path, fingers):
    from roboenv.devices.export import export_devices

    geometry = {"description": "Three opposed fingers."}
    if fingers:
        geometry["fingers"] = {"index": {
            "chain": [{"joint": "knuckle", "origin": [0, 0, 0, 0, 0, 0, 1],
                       "axis": [0, 0, 1], "anchor": [0, 0, 0], "reference": 0}],
            "tip_frame": "index_tip", "tip_offset": [0, 0, .05, 0, 0, 0, 1]}}
    device = {"name": "hand", "type": "dexterous_hand", "intrinsics": {"geometry": geometry},
              "extrinsics": {}, "actions": {"read": {"supported": True}}}
    text = render_documentation({}, [device], {"endpoint": "http://sim:8123"})
    assert "Three opposed fingers." in text
    export_devices([device], tmp_path, {"endpoint": "http://sim:8123"})
    exported = yaml.safe_load((tmp_path / "hand/profile.yaml").read_text())
    assert ("fk" in exported["access"]["driver"]["actions"]) is fingers
    assert exported["intrinsics"]["geometry"] == geometry
