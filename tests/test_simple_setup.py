"""SIMPLE asset migration and managed Isaac integration, without GPU imports."""

import ast
import hashlib
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
from roboenv.provisioning.simple_assets import digest, prepare_assets, verify_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stale", [False, True])
def test_asset_layout_migration_and_readonly_verification(tmp_path, stale, monkeypatch):
    from roboenv.provisioning import simple_assets

    depot = tmp_path / ".cache/simple"
    archive = depot / "downloads/materials.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("vMaterials_2/material.mdl", b"material")
    entries = [{
        "path": "vMaterials_2/material.mdl", "kind": "file", "bytes": 8,
        "sha256": hashlib.sha256(b"material").hexdigest(),
    }]
    serialized = json.dumps(entries, indent=2) + "\n"
    pin = json.dumps({"archives": [{
        "file": archive.name, "size": archive.stat().st_size,
        "sha256": digest(archive), "destination": ".",
        "inventory_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }]}).encode()
    monkeypatch.setattr(simple_assets, "asset_pin_bytes", lambda: pin)
    receipt = depot / "materials.zip.json"
    if stale:
        old = depot / "data/vMaterials_2/vMaterials_2/material.mdl"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"material")
        receipt.write_text("[]\n")
        with pytest.raises(ValueError, match="current pin"):
            verify_assets(tmp_path)
    assert prepare_assets(tmp_path)["environment_assets_verified"]
    assert receipt.read_text() == serialized
    assert (depot / "data/vMaterials_2/material.mdl").read_bytes() == b"material"
    if stale:
        assert old.read_bytes() == b"material"
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    assert verify_assets(tmp_path) == depot / "data"
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before
    # A refreshed receipt must not make changed asset bytes acceptable.
    (depot / "data/vMaterials_2/material.mdl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_assets(tmp_path)
    receipt.write_text("[]\n")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_assets(tmp_path)


def source(path):
    path = ROOT / "third_party/SIMPLE/src/simple" / path
    if not path.exists():
        pytest.skip("managed source checks require the selected SIMPLE checkout")
    return ast.parse(path.read_text())


def test_simulator_resolves_the_verified_depot(monkeypatch, tmp_path):
    tree = source("core/simulator.py")
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "get_data_dir")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "simulator.py", "exec"), namespace)
    monkeypatch.setitem(
        sys.modules, "simple.utils", types.SimpleNamespace(get_data_dir=lambda: tmp_path)
    )
    assert namespace["get_data_dir"](None) == tmp_path


def test_isaac_capture_does_not_integrate_mirrored_state():
    tree = source("engines/isaacsim.py")
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step")
    calls = []
    namespace = {
        "rep": types.SimpleNamespace(orchestrator=types.SimpleNamespace(
            step=lambda **kw: calls.append(kw)
        )),
        "omni": types.SimpleNamespace(physx=types.SimpleNamespace(
            get_physx_interface=lambda: types.SimpleNamespace(
                update_transformations=lambda *args: calls.append(args)
            )
        )),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "isaacsim.py", "exec"), namespace)
    world = types.SimpleNamespace(
        is_isaac_reset=True, step_id=0,
        world=types.SimpleNamespace(physics_sim_view=types.SimpleNamespace(
            update_articulations_kinematic=lambda: calls.append("kinematics")
        )),
        sync_states=lambda state: calls.append(state),
        update_visuals=lambda: None, _update_collision_spheres=lambda: None,
    )
    namespace["step"](world, "measured state")
    assert calls == [
        "measured state", "kinematics", (False, True, False),
        {"rt_subframes": 1, "delta_time": 0.0, "pause_timeline": True}
    ]
    assert world.step_id == 1


def test_rendered_sonic_waist_matches_physics_geometry():
    tree = source("engines/isaacsim.py")
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "add_robot")
    offsets = next(n.iter for n in ast.walk(method) if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Tuple))
    values = dict(ast.literal_eval(offsets))
    facts = json.loads((ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json").read_text())
    waist = facts["parts"]["waist"]["chain"]
    assert values == {
        "waist_roll_joint": tuple(waist[2]["origin"][:3]),
        "waist_pitch_joint": tuple(waist[4]["origin"][:3]),
    }


def test_isaac_shutdown_returns_to_owner(monkeypatch):
    tree = source("engines/isaac_app.py")
    monkeypatch.setitem(sys.modules, "simple.utils", types.SimpleNamespace(
        env_flag=lambda name, default: default,
    ))
    namespace = {}
    exec(compile(tree, "isaac_app.py", "exec"), namespace)
    configs = []

    def create(config):
        configs.append(config)
        return types.SimpleNamespace(set_setting=lambda *args: None)

    namespace["create_simulation_app"](create, headless=True)
    assert configs[0]["fast_shutdown"] is False


@pytest.mark.parametrize("outcome", ["passed", "mismatch", "exception"])
def test_replay_cleanup_preserves_receipt_and_exit_status(tmp_path, outcome):
    script = r'''
import json, runpy, sys
from pathlib import Path
from types import SimpleNamespace

outcome, directory = sys.argv[1:]
directory = Path(directory)
metadata = dict.fromkeys(("source_lock_sha256", "facts_sha256", "asset_pin_sha256",
                         "loaded_state_sha256", "initialization_frames"), 0)
metadata.update(env_id="test", sim_mode="mujoco_isaac", dr_level=0, seed=0,
                state_dict_json="{}", stabilize_steps=100,
                control_protocol="simple-g1-metric-wbc/4", arm_gravity_compensation=True)
record = dict(schema="roboenv/simple-record/3", execution_mode="stepped",
              initialization=metadata, implementation={}, frames=[], commands=[],
              pending_targets={}, controller_state={}, state=[0.0],
              budget=dict(used=0, max=800, unit="action_frames"),
              native_success=False, done=False, reward=0.0)

class World:
    def __init__(self, **kwargs):
        self.initialization = metadata
        self.implementation = {}
        self.control = SimpleNamespace(targets={})
    def configure_execution_mode(self, mode):
        pass
    def reset(self, seed):
        if outcome == "exception":
            raise RuntimeError("native initialization failed")
    def record(self):
        return dict(record, state=[1.0 if outcome == "mismatch" else 0.0])
    def close(self):
        (directory / "closed").touch()

import roboenv_runtime.adapters.simple.backend as backend
backend.SimpleEnv = World
input_path = directory / "record.json"
input_path.write_text(json.dumps(record))
sys.argv = ["replay_simple_devices.py", str(input_path), "--out", str(directory / "result.json")]
runpy.run_path("tools/replay_simple_devices.py", run_name="__main__")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, outcome, str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert (tmp_path / "closed").exists()
    assert result.returncode == (0 if outcome == "passed" else 1), result.stderr
    receipt = tmp_path / "result.json"
    if outcome == "exception":
        assert not receipt.exists()
        assert "native initialization failed" in result.stderr
    else:
        assert json.loads(receipt.read_text())["passed"] is (outcome == "passed")
