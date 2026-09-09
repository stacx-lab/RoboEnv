"""Official LIBERO contract gates; native physics stays in its isolated interpreter."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, load_tool
from roboenv.tasks.libero import RIGS, entries
from roboenv.tasks.resolve import resolve_task


def interpreter():
    value = os.environ.get("ROBOENV_LIBERO_TEST_PYTHON")
    if not value or not Path(value).is_file():
        pytest.skip("Requires prepared official LIBERO interpreter")
    return value


def run_native(script, *args, timeout=180):
    result = subprocess.run(
        [interpreter(), str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "MUJOCO_GL": "egl",
            "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("src", "runtime/src")),
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_official_registry_and_assets_are_complete():
    base = ROOT / "third_party/LIBERO/libero/libero"
    if not base.exists():
        pytest.skip("Requires pinned LIBERO checkout, but no simulator imports")
    tasks = list(entries(base))
    assert len(tasks) == 130
    assert set(t["robot"] for t in tasks) == set(RIGS.values())
    assert [
        sum(t["suite_name"] == suite for t in tasks)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    ] == [10, 10, 10, 10, 90]
    for task in ("spatial_0", "drawer-and-bowl", "mug-into-microwave"):
        resolved = resolve_task("libero/" + task, project=ROOT)
        assert resolved.task.action_abstraction == "devices"
        assert resolved.task.budget_mode == "practice"
        assert resolved.task.backend_kwargs["init_state_index"] == 0
        assert resolved.task.backend_kwargs["realtime"] is True
        assert not resolved.path.with_name("solution.py").exists()


def test_catalogs_are_complete_portable_and_use_the_shared_driver(tmp_path):
    from roboenv.devices.export import DRIVERS

    transport = (ROOT / "src/roboenv/assets/_roboenv.py").read_bytes()
    profiles = []
    for rig in RIGS.values():
        root = ROOT / "catalog/robots" / rig
        assert {p.name for p in root.iterdir()} == {"workspace.yaml", "devices"}
        assert {p.name for p in (root / "devices").iterdir()} == {
            "arm",
            "gripper",
            "camera",
            "camera_wrist",
            "world",
        }
        for file in root.rglob("*"):
            if file.is_file():
                assert b"???" not in file.read_bytes()
                assert file.name != "connection.json"
        for path in root.glob("devices/*/profile.yaml"):
            profile = yaml.safe_load(path.read_text())
            profiles.append(profile)
            if profile["type"] == "world":
                assert not path.with_name("driver.py").exists()
            else:
                assert path.with_name("driver.py").read_bytes() == (
                    ROOT / "src/roboenv/assets" / DRIVERS[profile["type"]]
                ).read_bytes()
                assert path.with_name("_roboenv.py").read_bytes() == transport
            if profile["type"] == "manipulator":
                assert len(profile["intrinsics"]["kinematics"]["model"]["joints"]) == 7
                assert profile["intrinsics"]["control"]["speed"] is None
            if profile["type"] == "rgb":
                assert profile["intrinsics"]["color"]["width"] == 128
        assert "simulator" not in str(
            yaml.safe_load((root / "devices/arm/profile.yaml").read_text())["access"]
        )
    kinex_root = ROOT.parent / "kinex"
    bun = shutil.which("bun")
    if not bun or not (kinex_root / "node_modules/zod").exists():
        pytest.skip("Kinex schema integration requires sibling Kinex and Bun")
    payload = tmp_path / "profiles.json"
    payload.write_text(json.dumps(profiles))
    workspace = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare("libero/spatial_0", workspace, project=ROOT)
    checked = subprocess.run(
        [
            bun,
            "-e",
            "const {Device} = await import('./src/calibration/schema.ts'); "
            "const {missingActions} = await import('./src/device/interface.ts'); "
            "for (const p of JSON.parse(await Bun.file(process.argv[1]).text())) { "
            "const d = Device.parse(p); "
            "if(missingActions(d, d.access?.driver?.actions ?? []).length) "
            "throw new Error('missing actions'); } "
            "const {calibrationPlan} = await import('./src/calibration/dag.ts'); "
            "const p = await calibrationPlan(process.argv[2]); "
            "if(!p.items.some(i=>i.id==='intrinsics:arm')) throw new Error('no devices'); "
            "for(const i of p.items) if(i.device && i.status!=='done') "
            "throw new Error(JSON.stringify(i)); "
            "if(p.items.find(i=>i.id==='task-ready')?.status!=='done') "
            "throw new Error('not ready');",
            str(payload),
            str(workspace),
        ],
        cwd=kinex_root,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "KINEX_CONFIG_DIR": str(tmp_path / "config")},
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_static_generation_is_deterministic(tmp_path):
    for name in ("first", "second"):
        destination = tmp_path / name
        run_native("tools/generate_libero_config.py", "--out", destination)
        for path in destination.rglob("*"):
            if path.is_file():
                assert (
                    path.read_bytes()
                    == (ROOT / "catalog" / path.relative_to(destination)).read_bytes()
                )


@pytest.mark.parametrize("case", range(7))
@pytest.mark.parametrize("init", [0, 1])
def test_native_geometry_control_and_initialization(tmp_path, case, init):
    run_native("tests/libero_native.py", "--out", tmp_path, "--case", case, "--init", init)
    report = json.loads((tmp_path / "report.json").read_text())[0]
    assert report["passed"] and not report["full_task_attempted"]
    assert report["fk_error_m"] < 1e-9


def test_clock_budget_and_replay(tmp_path):
    run_native("tests/libero_clock.py", "--out", tmp_path)


def test_native_contact_stop_and_release(tmp_path):
    run_native("tests/libero_contact.py", "--out", tmp_path / "contact.json")
    assert json.loads((tmp_path / "contact.json").read_text())["passed"]
