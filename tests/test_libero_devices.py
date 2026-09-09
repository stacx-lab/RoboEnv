"""Official LIBERO contract gates; native physics stays in its isolated interpreter."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, load_tool
from roboenv.tasks.libero import PROBLEMS, ROBOT, device_bindings, entries
from roboenv.tasks.resolve import resolve_task


def require(condition, message):
    if not condition:
        if os.environ.get("ROBOENV_LIBERO_ACCEPTANCE") == "1":
            pytest.fail(message)
        pytest.skip(message)


def interpreter():
    value = os.environ.get("ROBOENV_LIBERO_TEST_PYTHON")
    require(value and Path(value).is_file(), "Requires prepared official LIBERO interpreter")
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
    require(base.exists(), "Requires pinned LIBERO checkout, but no simulator imports")
    tasks = list(entries(base))
    assert len(tasks) == 130
    assert {t["robot"] for t in tasks} == {ROBOT}
    assert {t["problem"] for t in tasks} == set(PROBLEMS)
    assert [
        sum(t["suite_name"] == suite for t in tasks)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    ] == [10, 10, 10, 10, 90]
    assert {p.parent.name for p in (ROOT / "catalog/tasks/libero").glob("*/task.yaml")} == {
        "spatial_0", "90_0", "10_3", "10_9", "drawer-and-bowl",
    }
    for task in ("spatial_0", "90_0", "10_3", "10_9", "drawer-and-bowl"):
        resolved = resolve_task("libero/" + task, project=ROOT)
        assert resolved.task.action_abstraction == "devices"
        assert resolved.task.budget_mode == "practice"
        assert resolved.task.backend_kwargs["init_state_index"] == 0
        assert resolved.task.backend_kwargs["realtime"] is True
        assert not resolved.path.with_name("solution.py").exists()


def test_discovered_scene_mounts_reach_prepared_kinex_workspaces(tmp_path):
    from roboenv.tasks.catalog import discover

    require((ROOT / "third_party/LIBERO/libero/libero").exists(), "Requires pinned LIBERO")
    destination = tmp_path / "discovered"
    assert discover(ROOT, destination, backends=["libero"])["tasks"] == 130
    native = {
        (e["suite_name"], e["task_id"]): e
        for e in entries(ROOT / "third_party/LIBERO/libero/libero")
    }
    checked = set()
    bindings = {p: device_bindings(ROOT / "catalog", p) for p in PROBLEMS}
    for path in (destination / "tasks/libero").glob("*/task.yaml"):
        resolved = resolve_task(path, catalog=ROOT / "catalog")
        kwargs = resolved.task.backend_kwargs
        problem = native[kwargs["suite_name"], kwargs["task_id"]]["problem"]
        assert resolved.task.device_bindings == bindings[problem]
        if problem in checked:
            continue
        workspace = tmp_path / problem
        load_tool("prepare_kinex_workspace").prepare(
            path, workspace, project=ROOT, catalog=ROOT / "catalog"
        )
        for spec in resolved.bindings.devices:
            profile = workspace / ".kinex/devices" / spec.name / "profile.yaml"
            assert yaml.safe_load(profile.read_text()) == spec.resolved_profile(resolved.robot)
        checked.add(problem)
    assert checked == set(PROBLEMS)


def test_catalogs_are_complete_portable_and_use_the_shared_driver(tmp_path):
    from roboenv.devices.export import DRIVERS

    transport = (ROOT / "src/roboenv/assets/_roboenv.py").read_bytes()
    profiles = []
    for rig in (ROBOT,):
        root = ROOT / "catalog/robots" / rig
        assert {p.name for p in root.iterdir()} == {"workspace.yaml", "devices", "mounts.yaml"}
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
    kinex_root = Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex"))
    bun = shutil.which(os.environ.get("BUN", "bun"))
    require(bun and (kinex_root / "node_modules/zod").exists(),
            "Kinex schema integration requires prepared Kinex and Bun")
    payload = tmp_path / "profiles.json"
    from roboenv.devices.bindings.spec import BindingPlan

    for problem in PROBLEMS:
        plan = BindingPlan.load(root, backend="libero", devices=device_bindings(
            ROOT / "catalog", problem,
        ))
        profiles.extend(spec.resolved_profile(root) for spec in plan.devices)
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


@pytest.mark.parametrize("case", range(8))
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
