"""Official LIBERO contract gates; native physics stays in its isolated interpreter."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from helpers import ROOT, kinex_sources, load_tool
from roboenv.tasks.libero import ROBOTS, entries
from roboenv.tasks.resolve import resolve_task


def require(condition, message):
    if not condition:
        if os.environ.get("ROBOENV_LIBERO_ACCEPTANCE") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.mark.parametrize("size", [(128, 128), (256, 128), (512, 512)])
def test_camera_size_must_match_catalog(size):
    from roboenv_runtime.adapters.libero.backend import LiberoEnv

    with pytest.raises(ValueError, match="catalogs require camera_size"):
        LiberoEnv(camera_size=size)


def interpreter():
    value = os.environ.get("ROBOENV_LIBERO_TEST_PYTHON")
    require(value and Path(value).is_file(), "Requires prepared official LIBERO interpreter")
    return value


def run_native(script, *args, timeout=180, render=False):
    result = subprocess.run(
        [interpreter(), str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        env={
            **os.environ,
            **({} if render else {"CUDA_VISIBLE_DEVICES": ""}),
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
    assert {t["robot"] for t in tasks} == set(ROBOTS.values())
    assert {t["problem"] for t in tasks} == set(ROBOTS)
    lookup = {(t["suite_name"], t["task_id"]): t["robot"] for t in tasks}
    for suite, task_id, scene in [
        ("libero_spatial", 0, "tabletop"),
        ("libero_object", 0, "floor"),
        ("libero_10", 0, "living-room"),
        ("libero_10", 2, "kitchen"),
        ("libero_10", 5, "study"),
    ]:
        assert lookup[suite, task_id] == f"libero/libero-panda-{scene}"
    assert [
        sum(t["suite_name"] == suite for t in tasks)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    ] == [10, 10, 10, 10, 90]
    assert {p.parent.name for p in (ROOT / "catalog/tasks/libero").glob("*/task.yaml")} == {
        "bowl-onto-plate",
        "close-top-drawer",
        "bowl-into-bottom-drawer",
        "mug-into-microwave",
        "drawer-and-bowl",
    }
    for task in (
        "bowl-onto-plate",
        "close-top-drawer",
        "bowl-into-bottom-drawer",
        "mug-into-microwave",
        "drawer-and-bowl",
    ):
        resolved = resolve_task("libero/" + task, project=ROOT)
        assert resolved.task.backend == "libero"
        assert resolved.task.budget_mode == "practice"
        assert resolved.task.backend_kwargs["init_state_index"] == 0
        assert "realtime" not in resolved.task.backend_kwargs
        kwargs = resolved.task.backend_kwargs
        assert resolved.task.robot == lookup[kwargs["suite_name"], kwargs["task_id"]]
        assert resolved.task.device_bindings is None
        assert not resolved.path.with_name("solution.py").exists()


def test_discovered_scene_robots_reach_prepared_kinex_workspaces(tmp_path):
    from roboenv.tasks.catalog import discover

    require((ROOT / "third_party/LIBERO/libero/libero").exists(), "Requires pinned LIBERO")
    destination = tmp_path / "discovered"
    assert discover(ROOT, destination, backends=["libero"])["tasks"] == 130
    native = {
        (e["suite_name"], e["task_id"]): e
        for e in entries(ROOT / "third_party/LIBERO/libero/libero")
    }
    checked = set()
    for path in (destination / "tasks/libero").glob("*/task.yaml"):
        resolved = resolve_task(path, catalog=ROOT / "catalog")
        kwargs = resolved.task.backend_kwargs
        problem = native[kwargs["suite_name"], kwargs["task_id"]]["problem"]
        assert resolved.task.robot == native[kwargs["suite_name"], kwargs["task_id"]]["robot"]
        assert resolved.task.device_bindings is None
        assert all(not spec.overrides for spec in resolved.bindings.devices)
        if problem in checked:
            continue
        workspace = tmp_path / problem
        load_tool("prepare_kinex_workspace").prepare(
            path, workspace, project=ROOT, catalog=ROOT / "catalog"
        )
        for spec in resolved.bindings.devices:
            profile = workspace / ".kinex/devices" / spec.name / "profile.yaml"
            assert yaml.safe_load(profile.read_text()) == yaml.safe_load(
                (resolved.robot / spec.profile).read_text()
            )
        assert (workspace / ".kinex/workspace.yaml").read_bytes() == (
            resolved.robot / "workspace.yaml"
        ).read_bytes()
        checked.add(problem)
    assert checked == set(ROBOTS)


def test_catalogs_are_complete_portable_and_use_the_shared_driver(tmp_path):
    from roboenv.devices.export import DRIVERS

    transport = (ROOT / "src/roboenv/assets/_roboenv.py").read_bytes()
    profiles = []
    assert {p.name for p in (ROOT / "catalog/robots/libero").iterdir()} == {
        Path(rig).name for rig in ROBOTS.values()
    }
    for rig in ROBOTS.values():
        root = ROOT / "catalog/robots" / rig
        assert {p.name for p in root.iterdir()} == {"workspace.yaml", "devices"}
        assert {p.name for p in (root / "devices").iterdir()} == {
            "arm",
            "gripper",
            "camera",
            "camera_wrist",
            "world",
            "_roboenv.py",
        }
        for file in root.rglob("*"):
            if file.is_file():
                assert b"???" not in file.read_bytes()
                assert file.name != "connection.json"
        for path in root.glob("devices/*/profile.yaml"):
            profile = yaml.safe_load(path.read_text())
            profiles.append(profile)
            assert profile["access"]["driver"]["persistent"] is True
            assert (
                path.with_name("driver.py").read_bytes()
                == (ROOT / "src/roboenv/assets" / DRIVERS[profile["type"]]).read_bytes()
            )
            assert not path.with_name("_roboenv.py").exists()
            assert (root / "devices/_roboenv.py").read_bytes() == transport
            if profile["type"] == "joint":
                assert len(profile["intrinsics"]["model"]["joints"]) == 7
                assert profile["intrinsics"]["control"] == {
                    "mode": "position",
                    "limit_handling": "native",
                }
            if profile["type"] == "rgb":
                color = profile["intrinsics"]["color"]
                assert (color["width"], color["height"]) == (256, 256)
                focal = {
                    "camera": 309.019335983756,
                    "camera_wrist": 166.812847723674,
                }[profile["name"]]
                assert color["projection"]["fx"] == pytest.approx(focal)
                assert color["projection"]["fy"] == pytest.approx(focal)
                assert color["projection"]["cx"] == 128
                assert color["projection"]["cy"] == 128
        assert "simulator" not in str(
            yaml.safe_load((root / "devices/arm/profile.yaml").read_text())["access"]
        )
    kinex_root, kinex_src = kinex_sources()
    bun = shutil.which(os.environ.get("BUN", "bun"))
    require(
        bun and kinex_src and (kinex_root / "node_modules/zod").exists(),
        "Kinex schema integration requires prepared Kinex and Bun",
    )
    payload = tmp_path / "profiles.json"
    payload.write_text(json.dumps(profiles))
    workspace = tmp_path / "workspace"
    load_tool("prepare_kinex_workspace").prepare("libero/bowl-onto-plate", workspace, project=ROOT)
    checked = subprocess.run(
        [
            bun,
            "-e",
            f"const {{Device}} = await import('{kinex_src}/device/profile/schema.ts'); "
            f"const {{missingActions}} = await import('{kinex_src}/device/interface.ts'); "
            "for (const p of JSON.parse(await Bun.file(process.argv[1]).text())) { "
            "const d = Device.parse(p); "
            "if(missingActions(d, d.access?.driver?.actions ?? []).length) "
            "throw new Error('missing actions'); } "
            f"const {{calibrationPlan}} = await import('{kinex_src}/calibration/dag.ts'); "
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
        expected = {
            path.relative_to(ROOT / "catalog")
            for subtree in ("robots/libero",)
            for path in (ROOT / "catalog" / subtree).rglob("*")
            if path.is_file()
        }
        actual = {p.relative_to(destination) for p in destination.rglob("*") if p.is_file()}
        assert actual == expected
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


def test_native_camera_pixels_survive_concurrent_rpc(tmp_path):
    require(
        os.environ.get("ROBOENV_LIBERO_RENDER_TESTS") == "1",
        "Requires explicitly enabled native EGL camera tests",
    )
    run_native(
        "tests/libero_camera_rpc.py",
        "--out",
        tmp_path,
        render=True,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] and report["camera_reads"] == 10
