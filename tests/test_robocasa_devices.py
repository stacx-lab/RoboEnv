"""Task-specific devices and the small endpoint client. No GPU or model runs."""

import ast
import json
import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml
from helpers import kinex_sources, native_interpreter
from roboenv.client import Client, RemoteError
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.client import Environment
from roboenv.devices.export import export_devices
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def test_task_selects_and_renames_its_own_devices(tmp_path):
    task = yaml.safe_load(
        (ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml").read_text()
    )
    task["device_bindings"] = {
        "overview": {
            "binding": "robocasa.camera/1",
            "entity": "camera",
            "profile": "devices/camera_left/profile.yaml",
            "config": {"camera": "agentview"},
        },
        "world": {
            "binding": "robocasa.world/1",
            "entity": "world",
            "profile": "devices/world/profile.yaml",
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    resolved = resolve_task(path, catalog=ROOT / "catalog")
    assert [s.name for s in resolved.bindings.devices] == ["overview", "world"]
    assert resolved.bindings.devices[0].config == {"camera": "agentview"}
    task["device_bindings"]["overview"]["profile"] = "../../missing.yaml"
    path.write_text(yaml.safe_dump(task))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_task(path, catalog=ROOT / "catalog")


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
def test_endpoint_client_has_no_primitive_catalog_and_returns_data(device_service):
    _, supervisor, _ = device_service
    transport = Client(supervisor.http_endpoint)
    env = Environment(transport)
    for name in ("api", "call", "observe", "request", "describe"):
        assert not hasattr(env, name)
    with pytest.raises(FrozenInstanceError):
        env.task.instruction = "changed"
    profiles = env.devices()
    profiles[0]["name"] = "changed"
    assert env.devices()[0]["name"] == "controller"
    with pytest.raises(KeyError):
        env.device("missing")
    controller = env.device("controller")
    assert controller.read() == transport.device("controller").read()["data"]
    for op in ("api", "call", "observe", "score.peek", "capture"):
        with pytest.raises((ValueError, RemoteError)):
            transport.request(op)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        controller.read()
    assert transport.request("health")["worker"]["alive"]


def test_type_specific_handles_and_profile_only_world(tmp_path):
    class Transport:
        def describe(self):
            return {"task": {"id": "test", "instruction": "Pick a bottle", "context": {"a": [1]}}}

        def devices(self):
            return {
                "devices": [
                    {"name": "camera", "type": "rgb", "actions": {"read": {"supported": True}}},
                    {
                        "name": "world",
                        "type": "world",
                        "intrinsics": {},
                        "extrinsics": {},
                        "actions": {"read": {"supported": True}, "wait": {"supported": True}},
                    },
                ]
            }

    env = Environment(Transport())
    assert hasattr(env.device("camera"), "read")
    assert not hasattr(env.device("camera"), "write")
    assert not hasattr(env.device("camera"), "ik")
    assert hasattr(env.device("world"), "read")
    assert hasattr(env.device("world"), "wait")
    with pytest.raises(TypeError):
        env.task.context["a"] = []
    export_devices([env.device("world").profile], tmp_path, {"endpoint": "http://sim:8123"})
    assert (tmp_path / "world/profile.yaml").is_file()
    assert (tmp_path / "world/driver.py").exists()


def test_native_panda_devices_in_isolated_cpu_interpreter(tmp_path):
    kinex, _ = kinex_sources()
    if not (kinex / "packages/core/assets/python/kinex/kinematics.py").is_file():
        pytest.skip("Driver-local IK/FK integration requires the sibling Kinex utility")
    interpreter = native_interpreter()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "runtime/src")]),
    }
    result = subprocess.run(
        [interpreter, str(ROOT / "tests/robocasa_cpu.py"), str(ROOT), str(tmp_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["passed"] and not report["gpu_replay"]
    check_kinex_profiles(tmp_path, list((tmp_path / "devices").glob("*/profile.yaml")))


def check_kinex_profiles(tmp_path, paths):
    # Check actual Kinex schema/interface compatibility when the sibling harness
    # is present. RoboEnv itself retains no runtime dependency on Kinex.
    import shutil

    kinex_root, kinex_src = kinex_sources()
    if kinex_src and not (kinex_root / "node_modules/zod").exists():
        isolated = ROOT / ".cache/kinex-schema"
        if (isolated / "node_modules/zod").exists():
            # Reuse dependencies only; always test the sibling's CURRENT sources,
            # including its uncommitted contract updates.
            relative = Path(kinex_src.removeprefix("./"))
            current = tmp_path / "kinex-current"
            (current / relative.parent).mkdir(parents=True, exist_ok=True)
            shutil.copytree(kinex_root / relative, current / relative)
            shutil.copyfile(kinex_root / "package.json", current / "package.json")
            (current / "node_modules").symlink_to(
                isolated / "node_modules", target_is_directory=True
            )
            kinex_root = current
    if not shutil.which("bun") or not kinex_src or not (kinex_root / "node_modules/zod").exists():
        pytest.skip("Requires Bun and Kinex schema dependencies")
    payload = tmp_path / "profiles.json"
    payload.write_text(json.dumps([yaml.safe_load(path.read_text()) for path in paths]))
    code = (
        f"const {{Device, RobotManifest}} = await import('{kinex_src}/device/profile/schema.ts'); "
        "RobotManifest.parse(JSON.parse(await Bun.file(process.argv[2]).text())); "
        f"const {{missingActions}} = await import('{kinex_src}/device/interface.ts'); "
        "for (const value of JSON.parse(await Bun.file(process.argv[1]).text())) { "
        "const device = Device.parse(value); "
        "if (device.type === 'joint' && "
        "(!device.intrinsics.home || "
        "Object.keys(device.intrinsics.home).length !== "
        "device.intrinsics.model.joints.length)) "
        "throw new Error('Missing home joints: ' + device.name); "
        "if (missingActions(device, device.access?.driver?.actions ?? []).length) "
        "throw new Error('Missing interface actions: ' + device.name); } "
        f"const {{calibrationPlan}} = await import('{kinex_src}/calibration/dag.ts'); "
        "const plan = await calibrationPlan(process.argv[3]); "
        "for (const item of plan.items) { "
        "if (item.device && item.status !== 'done') "
        "throw new Error(item.id + ': ' + item.detail); } "
        "if (!plan.items.some(i => i.id === 'intrinsics:arm')) throw new Error('No devices'); "
        "if (!plan.items.find(i => i.id === 'extrinsics:base')"
        "?.detail.includes('runtime pose in world')) "
        "throw new Error('Mobile base must use its declared live world pose'); "
        "if (plan.items.find(i => i.id === 'task-ready')?.status !== 'done') "
        "throw new Error('Explicit null caps must satisfy the safety policy');"
    )
    manifest = tmp_path / "workspace.json"
    manifest.write_text(
        json.dumps(
            yaml.safe_load(
                (ROOT / "catalog/robots/robocasa/robocasa-panda-omron/workspace.yaml").read_text()
            )
        )
    )
    workspace = tmp_path / "kinex-workspace"
    shutil.copytree(
        ROOT / "catalog/robots/robocasa/robocasa-panda-omron",
        workspace / ".kinex",
    )
    checked = subprocess.run(
        ["bun", "-e", code, str(payload), str(manifest), str(workspace)],
        cwd=kinex_root,
        env={**os.environ, "KINEX_CONFIG_DIR": str(tmp_path / "kinex-config")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_static_catalog_conforms_to_kinex(tmp_path):
    devices = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices"
    check_kinex_profiles(
        tmp_path,
        sorted(devices.glob("*/profile.yaml")),
    )


def test_static_robocasa_config_regenerates_without_a_task(tmp_path):
    result = subprocess.run(
        [
            native_interpreter(),
            str(ROOT / "tools/generate_robocasa_config.py"),
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    robot = ROOT / "catalog/robots/robocasa/robocasa-panda-omron"
    assert {p.name for p in (robot / "devices").iterdir()} == {
        "arm",
        "base",
        "torso",
        "gripper",
        "camera_left",
        "camera_right",
        "camera_wrist",
        "world",
        "_roboenv.py",
    }
    assert not list(robot.rglob("connection.json"))
    assert {p.name for p in robot.iterdir()} == {"workspace.yaml", "devices"}
    generated = {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    expected = {
        p.relative_to(robot): p.read_bytes()
        for p in robot.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert generated == expected
    for name in ("arm", "torso", "base", "gripper"):
        actuator = yaml.safe_load((robot / "devices" / name / "profile.yaml").read_text())
        assert "control_interval_s" not in actuator["intrinsics"]
        assert "frequency_hz" not in actuator["access"]
    camera = yaml.safe_load((robot / "devices/camera_left/profile.yaml").read_text())
    assert camera["extrinsics"]["mount"]["parent"] == "torso"
    assert camera["extrinsics"]["mount"]["parent_frame"] == "top"
    for name in ("camera_left", "camera_right", "camera_wrist"):
        profile = yaml.safe_load((robot / "devices" / name / "profile.yaml").read_text())
        assert profile["type"] == "rgbd"
        assert profile["intrinsics"]["depth"]["scale_m_per_unit"] == 0.001
        assert profile["intrinsics"]["depth"]["registration"] == "color"
    body = yaml.safe_load((robot / "devices/base/profile.yaml").read_text())
    assert body["type"] == "velocity"
    assert body["extrinsics"] == {"source": "introspected"}
    assert "execution" not in body["intrinsics"]["control"]
    assert body["intrinsics"]["feedback"]["pose_frame"] == "world"
    torso = yaml.safe_load((robot / "devices/torso/profile.yaml").read_text())
    assert torso["extrinsics"]["mount"]["parent"] == "base"
    world = yaml.safe_load((robot / "devices/world/profile.yaml").read_text())
    assert world["intrinsics"] == {
        "source": "default",
        "bounds": None,
        "surfaces": [],
        "method": "unknown",
        "calibrated_at": None,
    }
    assert "access" in world and "extrinsics" not in world


@pytest.mark.parametrize("device_service", ["http"], indirect=True)
@pytest.mark.parametrize("extension", ["png", "pgm"])
def test_images_download_to_client_files_and_are_episode_fenced(
    device_service, tmp_path, extension
):
    import http.client
    from urllib.parse import urlsplit

    _, supervisor, _ = device_service
    env = Environment(Client(supervisor.http_endpoint))
    captures = supervisor.config.run_dir / "captures"
    captures.mkdir()
    remote = captures / ("a" * 32 + "." + extension)
    remote.write_bytes(b"test-image")
    local = Path(env._download(str(remote)))
    assert local != remote and local.read_bytes() == b"test-image"
    endpoint = urlsplit(supervisor.http_endpoint)
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port)
    try:
        connection.request("GET", "/artifacts/old-episode/0/" + remote.name)
        assert connection.getresponse().status == 409
    finally:
        connection.close()
    env.close()
    assert not local.exists()
    assert remote.is_file()


def test_two_native_task_services_are_independent(tmp_path, monkeypatch):
    import shutil
    import threading
    import time
    from contextlib import ExitStack

    import roboenv
    from roboenv_runtime.runtime.protocol import socket_address
    from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig

    interpreter = native_interpreter()
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(str(ROOT / relative) for relative in ("src", "runtime/src", "tests")),
    )
    monkeypatch.delenv("ROBOENV_SOCKET_ROOT", raising=False)
    monkeypatch.delenv("ROBOENV_SERVER_MANIFEST", raising=False)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setenv("ROBOENV_TASKS", str(tasks))
    environments, owners, supervisors = [], [], []
    with ExitStack() as stack:
        for index in range(2):
            task_id = f"cpu-robocasa-{index}"
            (tasks / f"{task_id}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": task_id,
                        "backend": "robocasa",
                        "backend_task": task_id,
                        "robot": "robocasa/robocasa-panda-omron",
                        "instruction": "Lift the cube.",
                    }
                )
            )
            robot = tmp_path / f"robot-{index}"
            shutil.copytree(ROOT / "catalog/robots/robocasa/robocasa-panda-omron", robot)
            plan = BindingPlan.load(
                ROOT / "catalog/robots/robocasa/robocasa-panda-omron", backend="robocasa"
            ).as_dict()
            if index:
                plan["devices"] = {
                    name: value
                    for name, value in plan["devices"].items()
                    if name in {"world", "base", "camera_left"}
                }
            (robot / "bindings.yaml").write_text(yaml.safe_dump(plan))
            monkeypatch.setenv("ROBOENV_SERVER_ROBOT", str(robot))
            config = SupervisorConfig(
                run_dir=tmp_path / f"run-{index}",
                task=task_id,
                seed=index,
                budget_mode="official",
                session=f"cpu-{index}",
                http_port=0,
                http_host="127.0.0.1" if index == 0 else "0.0.0.0",
                python_executable=interpreter,
                worker_module="robocasa_cpu",
                worker_extra_args=("--execution-mode", "stepped"),
            )
            supervisor = Supervisor(config)
            stack.callback(supervisor.close)
            supervisor.start_worker()
            supervisor.open_server()
            thread = threading.Thread(target=supervisor.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, 5)
            stack.callback(supervisor._schedule_server_shutdown)
            owner = Client(socket_address(config.public_socket), session=config.session)
            owner.request("devices.list", timeout=30)
            deadline = time.monotonic() + 5
            while supervisor.http_endpoint is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            environments.append(roboenv.connect(supervisor.http_endpoint))
            owners.append(owner)
            supervisors.append(supervisor)
        first, second = environments
        assert supervisors[0].http_endpoint != supervisors[1].http_endpoint
        assert first.task.id != second.task.id
        assert first.task.instruction == "Lift the cube (CPU service fixture)."
        assert dict(first.task.context) == {}
        assert "scene" not in {p["name"] for p in first.devices()}
        public = Client(supervisors[0].http_endpoint)
        description = public.describe()
        assert set(description) == {"task", "devices", "session", "generation", "device_contract"}
        assert set(description["task"]) == {"id", "instruction", "context"}
        for operation in ("observe", "scene", "score.peek", "api"):
            public.allowed_ops = public.allowed_ops | {operation}
            with pytest.raises(RemoteError, match="unsupported_op"):
                public.request(operation)
        # Even a known verifier request ID cannot expose its privileged result.
        privileged = owners[0].request("observe", request_id="private-observation")
        assert "cube" in privileged["objects"]
        with pytest.raises(RemoteError, match="unsupported_op"):
            public.operation("private-observation")
        assert "seed" not in public.request("health")
        assert {p["name"] for p in second.devices()} == {"world", "base", "camera_left"}
        with pytest.raises(KeyError):
            second.device("arm")
        first.device("arm").write(
            duration_s=len([first.device("arm").read()["positions"]]) / 20,
            actions=[first.device("arm").read()["positions"]],
        )
        first.device("world").wait(0.05)
        assert owners[0].request("observe")["budget"]["used"] == 1
        assert owners[1].request("observe")["budget"]["used"] == 0
        image = first.device("camera_left").read()
        assert Path(image["file"]).read_bytes().startswith(b"\x89PNG")
        assert image["width"] == 12
        budget_before = owners[0].request("observe")["budget"]["used"]
        frames = []
        depths = []
        for name in ("camera_left", "camera_right", "camera_wrist"):
            reading = first.device(name).read()
            frames.append(Path(reading["file"]).read_bytes())
            depth = Path(reading["depth_file"])
            assert depth.parent == Path(reading["file"]).parent
            assert depth.parent != supervisors[0].config.run_dir / "captures"
            assert reading["sample_time_s"] == reading["depth_sample_time_s"]
            assert (reading["depth_width"], reading["depth_height"]) == (12, 8)
            depths.append(depth.read_bytes())
        assert all(d.startswith(b"P5\n12 8\n65535\n") for d in depths)
        assert len(set(depths)) == 3
        assert len(set(frames)) == 3
        assert owners[0].request("observe")["budget"]["used"] == budget_before
        with pytest.raises(KeyError):
            second.device("camera_right")
        import shutil
        import sys

        copied = tmp_path / "workspace/.kinex/robots/robocasa/robocasa-panda-omron"
        shutil.copytree(ROOT / "catalog/robots/robocasa/robocasa-panda-omron", copied)
        # This task renders at 12x8; install its live calibration before reading.
        # Keep endpoint selection through deployment environment variables.
        export_devices(
            [p for p in first.devices() if p["type"] in {"rgb", "rgbd"}],
            copied / "devices",
            {"endpoint": supervisors[0].http_endpoint},
        )
        for connection in copied.rglob("connection.json"):
            connection.unlink()
        before = {p.relative_to(copied): p.read_bytes() for p in copied.rglob("*") if p.is_file()}
        for supervisor in supervisors:
            driver = subprocess.run(
                [sys.executable, str(copied / "devices/camera_left/driver.py"), "read", "{}"],
                cwd=copied,
                env={**os.environ, "ROBOENV_ENDPOINT": supervisor.http_endpoint},
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert driver.returncode == 0, driver.stderr
            response = json.loads(driver.stdout)
            assert response["ok"], response
            local_image = Path(response["file"])
            assert local_image.read_bytes().startswith(b"\x89PNG")
            local_depth = Path(response["depth_file"])
            assert local_depth.read_bytes().startswith(b"P5\n")
            local_depth.unlink()
            local_image.unlink()
            local_image.parent.rmdir()
        assert before == {
            p.relative_to(copied): p.read_bytes() for p in copied.rglob("*") if p.is_file()
        }
        driver = subprocess.run(
            [sys.executable, str(copied / "devices/arm/driver.py"), "read", "{}"],
            cwd=copied,
            env={**os.environ, "ROBOENV_ENDPOINT": supervisors[1].http_endpoint},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert not json.loads(driver.stdout)["ok"]  # second test rig deliberately omits arm
        driver = subprocess.run(
            [sys.executable, str(copied / "devices/camera_left/driver.py"), "read", "{}"],
            cwd=copied,
            env={**os.environ, "ROBOENV_ENDPOINT": supervisors[0].http_endpoint},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert driver.returncode == 0, driver.stderr
        reply = json.loads(driver.stdout)
        assert reply["ok"], reply
        # The one-shot process must not delete the image as it exits.
        driver_image = Path(reply["file"])
        assert driver_image.read_bytes().startswith(b"\x89PNG")
        driver_depth = Path(reply["depth_file"])
        assert driver_depth.read_bytes().startswith(b"P5\n")
        driver_depth.unlink()
        driver_image.unlink()
        driver_image.parent.rmdir()
        assert not hasattr(first.device("camera_left"), "write")
        first.close()
        second.close()
        assert owners[0].request("health")["worker"]["alive"]


def test_native_sync_motion(tmp_path):
    kinex, _ = kinex_sources()
    if not (kinex / "packages/core/assets/builtin/tool/servo/tool.py").is_file():
        pytest.skip("requires Kinex servo sources")
    result = subprocess.run(
        [
            native_interpreter(),
            str(ROOT / "tests/robocasa_sync_cpu.py"),
            str(ROOT),
            str(tmp_path),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "runtime/src")]),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["passed"]


@pytest.mark.parametrize(
    "device_type,config", [("world", {}), ("rgb", {}), ("world", {"unused": True})]
)
def test_world_binding_preserves_profile_geometry(tmp_path, device_type, config):
    from roboenv.devices.bindings.registry import BindingContext, BindingRegistry
    from roboenv.devices.bindings.spec import BindingSpec
    from roboenv_runtime.bindings.backends.robocasa import register

    intrinsics = {
        "frames": {"fixture": [1, 0, 0, 0, 0, 0, 1]},
        "bounds": {"min": [-2, -2, 0], "max": [2, 2, 3]},
        "surfaces": [],
        "source": "calibration",
    }
    (tmp_path / "world.yaml").write_text(
        yaml.safe_dump({"type": device_type, "intrinsics": intrinsics})
    )
    spec = BindingSpec("room", "robocasa.world/1", "world", "world.yaml", config, {})
    plan = BindingPlan("robocasa", (spec,))
    registry = BindingRegistry()
    register(registry)
    context = BindingContext("robocasa", "robot", None)
    if device_type != "world" or config:
        with pytest.raises(ValueError, match="world binding"):
            registry.assemble(plan, tmp_path, context)
        return
    service = registry.assemble(plan, tmp_path, context)
    descriptor = service.describe("room")
    assert descriptor["intrinsics"] == intrinsics
    assert {"read", "wait"} <= descriptor["actions"].keys()
    service.close()


def test_config_provenance_names_only_existing_sources():
    # A removed module left in the receipt's source list made --provenance crash after the
    # catalog was written; every in-repository input must still exist.
    tree = ast.parse((ROOT / "tools/generate_robocasa_config.py").read_text())
    sources = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "sources" for target in node.targets)
    )
    paths = [ast.literal_eval(item) for item in sources.elts]
    local = [path for path in paths if not path.startswith("third_party/")]
    assert local and all((ROOT / path).is_file() for path in local)
    receipt = json.loads((ROOT / "docs/robocasa-config.provenance.json").read_text())
    assert set(receipt["sources"]) == set(paths)
