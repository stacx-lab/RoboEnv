"""Go2 binding contracts on CPU; native rendered replay is explicitly opt-in."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import DeviceError
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.geometry import relative_pose, rotation
from roboenv_runtime.adapters.isaaclab.rig import CAMERA_ROTATION, FEET, JOINTS, facts
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]


def test_control_interval_samples_imu_once_per_physics_step():
    from roboenv_runtime.adapters.isaaclab.backend import IsaacLabEnv
    from roboenv_runtime.adapters.isaaclab.rig import CONTROL_DT, DECIMATION, PHYSICS_DT

    class Sensor:
        elapsed = 0.0
        samples = 0

        def update(self, dt, **kwargs):
            self.dt = dt
            self.elapsed += dt

        @property
        def data(self):
            assert self.dt == PHYSICS_DT
            self.samples += 1
            return 1.0 / self.dt

    class Scene(dict):
        def write_data_to_sim(self):
            pass

        def update(self, dt):
            for sensor in self.values():
                sensor.update(dt)

    env = IsaacLabEnv.__new__(IsaacLabEnv)
    env.scene = Scene({name: Sensor() for name in ("imu", *FEET)})
    env.robot = SimpleNamespace(set_joint_position_target_index=lambda **kwargs: None)
    env.sim = SimpleNamespace(step=lambda **kwargs: None)
    env.target = None
    env._interval()
    assert all(sensor.samples == DECIMATION for sensor in env.scene.values())
    assert all(sensor.elapsed == pytest.approx(CONTROL_DT) for sensor in env.scene.values())


class World:
    """Native effects spy: tests exercise the real binding and DeviceService."""

    def __init__(self):
        self.model = facts()
        self.steps = 0
        self.target = self.model["home"]

    def write(self, target):
        self.target = target.tolist()
        self.steps += 1
        return {"accepted": True}

    def stop(self):
        return {"stopped": True}

    def read_body(self):
        return {
            "positions": self.target,
            "velocities": [0.0] * 12,
            "frame_poses": {foot: [0, 0, 0, 0, 0, 0, 1] for foot in FEET},
            "sample_time_s": self.steps * 0.02,
        }

    def read_sensor(self, entity):
        return {
            "sample_time_s": self.steps * 0.02,
            **(
                {
                    "angular_velocity": {"x": 0, "y": 0, "z": 0},
                    "linear_acceleration": {"x": 0, "y": 0, "z": 9.81},
                    "orientation": None,
                    "magnetic_field": None,
                }
                if entity == "imu"
                else {"force": {"x": 0, "y": 0, "z": 20}, "torque": None}
            ),
        }


@pytest.fixture
def rig(tmp_path):
    resolved = resolve_task("isaaclab/go2-posture", project=ROOT, realtime=False)
    world = World()
    context = BindingContext("isaaclab", "go2", SimpleNamespace(_env=world, run_dir=tmp_path))
    service = default_registry().assemble(resolved.bindings, resolved.robot, context)
    yield world, service
    service.close()


def call(service, device, action="read", **args):
    return service.call({"device": device, "action": action, "args": args})["data"]


def test_go2_binding_commands_and_read_only_sensors(rig):
    world, service = rig
    target = facts()["posture"]
    for _ in range(2):
        assert call(service, "body", "write", target=target) == {"accepted": True}
    assert world.steps == 2 and world.target == target
    assert call(service, "body")["positions"] == target
    assert call(service, "imu")["linear_acceleration"] == {"x": 0, "y": 0, "z": 9.81}
    assert call(service, "imu")["orientation"] is None
    assert call(service, "imu")["magnetic_field"] is None
    for foot in FEET:
        assert call(service, f"contact_{foot.lower()}")["force"] == {"x": 0, "y": 0, "z": 20}
        assert call(service, f"contact_{foot.lower()}")["torque"] is None
    assert call(service, "body", "stop") == {"stopped": True}
    assert call(service, "body", "unload") == {"unloaded": True}
    assert world.steps == 2
    for target in ([0] * 11, [True] * 12, [float("nan")] * 12, [99] * 12):
        with pytest.raises(DeviceError):
            call(service, "body", "write", target=target)
    for entity, action in (("body", "ik"), ("imu", "write"), ("world", "read")):
        with pytest.raises(DeviceError):
            call(service, entity, action)
    assert world.steps == 2


def test_go2_rejects_overlapping_ownership_and_realtime(tmp_path):
    task = resolve_task("isaaclab/go2-posture", project=ROOT)
    body = task.bindings.devices[0]
    duplicate = replace(body, name="second_body")
    context = BindingContext("isaaclab", "go2", SimpleNamespace(_env=World()))
    with pytest.raises(ValueError, match="one logical"):
        default_registry().assemble(
            replace(task.bindings, devices=(body, duplicate)),
            task.robot,
            context,
        )
    with pytest.raises(ValueError, match="Clock"):
        resolve_task("isaaclab/go2-posture", project=ROOT, realtime=True)


def test_frame_and_camera_conventions():
    # Optical right/down/forward -> body left-negative/down/forward.
    assert np.allclose(rotation(CAMERA_ROTATION), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    parent = [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    child = [1, 3, 3, *parent[3:]]
    assert np.allclose(relative_pose(parent, child), [1, 0, 0, 0, 0, 0, 1])


def test_isaaclab_discovery_does_not_import_simulator():
    code = """
import sys
from roboenv_runtime.core.adapter import BUILTIN_ADAPTERS
from roboenv_runtime.adapters.isaaclab import IsaacLabAdapter
from roboenv.tasks.catalog import _isaaclab
from pathlib import Path
assert 'isaaclab' in BUILTIN_ADAPTERS
assert next(_isaaclab(Path('.'), []))[0].id == 'isaaclab/go2-posture'
assert 'isaaclab' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'runtime/src'}"},
    )


def test_go2_catalog_and_complex_schema(tmp_path):
    task = resolve_task("isaaclab/go2-posture", project=ROOT)
    body = yaml.safe_load((task.robot / "devices/body/profile.yaml").read_text())
    assert [j["name"] for j in body["intrinsics"]["joints"]] == list(JOINTS)
    assert body["intrinsics"]["home_joints"] == facts()["home"]
    assert body["extrinsics"]["frames"] == list(FEET)
    assert "mount" not in body["extrinsics"]
    for driver in task.robot.glob("devices/*/driver.py"):
        profile = yaml.safe_load(driver.with_name("profile.yaml").read_text())
        typed = profile["type"] in {"force_torque", "imu"}
        asset = (
            "imu-client.py"
            if profile["type"] == "imu"
            else "force-torque-client.py"
            if typed
            else "device-client.py"
        )
        assert driver.read_bytes() == (ROOT / "src/roboenv/assets" / asset).read_bytes()
        access = profile["access"]
        assert (access["connection"]["url"] if typed else access["url"]) == "http://127.0.0.1:8123"
        assert access["driver"]["persistent"] is False
    complex_root = ROOT.parent / "complex"
    if not shutil.which("bun") or not (complex_root / "node_modules/zod").exists():
        pytest.skip("Complex schema check requires sibling Complex and Bun")
    script = """
import { Device, RobotManifest } from './src/calibration/schema.ts';
import { missingActions } from './src/device/interface.ts';
const root = process.argv[1];
RobotManifest.parse(Bun.YAML.parse(await Bun.file(root + '/workspace.yaml').text()));
for await (const file of new Bun.Glob('devices/*/profile.yaml').scan(root)) {
  const p = Device.parse(Bun.YAML.parse(await Bun.file(root + '/' + file).text()));
  if (missingActions(p, p.access?.driver?.actions ?? []).length) throw Error(p.name);
}
"""
    subprocess.run(["bun", "-e", script, str(task.robot)], cwd=complex_root, check=True)


def test_go2_regeneration(tmp_path):
    python = os.environ.get("ROBOENV_ISAACLAB_MODEL_PYTHON")
    if not python:
        pytest.skip("Set ROBOENV_ISAACLAB_MODEL_PYTHON for USD regeneration")
    output = tmp_path / "catalog"
    regenerated = tmp_path / "facts.json"
    subprocess.run(
        [
            python,
            str(ROOT / "tools/generate_isaaclab_config.py"),
            "--out",
            str(output),
            "--facts-out",
            str(regenerated),
        ],
        cwd=ROOT,
        check=True,
    )
    assert json.loads(regenerated.read_text()) == facts()
    for path in output.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (ROOT / "catalog" / path.relative_to(output)).read_bytes()


def test_native_go2_in_prepared_interpreter(tmp_path):
    python = os.environ.get("ROBOENV_ISAACLAB_TEST_PYTHON")
    if not python:
        pytest.skip("Requires prepared Isaac Lab and an RTX GPU; opt in explicitly")
    for attempt in range(2):
        result = subprocess.run(
            [python, str(ROOT / "tests/isaaclab_native.py"), str(tmp_path / str(attempt))],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'runtime/src'}"},
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_go2_worker_public_projection_and_terminal_freeze(tmp_path):
    from roboenv_runtime.adapters.isaaclab import IsaacLabAdapter
    from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError

    adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
    adapter._env = World()
    adapter._env.held = 25
    adapter._env.close = lambda: None
    worker = SimWorker(
        adapter,
        task_id="isaaclab.go2-posture",
        seed=17,
        run_dir=tmp_path,
        device_root=ROOT / "catalog/robots/isaaclab-go2",
        reset_adapter=False,
    )
    try:
        description = worker.dispatch("environment.describe", {})
        assert set(description) == {"task", "devices", "session", "generation"}
        assert description["task"]["context"] == {}
        for action in ("call", "vla", "reset", "checkpoint"):
            with pytest.raises(WorkerRPCError, match="basic device"):
                worker.dispatch(action, {})
        result = worker.dispatch("submit", {})
        assert result["completed"] and result["reward"] == 1
        with pytest.raises(WorkerRPCError):
            worker.dispatch(
                "devices.call",
                {"device": "body", "action": "write", "args": {"target": facts()["posture"]}},
            )
        assert adapter._env.steps == 0
    finally:
        worker.close()


def test_contact_profiles_regenerate_and_export_with_typed_connections(tmp_path, rig):
    import runpy

    from roboenv.devices.export import export_devices

    generator = runpy.run_path(str(ROOT / "tools/generate_isaaclab_config.py"))
    generator["generate"](tmp_path, facts())
    for foot in FEET:
        relative = Path(f"robots/isaaclab-go2/devices/contact_{foot.lower()}")
        for filename in ("profile.yaml", "driver.py"):
            assert (tmp_path / relative / filename).read_bytes() == (
                ROOT / "catalog" / relative / filename
            ).read_bytes()
        profile = yaml.safe_load((tmp_path / relative / "profile.yaml").read_text())
        assert profile["type"] == "force_torque"
        assert "origin" not in profile["intrinsics"]["measurement"]
        assert profile["intrinsics"]["measurement"]["target"] == {"device": "body", "frame": foot}
    _, service = rig
    exported = tmp_path / "exported"
    descriptors = [d for d in service.inventory()["devices"] if d["type"] == "force_torque"]
    assert len(descriptors) == 4
    export_devices(descriptors, exported, {"endpoint": "http://localhost:8123"})
    for descriptor in descriptors:
        profile = yaml.safe_load((exported / descriptor["name"] / "profile.yaml").read_text())
        assert profile["access"]["connection"]["device"] == descriptor["name"]
        assert "kind" not in profile["access"]
        assert (exported / descriptor["name"] / "driver.py").read_bytes() == (
            ROOT / "src/roboenv/assets/force-torque-client.py"
        ).read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", "contact_total"),
        ("torque", {"axes": ["z"]}),
        ("target", {"device": "body", "frame": "FR_foot"}),
    ],
)
def test_contact_binding_rejects_false_measurement_claims(tmp_path, field, value):
    from roboenv_runtime.bindings.backends.isaaclab.devices import bind

    task = resolve_task("isaaclab/go2-posture", project=ROOT)
    spec = next(s for s in task.bindings.devices if s.entity == "FL_foot")
    profile = yaml.safe_load((task.robot / spec.profile).read_text())
    profile["intrinsics"]["measurement"][field] = value
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    context = BindingContext(
        "isaaclab",
        "go2",
        SimpleNamespace(_env=World()),
        devices={"body": "body", "FL_foot": spec.name},
    )
    with pytest.raises(ValueError, match="Contact measurement"):
        bind(replace(spec, profile="profile.yaml"), tmp_path, context)


def test_native_contact_read_rotates_force_without_fabricating_torque():
    from roboenv_runtime.adapters.isaaclab.backend import IsaacLabEnv

    class Tensor:
        def __init__(self, data):
            self.value = np.asarray(data)
            self.torch = self

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    env = IsaacLabEnv.__new__(IsaacLabEnv)
    env.steps = 3
    env.foot_ids = [0, 1, 2, 3]
    # Quaternion xyzw: foot rotated 180 degrees around world Z.
    poses = [[[0, 0, 0, 0, 0, 1, 0] for _ in FEET]]
    env.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=Tensor(poses)))
    env.scene = {
        foot: SimpleNamespace(data=SimpleNamespace(net_forces_w=Tensor([[[1, 2, 3]]])))
        for foot in FEET
    }
    for foot in FEET:
        reading = env.read_sensor(foot)
        assert reading["force"] == {"x": -1, "y": -2, "z": 3}
        assert reading["torque"] is None
    assert env.steps == 3


def test_imu_regeneration_export_and_portable_driver(tmp_path, rig):
    import runpy
    from unittest.mock import patch

    from roboenv.devices.export import export_devices

    generator = runpy.run_path(str(ROOT / "tools/generate_isaaclab_config.py"))
    generator["generate"](tmp_path, facts())
    relative = Path("robots/isaaclab-go2/devices/imu")
    for filename in ("profile.yaml", "driver.py"):
        assert (tmp_path / relative / filename).read_bytes() == (
            ROOT / "catalog" / relative / filename
        ).read_bytes()
    _, service = rig
    descriptor = service.describe("imu")
    measurement = descriptor["intrinsics"]["measurement"]
    assert measurement["linear_acceleration"]["quantity"] == "specific_force"
    assert measurement["orientation"] is None and measurement["magnetic_field"] is None
    assert "origin" not in measurement
    exported = tmp_path / "exported"
    export_devices([descriptor], exported, {"endpoint": "http://localhost:8123"})
    profile_path = exported / "imu/profile.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    assert profile["extrinsics"]["mount"]["translation_m"] == facts()["imu_position"]
    assert profile["access"]["connection"]["device"] == "imu"
    profile["access"]["connection"]["device"] = "remote-imu"
    profile_path.write_text(yaml.safe_dump(profile))
    driver = runpy.run_path(str(exported / "imu/driver.py"))
    requests = []

    def exchange(endpoint, method, path="/", body=None, destination=None):
        if method == "GET":
            return json.dumps({"protocol": "roboenv/1", "session": "s1", "generation": 0}).encode()
        request = json.loads(body)
        requests.append(request)
        return (
            json.dumps(
                {
                    **{k: request[k] for k in ("protocol", "session", "generation", "request_id")},
                    "ok": True,
                    "result": {"data": call(service, "imu")},
                }
            )
            + "\n"
        ).encode()

    with patch.dict(driver["remote"].__globals__, exchange=exchange):
        result = driver["handle"]("read", {})
        assert result["linear_acceleration"]["z"] == 9.81
        assert result["orientation"] is None
        assert requests[0]["args"]["device"] == "remote-imu"
        assert driver["handle"]("write", {})["ok"] is False
        assert len(requests) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"linear_acceleration": {"quantity": "gravity_removed", "axes": ["x", "y", "z"]}},
        {"orientation": {"reference": {"kind": "frame", "frame": "world"}}},
        {"magnetic_field": {"axes": ["x", "y", "z"]}},
        {"angular_velocity": None},
    ],
)
def test_imu_binding_rejects_measurements_the_native_device_does_not_supply(tmp_path, change):
    from roboenv_runtime.bindings.backends.isaaclab.devices import bind

    task = resolve_task("isaaclab/go2-posture", project=ROOT)
    spec = next(s for s in task.bindings.devices if s.entity == "imu")
    profile = yaml.safe_load((task.robot / spec.profile).read_text())
    profile["intrinsics"]["measurement"].update(change)
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
    context = BindingContext(
        "isaaclab", "go2", SimpleNamespace(_env=World()), devices={"body": "body", "imu": spec.name}
    )
    with pytest.raises(ValueError, match="IMU must declare"):
        bind(replace(spec, profile="profile.yaml"), tmp_path, context)


def test_native_imu_read_preserves_specific_force_and_unsupported_channels():
    from roboenv_runtime.adapters.isaaclab.backend import IsaacLabEnv

    class Tensor:
        def __init__(self, data):
            self.value = np.asarray(data)
            self.torch = self

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def cpu(self):
            return self

        def tolist(self):
            return self.value.tolist()

    env = IsaacLabEnv.__new__(IsaacLabEnv)
    env.steps = 3
    env.scene = {
        "imu": SimpleNamespace(
            data=SimpleNamespace(
                ang_vel_b=Tensor([[0.1, -0.2, 0.3]]),
                lin_acc_b=Tensor([[0, 0, 9.81]]),
            )
        )
    }
    reading = env.read_sensor("imu")
    assert reading["angular_velocity"] == {"x": 0.1, "y": -0.2, "z": 0.3}
    assert reading["linear_acceleration"] == {"x": 0, "y": 0, "z": 9.81}
    assert reading["orientation"] is None and reading["magnetic_field"] is None
    assert env.steps == 3
