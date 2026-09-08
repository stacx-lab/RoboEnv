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
                {"angular_velocity": [0, 0, 0], "linear_acceleration": [0, 0, 9.81]}
                if entity == "imu"
                else {"force": [0, 0, 20]}
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
    assert call(service, "imu")["linear_acceleration"] == [0, 0, 9.81]
    for foot in FEET:
        assert call(service, f"contact_{foot.lower()}")["force"] == [0, 0, 20]
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
        assert driver.read_bytes() == (ROOT / "src/roboenv/assets/device-client.py").read_bytes()
        access = yaml.safe_load(driver.with_name("profile.yaml").read_text())["access"]
        assert access["url"] == "http://127.0.0.1:8123"
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
