"""Separation, assembly, lifecycle and wire-compatibility tests. No model/GPU calls."""

import ast
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from roboenv.devices.bindings.registry import BindingContext, BindingRegistry
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.model import Action, BoundDevice, DeviceError
from roboenv.harbor.export import export_task
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def robot(tmp_path):
    source = ROOT / "tests/fixtures/native-toy/catalog/robots/toy-puck"
    target = tmp_path / "robot"
    shutil.copytree(source, target)
    return target


def test_neutral_core_has_no_simulator_client_or_binding_dependency():
    for name in ("model.py", "service.py"):
        tree = ast.parse((ROOT / "src/roboenv/devices" / name).read_text())
        modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(
            part in module
            for module in modules
            for part in ("runtime", "adapters", "bindings", "client", "runpy")
        )


def test_one_device_definition_binds_to_two_different_backend_representations(robot):
    original = (robot / "devices/controller/profile.yaml").read_bytes()
    registry = BindingRegistry()

    def factory(scale):
        def bind(spec, root, world):
            def read(args, execution):
                return {"x": world.native[spec.entity] / scale}

            def write(args, execution):
                execution.perform(
                    "native.write",
                    args,
                    lambda: world.native.__setitem__(spec.entity, args["x"] * scale),
                    mutating=True,
                )
                return read({}, execution)

            return BoundDevice(
                spec.definition(root),
                {"read": Action(read), "write": Action(write)},
                implementation=world.backend,
            )

        return bind

    registry.register("metres/1", factory(1))
    registry.register("millimetres/1", factory(1000))
    results = []
    for backend, scale in (("metres", 1), ("millimetres", 1000)):
        plan = BindingPlan.load(
            robot,
            backend=backend,
            devices={
                "motor": {
                    "binding": backend + "/1",
                    "entity": "robot0",
                    "profile": "devices/controller/profile.yaml",
                    "overrides": {"extrinsics": {"mount": {"parent": "body"}}},
                }
            },
        )
        native = {"robot0": 0}
        service = registry.assemble(plan, robot, BindingContext(backend, "test", native))
        result = service.call({"device": "motor", "action": "write", "args": {"x": 0.3}})
        assert native["robot0"] == pytest.approx(0.3 * scale)
        assert service.describe("motor")["extrinsics"]["mount"]["parent"] == "body"
        results.append(result["data"])
        service.close()
        assert native["robot0"] == pytest.approx(0.3 * scale)  # device close never resets world
    assert results[0] == results[1]
    assert (robot / "devices/controller/profile.yaml").read_bytes() == original


def test_native_device_works_without_primitive_registry_or_legacy_driver(robot, tmp_path):
    class NativeOnly(FakeShell):
        def tools(self):
            return {}

    (robot / "devices/controller/driver.py").write_text("raise AssertionError('must not import')")
    worker = SimWorker(
        NativeOnly(), task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot
    )
    try:
        assert worker.api()["tools"] == {}
        assert worker.devices._lock is worker.world.lock
        result = worker.dispatch(
            "devices.call", {"device": "controller", "action": "write", "args": {"target": [0.3]}}
        )
        assert result["data"]["position"][0] == pytest.approx(0.3)
        assert result["budget"]["used"] == 50
        assert [entry["op"] for entry in result["calls"]] == ["binding.write"]
        assert worker.basic_call_count == 0
        assert worker.dispatch("score.peek", {})["completed"]
        worker.dispatch("submit", {})
        with pytest.raises(WorkerRPCError, match="frozen"):
            worker.dispatch(
                "devices.call", {"device": "controller", "action": "write", "args": {"target": [0]}}
            )
    finally:
        worker.close()


def test_reset_and_restore_rebind_devices_but_preserve_sequence(robot, tmp_path):
    worker = SimWorker(
        FakeShell(), task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot
    )
    command = {"device": "controller", "action": "write", "args": {"target": [0.1]}}
    try:
        first = worker.devices
        result = worker.dispatch("devices.call", command)
        closed_positions = []
        first._devices["controller"]._on_close = lambda: closed_positions.append(
            worker.adapter.pos[0]
        )
        worker.dispatch("reset", {"seed": 1})
        assert closed_positions == [0.1]  # release handles before native reset destroys entities
        assert first is not worker.devices
        with pytest.raises(DeviceError, match="closed"):
            first.call(command)
        assert worker.adapter.pos[0] == 0
        assert worker.devices.sequence == result["sequence"]
        token = worker.dispatch("checkpoint", {})
        worker.dispatch("devices.call", command)
        before_restore = worker.devices
        before_restore._devices["controller"]._on_close = lambda: closed_positions.append(
            worker.adapter.pos[0]
        )
        worker.dispatch("restore", {"token": token})
        assert closed_positions == [0.1, 0.1]
        assert before_restore is not worker.devices
        assert worker.adapter.pos[0] == 0
        assert worker.devices.sequence == result["sequence"] + 1
    finally:
        worker.close()


@pytest.mark.parametrize("change", ["backend", "embodiment", "provider", "entity"])
def test_binding_mismatch_is_a_boot_failure_and_releases_world(robot, tmp_path, change):
    path = robot / "bindings.yaml"
    plan = yaml.safe_load(path.read_text())
    if change in {"backend", "embodiment"}:
        plan[change] = "wrong"
    else:
        plan["devices"]["controller"]["binding" if change == "provider" else "entity"] = "wrong"
    path.write_text(yaml.safe_dump(plan))

    class World(FakeShell):
        closed = False

        def close(self):
            self.closed = True

    world = World()
    with pytest.raises(ValueError):
        SimWorker(world, task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot)
    assert world.closed


def test_registry_cleanup_on_partial_assembly_failure(robot):
    registry = BindingRegistry()
    closed = []

    def good(spec, root, context):
        return BoundDevice(
            spec.definition(root),
            {"read": Action(lambda args, ctx: {})},
            implementation="test",
            on_close=lambda: closed.append(spec.name),
        )

    def bad(*args):
        raise RuntimeError("connection failed")

    registry.register("good/1", good)
    registry.register("bad/1", bad)
    plan = BindingPlan.load(
        robot,
        backend="test",
        devices={
            name: {
                "binding": binding,
                "entity": "puck",
                "profile": "devices/controller/profile.yaml",
            }
            for name, binding in (("a", "good/1"), ("b", "bad/1"))
        },
    )
    with pytest.raises(RuntimeError, match="connection failed"):
        registry.assemble(plan, robot, BindingContext("test", "test", None))
    assert closed == ["a"]


def test_missing_binding_plan_does_not_execute_catalog_drivers(robot):
    (robot / "bindings.yaml").unlink()
    with pytest.raises(ValueError, match="explicit bindings.yaml"):
        BindingPlan.load(robot, backend="fakeshell")


def test_catalog_execution_is_not_a_registered_backend(robot):
    from roboenv_runtime.bindings import default_registry

    plan = BindingPlan.load(
        robot,
        backend="fakeshell",
        devices={
            "controller": {"binding": "catalog/1", "entity": "controller"},
        },
    )
    with pytest.raises(ValueError, match="unknown registered binding"):
        default_registry().assemble(
            plan, robot, BindingContext("fakeshell", "general", FakeShell())
        )


def test_native_read_context_refuses_effectful_backend_operation(robot):
    registry = BindingRegistry()
    moved = []

    def factory(spec, root, context):
        return BoundDevice(
            spec.definition(root),
            {
                "read": Action(
                    lambda args, ctx: ctx.perform(
                        "native.move", {}, lambda: moved.append(True), mutating=True
                    )
                )
            },
            implementation="test",
        )

    registry.register("bad-read/1", factory)
    plan = BindingPlan.load(
        robot,
        backend="test",
        devices={
            "sensor": {
                "binding": "bad-read/1",
                "entity": "puck",
                "profile": "devices/controller/profile.yaml",
            }
        },
    )
    service = registry.assemble(plan, robot, BindingContext("test", "test", None))
    with pytest.raises(DeviceError, match="mutating"):
        service.call({"device": "sensor", "action": "read", "args": {}})
    assert not moved


def test_task_export_freezes_explicit_bindings_and_only_selected_client_devices(robot, tmp_path):
    catalog = tmp_path / "catalog"
    shutil.copytree(robot, catalog / "robots/toy-puck")
    # Native bindings do not need a Python driver in the hardware definition.
    (catalog / "robots/toy-puck/devices/controller/driver.py").unlink()
    value = yaml.safe_load((ROOT / "tests/fixtures/native-toy/task.yaml").read_text())
    value["device_bindings"] = {
        "puck": {
            "binding": "fakeshell.puck/1",
            "entity": "puck",
            "profile": "devices/controller/profile.yaml",
            "overrides": {"extrinsics": {"mount": {"parent": "world"}}},
        }
    }
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(value))
    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(b"test input only")
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "fakeshell",
                "legacy_tree_required": False,
                "image": "sha256:" + "1" * 64,
            }
        )
    )
    out = tmp_path / "export"
    export_task(task, catalog, out, wheel, deployment_path=deployment)
    base = out / "environment"
    bound = yaml.safe_load((base / "sim/robot/bindings.yaml").read_text())
    assert set(bound["devices"]) == {"puck"}
    public = base / "definition/robot"
    assert not (public / "bindings.yaml").exists()
    assert not (public / "devices/controller").exists()
    assert "from roboenv" not in (public / "devices/puck/driver.py").read_text()
    assert yaml.safe_load((public / "devices/puck/profile.yaml").read_text())["name"] == "puck"


@pytest.mark.parametrize(
    "invalid", [{"profile": "../../escape.yaml"}, {"overrides": {"name": "other"}}, {"config": []}]
)
def test_unsafe_or_malformed_binding_inputs_fail(robot, invalid):
    with pytest.raises((ValueError, FileNotFoundError)):
        BindingPlan.load(
            robot,
            backend="fakeshell",
            devices={
                "puck": {
                    "binding": "fakeshell.puck/1",
                    "entity": "puck",
                    "profile": "devices/controller/profile.yaml",
                    **invalid,
                }
            },
        )


def test_duplicate_binding_yaml_is_rejected(robot):
    (robot / "bindings.yaml").write_text(
        "schema_version: roboenv/bindings/1\nbackend: fakeshell\nbackend: other\ndevices: {}\n"
    )
    with pytest.raises(ValueError, match="unique"):
        BindingPlan.load(robot, backend="fakeshell")


@pytest.mark.parametrize("device_service", ["native"], indirect=True)
def test_native_binding_over_real_socket_preserves_wire_and_dedup(device_service):
    client, _, _ = device_service
    descriptor = client.device("controller").describe()
    assert descriptor["implementation"] == "native-entity-binding"
    assert descriptor["control_mode"] == "position-1d"
    args = {"target": [0.3]}
    first = client.device("controller").call("write", args, request_id="native-write")
    assert client.device("controller").call("write", args, request_id="native-write") == first
    assert first["budget"]["used"] == 50
    assert first["calls"][0]["op"] == "binding.write"
    assert client.request("score.peek")["completed"]


def test_native_and_legacy_actions_share_one_execution_lane(robot, tmp_path):
    class SlowWorld(FakeShell):
        active, peak = 0, 0
        entered = threading.Event()

        def tools(self):
            return {**super().tools(), "push": self.push}

        def push(self, dx=0.1):
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.set()
            time.sleep(0.05)
            super().push(dx)
            self.active -= 1

    native = SlowWorld()
    worker = SimWorker(native, task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot)
    try:
        with ThreadPoolExecutor() as pool:
            one = pool.submit(
                worker.dispatch,
                "devices.call",
                {
                    "device": "controller",
                    "action": "invoke",
                    "args": {"tool_name": "push", "kwargs": {"dx": 0.1}},
                },
            )
            assert native.entered.wait(2)
            two = pool.submit(worker.dispatch, "call", {"name": "push", "kwargs": {"dx": 0.1}})
            one.result(timeout=3)
            two.result(timeout=3)
        assert native.peak == 1
        assert native.pos[0] == pytest.approx(0.2)
        assert native.used == 100
    finally:
        worker.close()
