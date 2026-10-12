"""Separation, assembly, lifecycle and wire-compatibility tests. No model/GPU calls."""

import ast
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
from roboenv.tasks.resolve import resolve_task
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


def test_worker_binds_native_devices_without_executing_public_drivers(robot, tmp_path):
    (robot / "devices/controller/driver.py").write_text("raise AssertionError('must not import')")
    worker = SimWorker(
        FakeShell(), task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot
    )
    try:
        assert worker.devices._lock is worker.world.lock
        result = worker.dispatch(
            "devices.call", {"device": "controller", "action": "write", "args": {"target": [0.3]}}
        )
        assert result["data"]["position"][0] == pytest.approx(0.3)
        assert result["budget"]["used"] == 50
        assert [entry["op"] for entry in result["calls"]] == ["binding.write"]
        assert worker.dispatch("score.peek", {})["completed"]
        worker.dispatch("submit", {})
        with pytest.raises(WorkerRPCError, match="frozen"):
            worker.dispatch(
                "devices.call", {"device": "controller", "action": "write", "args": {"target": [0]}}
            )
    finally:
        worker.close()


@pytest.mark.parametrize("native_goal", ["Move the cylinder left of the car.", "", None])
def test_describe_keeps_catalog_fallback_and_sampled_context(
    robot, tmp_path, monkeypatch, native_goal
):
    adapter = FakeShell()
    monkeypatch.setattr(adapter, "goal", lambda: native_goal)
    context = {"objects": {"A": {"description": "yellow cylinder"},
                           "B": {"description": "teal car"}}}
    monkeypatch.setattr(adapter, "task_context", lambda: context)
    worker = SimWorker(adapter, task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot)
    catalog_goal = "Place object A to the left of object B."
    worker.devices.manifest["task"] = {"id": "test", "instruction": catalog_goal}
    try:
        for _ in range(2):
            task = worker.dispatch("environment.describe", {})["task"]
            assert task["instruction"] == (native_goal or catalog_goal)
            assert task["native_instruction"] == (native_goal or "")
            assert task["context"] == context
        assert adapter.budget().used == 0
        assert worker.devices.manifest["task"]["instruction"] == catalog_goal
    finally:
        worker.close()


def test_reset_rebinds_devices_but_preserves_sequence(robot, tmp_path):
    worker = SimWorker(
        FakeShell(), task_id="test", seed=0, run_dir=tmp_path / "run", device_root=robot
    )
    command = {"device": "controller", "action": "write", "args": {"target": [0.1]}}
    try:
        assert worker.adapter.describe() == {"name": "fakeshell", "embodiment": "general"}
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
        continued = worker.dispatch("devices.call", command)
        assert continued["sequence"] == result["sequence"] + 1
        assert worker.adapter.pos[0] == 0.1
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


@pytest.mark.parametrize("reference", ["toy-puck", "fakeshell/toy-puck"])
def test_task_resolution_uses_explicit_bindings_and_only_selected_devices(
    robot,
    tmp_path,
    reference,
):
    catalog = tmp_path / "catalog"
    target = catalog / "robots" / reference
    shutil.copytree(robot, target)
    # Native bindings do not need a Python driver in the hardware definition.
    (target / "devices/controller/driver.py").unlink()
    value = yaml.safe_load((ROOT / "tests/fixtures/native-toy/task.yaml").read_text())
    value["robot"] = reference
    devices = {
        "puck": {
            "binding": "fakeshell.puck/1",
            "entity": "puck",
            "profile": "devices/controller/profile.yaml",
            "overrides": {"extrinsics": {"mount": {"parent": "world"}}},
        }
    }
    (target / "bindings.yaml").write_text(yaml.safe_dump({
        "schema_version": "roboenv/bindings/1", "backend": "fakeshell", "devices": devices,
    }))
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(value))
    resolved = resolve_task(task, catalog=catalog)
    assert [spec.name for spec in resolved.bindings.devices] == ["puck"]
    from roboenv_runtime.bindings import default_registry

    world = FakeShell()
    service = default_registry().assemble(
        resolved.bindings, resolved.robot, BindingContext("fakeshell", "general", world)
    )
    try:
        assert service.describe("puck")["extrinsics"]["mount"]["parent"] == "world"
        with pytest.raises(DeviceError, match="unknown device"):
            service.describe("controller")
    finally:
        service.close()


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
    assert descriptor["type"] == "other"
    args = {"target": [0.3]}
    first = client.device("controller").call("write", args, request_id="native-write")
    assert client.device("controller").call("write", args, request_id="native-write") == first
    assert first["budget"]["used"] == 50
    assert first["calls"][0]["op"] == "binding.write"
    assert client.request("score.peek")["completed"]


def test_native_actions_share_one_execution_lane(robot, tmp_path):
    class SlowWorld(FakeShell):
        active, peak = 0, 0
        entered = threading.Event()

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
                    "action": "write",
                    "args": {"target": [0.1]},
                },
            )
            assert native.entered.wait(2)
            two = pool.submit(
                worker.dispatch,
                "devices.call",
                {
                    "device": "controller",
                    "action": "write",
                    "args": {"target": [0.2]},
                },
            )
            one.result(timeout=3)
            two.result(timeout=3)
        assert native.peak == 1
        assert native.pos[0] == pytest.approx(0.2)
        assert native.used == 100
    finally:
        worker.close()
