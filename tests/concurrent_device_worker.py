"""Real CPU worker fixture with two independent bounded device controllers."""

import os
import threading

from roboenv.devices.execution import map_result
from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.bindings import assembly, default_registry
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.worker import main

_original_init = FakeShell.__init__


def initialize(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.owner = threading.get_ident()
    self.native_threads = set()
    self.positions = {name: [0.0] for name in ("left", "right")}
    self.control = DeviceControl(
        {name: [(-2, 2)] for name in self.positions},
        self.positions,
        self.apply,
        lambda: None,
        realtime=True,
        exhausted=lambda: False,
        interval_s=0.02,
    )


def apply(self, targets):
    self.native_threads.add(threading.get_ident())
    self.positions = {name: list(value) for name, value in targets.items()}


def bind(spec, root, context):
    adapter = context.native
    name = spec.name

    def read():
        adapter.native_threads.add(threading.get_ident())
        return {
            "positions": adapter.positions[name],
            "velocities": [0],
            "efforts": [],
            "faults": [],
            "sample_time_s": adapter.control.frame * 0.02,
            "commands": [
                entry for entry in adapter.control.commands if name in entry.get("commands", {})
            ],
            "owner_only": adapter.native_threads <= {adapter.owner},
        }

    def write(args, execution):
        result = execution.perform(
            "fixture.write",
            args,
            lambda: adapter.control.write(name, args["target"], for_s=args["for_s"]),
            mutating=True,
        )
        return map_result(result, lambda value: {**value, "reached": False, "state": read()})

    def stop(args, execution):
        return execution.perform(
            "fixture.stop",
            args,
            lambda: adapter.control.stop(name, adapter.positions[name]),
            mutating=True,
        )

    return BoundDevice(
        spec.definition(root),
        {
            "read": Action(lambda args, execution: read()),
            "load": Action(lambda args, execution: {"loaded": True}),
            "unload": Action(lambda args, execution: {"unloaded": True}),
            "status": Action(lambda args, execution: read()),
            "write": Action(write),
            "stop": Action(stop),
        },
        implementation="concurrency-fixture",
        binding={"entity": name},
    )


def registry():
    value = default_registry()
    value.register("test.concurrent/1", bind)
    return value


if __name__ == "__main__":
    FakeShell.__init__ = initialize
    FakeShell.apply = apply
    FakeShell.realtime_interval = 0.02
    FakeShell.realtime_tick = lambda self: self.control.tick()
    FakeShell.realtime_on_main_thread = os.environ.get("TEST_OWNER_THREAD") == "1"
    FakeShell.budget = lambda self: Budget(used=self.control.frame, max=None)
    assembly.default_registry = registry
    raise SystemExit(main())
