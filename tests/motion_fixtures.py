"""CPU devices using real buffer scheduling and world ownership."""

from types import SimpleNamespace

from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition
from roboenv.devices.service import DeviceService
from roboenv_runtime.bindings.motion import control_binding
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.world import WorldSession


class Engine:
    name = embodiment = "fixture"
    control_interval_s = 0.01
    owner_thread_required = True

    def __init__(self):
        self.frame = 0
        self.maximum = 1000
        self.stop_failure = False
        self.positions = {name: 0.0 for name in ("left", "right")}
        self.targets = dict(self.positions)
        self.events = []
        self.fault = None

    def configure_execution_mode(self, mode):
        self.execution_mode = mode

    def budget(self):
        return Budget(used=self.frame, max=self.maximum, unit="action_frames")

    def advance_control(self):
        self.frame += 1
        for name, target in self.targets.items():
            self.positions[name] += max(-0.01, min(0.01, target - self.positions[name]))

    def close(self):
        pass


def bind_devices(native, world, *, kind="joint", control=None):
    period = native.control_interval_s

    @control_binding
    def bind(spec, root, context):
        name = spec.entity
        facts = {
            "model": {
                "joints": [{"name": "j", "type": "revolute", "limits": {"min": -2, "max": 2}}]
            },
            "control": control or {"mode": "position"},
            "feedback": {"velocities": True},
            "stop": {"behavior": "hold", "affected_devices": [name]},
        }

        def write(args, _):
            if getattr(native, "write_error", None) is not None:
                raise native.write_error
            native.targets[name] = args["target"][0]
            native.events.append((native.frame, name, "write", native.targets[name]))

        def read(args, _):
            return {
                "positions": [native.positions[name]],
                "velocities": [0],
                "efforts": [],
                "faults": [{"code": "fault"}] if native.fault == name else [],
                "sample_time_s": native.frame * period,
            }

        def stop(args, _):
            if native.stop_failure:
                raise RuntimeError("broken stop")
            native.targets[name] = native.positions[name]
            native.events.append((native.frame, name, "stop", native.targets[name]))
            return {"stopped": True, **facts["stop"]}

        return BoundDevice(
            DeviceDefinition(name, kind, facts),
            {
                "write": Action(write),
                "read": Action(read),
                "stop": Action(stop),
            },
            implementation="fixture",
            binding={"entity": name},
        )

    return [
        bind(
            SimpleNamespace(entity=name),
            None,
            BindingContext("fixture", "fixture", native, world=world),
        )
        for name in native.positions
    ]


def fixture(mode="stepped", *, period=0.01, kind="joint", control=None):
    native = Engine()
    native.control_interval_s = period
    world = WorldSession(native, mode)
    service = DeviceService(
        bind_devices(native, world, kind=kind, control=control),
        execution_lock=world.lock,
        motion_scheduler=world.schedule,
        resolve=world.resolve,
        flush=world.poll_motions,
    )
    return native, world, service


def submit(service, name, action="write", **args):
    return service.submit({"device": name, "action": action, "args": args})
