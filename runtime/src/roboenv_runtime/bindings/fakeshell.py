"""Native reference binding: existing puck entity, no Client/tools()/primitive RPC."""

import math

from roboenv.devices.model import Action, BoundDevice, DeviceError

from roboenv_runtime.adapters.fakeshell import FakeShell


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise DeviceError("invalid_args", "position/displacement must be a finite number")
    return float(value)


def bind_puck(spec, root, context):
    native = context.native
    if context.backend != "fakeshell" or not isinstance(native, FakeShell):
        raise ValueError("fakeshell.puck/1 requires a fakeshell world")
    if spec.entity != "puck":
        raise ValueError(f"unknown fakeshell entity: {spec.entity!r}")
    if spec.config:
        raise ValueError("fakeshell.puck/1 has no binding-specific configuration")
    definition = spec.definition(root)
    # Published profiles must satisfy Kinex, whose "other" devices have no action
    # space intrinsics; the write schema below declares the target instead.
    if "actions" in definition.intrinsics:
        raise ValueError("fakeshell puck declares its action space in the write schema")

    def read(args, execution):
        if args:
            raise DeviceError("invalid_args", "puck.read takes no arguments")
        return execution.perform(
            "binding.read",
            {"entity": spec.entity},
            lambda: {
                "result": {
                    "objects": {
                        "puck": {
                            "position": list(native.pos),
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                    "budget": native.budget().as_dict(),
                }
            },
        )

    def write(args, execution):
        target = args.get("target")
        if set(args) != {"target"} or not isinstance(target, list) or len(target) != 1:
            raise DeviceError("invalid_args", "puck.write requires target: [x_metres]")
        x = _number(target[0])
        execution.perform(
            "binding.write",
            {"entity": spec.entity, "target": [x]},
            lambda: native.push(x - native.pos[0]),
            mutating=True,
        )
        return {"position": list(native.pos)}

    def status(args, execution):
        if args:
            raise DeviceError("invalid_args", "status takes no arguments")
        return {"loaded": True, "entity": spec.entity}

    empty = {"type": "object", "additionalProperties": False}
    actions = {
        "read": Action(read, empty),
        "state": Action(read, empty),
        "status": Action(status, empty),
        "load": Action(status, empty),
        "write": Action(
            write,
            {
                "type": "object",
                "required": ["target"],
                "properties": {
                    "target": {
                        "type": "array",
                        "description": "Absolute puck x position in metres",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "number"},
                    }
                },
                "additionalProperties": False,
            },
        ),
        "stop": Action(reason="Discrete synchronous puck motion has no interrupt controller"),
        # Kinex requires load/unload/status on every driver; neither changes the world.
        "unload": Action(lambda args, execution: {"unloaded": True}, empty),
    }
    return BoundDevice(
        definition,
        actions,
        implementation="native-entity-binding",
        validation="entity-resolved; synthetic CPU backend, not physical hardware",
        binding={"id": spec.binding, "entity": spec.entity, "backend": context.backend},
    )
