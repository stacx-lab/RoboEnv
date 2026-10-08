"""Typed YAM controls; public extrinsics are calibration data, not simulator poses."""

from pathlib import Path

from roboenv.devices.model import Action, BoundDevice

from roboenv_runtime.bindings.motion import control_binding

from ..shared.profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


@control_binding
def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "yam":
        raise ValueError("Isaac YAM bindings require the YAM world")
    if spec.config or spec.entity != spec.name or context.devices.get(spec.entity) != spec.name:
        raise ValueError("Isaac YAM requires canonical unique devices")
    world = context.native.environment
    definition = spec.definition(root)
    expected = world.profiles.get(spec.name)
    if expected is None or expected["type"] != definition.type:
        raise ValueError("Unknown YAM device")
    validate(definition.intrinsics, expected["intrinsics"], spec.name + ".intrinsics")
    # Do not compare extrinsics with owner geometry or substitute solved mounts.
    actions = {}

    def register(name, fn, schema=EMPTY, mutation=False, preflight=None):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "isaaclab.yam." + name, args, lambda: fn(**args), mutating=mutation
            ),
            schema,
            validate=preflight,
        )

    if definition.type == "world":
        return BoundDevice(definition, {}, implementation="World reference", binding=spec.as_dict())
    register("load", lambda: {"loaded": True})
    register("unload", lambda: {"unloaded": True})
    register("status", lambda: {"loaded": True})
    if definition.type == "rgb":
        register(
            "read", lambda: world.read_camera(spec.name, Path(context.native.run_dir) / "captures")
        )
    else:
        register("read", lambda: world.read_device(spec.name))
        if definition.type in {"joint", "gripper"}:
            register("stop", lambda: world.stop(spec.name), mutation=True)
            count = 6 if definition.type == "joint" else 1
            schema = {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["target"],
                "additionalProperties": False,
            }
            register(
                "write",
                lambda target: world.set_target(spec.name, target),
                schema,
                True,
                lambda args: world.validate_target(spec.name, args["target"]),
            )
    return BoundDevice(
        definition, actions, implementation="Isaac Lab YAM practice", binding=spec.as_dict()
    )
