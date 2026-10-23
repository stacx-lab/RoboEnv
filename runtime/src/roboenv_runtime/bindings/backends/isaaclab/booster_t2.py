"""T2 devices use the existing owner clock and buffered control contract."""

from pathlib import Path

from roboenv.devices.model import Action, BoundDevice
from roboenv.devices.velocity import command_schema
from roboenv_runtime.bindings.motion import control_binding

from ..shared.profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


@control_binding
def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "booster_t2":
        raise ValueError("T2 binding requires a T2 world")
    world = context.native.environment
    definition = spec.definition(root)
    if spec.config or spec.entity != spec.name or context.devices.get(spec.entity) != spec.name:
        raise ValueError("T2 requires canonical unique devices")
    expected = world.profiles.get(spec.name)
    if expected is None or expected["type"] != definition.type:
        raise ValueError("Unknown T2 device")
    for section in ("intrinsics", "extrinsics"):
        validate(getattr(definition, section), expected.get(section, {}), spec.name + "." + section)
    actions = {}

    def register(name, fn, schema=EMPTY, mutation=False, preflight=None):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "isaaclab.booster_t2." + name, args, lambda: fn(**args), mutating=mutation
            ), schema, validate=preflight,
        )

    if spec.name == "world":
        return BoundDevice(definition, {}, implementation="World reference", binding=spec.as_dict())
    register("load", lambda: {"loaded": True})
    register("unload", lambda: {"unloaded": True})
    register("status", lambda: {"loaded": True})
    if spec.name == "head_camera":
        register("read", lambda: world.read_camera(spec.name, Path(context.native.run_dir) / "captures"))
    else:
        register("read", lambda: world.read_device(spec.name))
        register("stop", lambda: world.stop(spec.name), mutation=True)
        count = len(world.robot.joint_groups.get(spec.name, ()))
        schema = (command_schema(expected["intrinsics"]["control"]) if spec.name == "base" else {
            "type": "object", "properties": {"target": {"type": "array",
                "items": {"type": "number"}, "minItems": count, "maxItems": count}},
            "required": ["target"], "additionalProperties": False,
        })
        register("write", lambda **args: world.set_target(spec.name, args), schema, True,
                 lambda args: world._validate(spec.name, args))
    return BoundDevice(definition, actions, implementation="T2 whole-body and hand control",
                       binding=spec.as_dict())
