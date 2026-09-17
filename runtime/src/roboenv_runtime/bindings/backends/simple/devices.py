"""Device bindings for the shared G1 scene; the worker serializes all calls."""

from pathlib import Path

from roboenv.devices.velocity import command_schema

from roboenv.devices.model import Action, BoundDevice
from ..isaaclab.profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def bind(spec, root, context, *, kind):
    if context.backend != "simple" or context.embodiment != "g1":
        raise ValueError("SIMPLE requires the G1 embodiment")
    world = context.native._env
    control = world.control
    definition = spec.definition(root)
    if spec.config or spec.entity != spec.name or context.devices.get(spec.entity) != spec.name:
        raise ValueError("G1 requires canonical, unique component names")
    expected = control.profiles.get(spec.name)
    if (
        expected is None
        or definition.type != expected["type"]
        or definition.type
        not in {
            "joint": {"joint", "dexterous_hand"},
            "velocity": {"velocity"},
            "camera": {"rgb"},
            "world": {"world"},
        }[kind]
    ):
        raise ValueError("Unknown G1 component")
    for section in ("intrinsics", "extrinsics"):
        validate(getattr(definition, section), expected.get(section, {}), spec.name + "." + section)
    actions = {}

    def register(name, fn, schema=EMPTY, mutation=False):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "simple.g1." + name, args, lambda: fn(**args), mutating=mutation
            ),
            schema,
        )

    if spec.name == "world":
        return BoundDevice(
            definition, {}, implementation="G1 world reference", binding=spec.as_dict()
        )
    register("load", lambda: {"loaded": True})
    register("status", lambda: {"loaded": True})
    register("unload", lambda: {"unloaded": True})
    if spec.name in world.model["cameras"]:
        register(
            "read", lambda: world.read_camera(spec.name, Path(context.native.run_dir) / "captures")
        )
        return BoundDevice(
            definition, actions, implementation="G1 RGB camera", binding=spec.as_dict()
        )
    register("read", lambda: world.read_device(spec.name))
    register("stop", lambda: control.stop(spec.name), mutation=True)
    register(
        "write",
        lambda **args: control.write(spec.name, args),
        command_schema(control.profiles["base"]["intrinsics"]["control"])
        if spec.name == "base"
        else {
            "type": "object",
            "properties": {"target": {"type": "array", "items": {"type": "number"}}},
            "required": ["target"],
            "additionalProperties": False,
        },
        mutation=True,
    )
    return BoundDevice(
        definition,
        actions,
        implementation="G1 joint-control practice interface",
        binding=spec.as_dict(),
    )
